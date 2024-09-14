import torch
from .trainer import CustomSeq2SeqTrainer


from tqdm import tqdm
def compute_metrics(eval_pred, scores=None):
    def find_last(pred: str) -> str:
        for word in reversed(pred.replace(':', ' ').split()):
            if word.lower() in ['yes', 'no']:
                return word.lower()

    def calc_auroc(preds_score, labels: list[int]) -> float:
        from sklearn.metrics import roc_auc_score
        try:
            auroc = roc_auc_score(labels, preds_score)
            rank0_print(f'AUROC: {auroc}')
            return auroc
        except Exception as e:
            logging.error(f'Failed to calculate AUROC: {e}')
            return 0.5

    tqdm.write(f'sample output: {tokenizer.decode(eval_pred.predictions[0], skip_special_tokens=False)}')
    tqdm.write(f'sample label: {tokenizer.decode([x for x in eval_pred.label_ids[0] if x > 0], skip_special_tokens=False)}')
    total, correct = 0, 0
    labels_for_auroc = []
    for pred, label in zip(eval_pred.predictions, eval_pred.label_ids):
        pred = tokenizer.decode(pred, skip_special_tokens=True)
        label = tokenizer.decode([x for x in label if x > 0], skip_special_tokens=True)
        label = find_last(label)
        labels_for_auroc.append(1 if label == 'yes' else 0)
        try:
            correct += int(find_last(pred) == label)
            total += 1
        except:
            tqdm.write(f'Failed to evaluate sample:\n'
                       f'Prediction: {pred}\n'
                       f'Label: {label}')
    rank0_print(f'acc {correct / total if total else 0.}')
    r = {'acc': (correct / total if total else 0.)}
    if scores: r['auroc'] = calc_auroc(scores, labels_for_auroc)
    return r


@torch.inference_mode()
def generate_eval(self: CustomSeq2SeqTrainer,
    eval_dataset: Optional[Dataset] = None,
    ignore_keys: Optional[List[str]] = None,
    metric_key_prefix: str = "eval",
) -> Dict[str, float]:
    eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
    eval_dataloader = self.get_eval_dataloader(eval_dataset)
    yes_no_tokens = [i for i in range(self.tokenizer.vocab_size) if self.tokenizer.decode(i).lower().strip() in ['yes', 'no']]  # all possible class tokens
    yes_no_tokens_tensor = torch.tensor(yes_no_tokens, dtype=torch.int64, device=self.args.device)
    outputs = []
    output_scores = []
    labels = []

    def get_classification_score(tokens: list[int], pred_scores: list[torch.FloatTensor]) -> float:
        yes_no_pos = None
        for x in yes_no_tokens:
            try:
                pos = tokens[::-1].index(x)
                if yes_no_pos is None or pos < yes_no_pos:
                    yes_no_pos = pos
            except ValueError:
                continue
        if yes_no_pos is None:
            logging.warning(f'Yes/No token not found in prediction: {tokenizer.decode(tokens, skip_special_tokens=True)}')
            return 0.5
        yes_no_idx = len(tokens) - yes_no_pos - 1
        yes_no_tok = tokens[yes_no_idx]
        pred_scores = torch.stack(pred_scores, dim=0)
        softmax_scores = pred_scores.softmax(dim=-1)
        score = softmax_scores[yes_no_idx, yes_no_tok] / softmax_scores.index_select(1, yes_no_tokens_tensor).sum()
        if tokenizer.decode(yes_no_tok).lower().strip() == 'no':
            score = 1 - score
        return score.cpu().item()

    gradient_checkpointing_old = self.args.gradient_checkpointing
    self.model.gradient_checkpointing_disable()
    model.generation_config.pad_token_id = model.generation_config.eos_token_id  # suppresses the warning

    for batch in tqdm(eval_dataloader, desc=f'RANK[{local_rank}] Eval'):
        batch['images'] = [i.to(compute_dtype) for i in batch['images']] if isinstance(batch['images'], list) else batch['images'].to(compute_dtype)
        for i in range(len(batch['input_ids'])):
            for j in range(len(batch['input_ids'][i])):
                if batch['input_ids'][i][j:j+3].tolist() == close_inst_seq:
                    batch['attention_mask'][i][j+3:] = 0
                    break

        output: GenerateDecoderOnlyOutput = self.model.generate(
            inputs=batch['input_ids'],
            images=batch['images'],
            image_sizes=[x.shape[1:] for x in batch['images']],
            attention_mask=batch['attention_mask'],
            max_new_tokens=512,
            output_scores=True,
            return_dict_in_generate=True,
            # since we are doing explanation -> classification, it is better with sampling
            do_sample=training_args.eval_with_sampling,
            min_p=0.1 if training_args.eval_with_sampling else None,
            temperature=0.9 if training_args.eval_with_sampling else None,
        )
        output_seq_list = output['sequences'].cpu().numpy().tolist()
        outputs.extend(output_seq_list)
        # make bs the first dim to be iterated. output['scores'] is a tuple of len seq_len, each is a tensor of shape (bs, vocab_size)
        scores = [[] for _ in range(training_args.per_device_eval_batch_size)]
        for batched_logits_per_token in output['scores']:
            for i in range(len(batched_logits_per_token)):
                scores[i].append(batched_logits_per_token[i])
        for (seq, score) in zip(output_seq_list, scores):
            output_scores.append(get_classification_score(seq, score))
        labels.extend(batch['input_ids'].cpu().numpy().tolist())

    if torch.distributed.is_initialized():
        # all ranks must have eval done for save best model to work
        world_size = torch.distributed.get_world_size() if self.args.local_rank != -1 else 1
        all_outputs = [None for _ in range(world_size)]
        all_outputs_scores = [None for _ in range(world_size)]
        all_labels = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(all_outputs, outputs)
        torch.distributed.all_gather_object(all_outputs_scores, output_scores)
        torch.distributed.all_gather_object(all_labels, labels)
        combined_outputs = [item for sublist in all_outputs for item in sublist]
        combined_output_scores = [item for sublist in all_outputs_scores for item in sublist]
        combined_labels = [item for sublist in all_labels for item in sublist]
    else:
        # we doing single GPU training
        combined_outputs = outputs
        combined_output_scores = output_scores
        combined_labels = labels

    metrics = {}
    if self.compute_metrics:
        metrics = self.compute_metrics(EvalPrediction(predictions=combined_outputs, label_ids=combined_labels), scores=combined_output_scores)
        metrics = {f'{metric_key_prefix}_{k}': v for k, v in metrics.items()}
        rank0_print(metrics)
        if local_rank == 0:
            wandb.init()
            wandb.log({k.replace('_', '/'): v for k, v in metrics.items()})
    if gradient_checkpointing_old:
        self.model.gradient_checkpointing_enable()
    return metrics