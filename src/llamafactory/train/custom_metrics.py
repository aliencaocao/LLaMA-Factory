from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer
    from transformers.generation import GenerateDecoderOnlyOutput

import torch
import numpy as np


@dataclass
class CustomMetric:
    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {}
        return result

    def __post_init__(self):
        self._dump()


class LastTokenClassification(CustomMetric):
    """
    Computes accuracy and AUROC based on model output and logits. Classes are specific tokens.
    """

    def __init__(self, tokenizer: "PreTrainedTokenizer", label_words: List[str]):
        super().__init__()
        self.tokenizer = tokenizer
        self.label_words = label_words
        self.label_tokens = [i for i in range(self.tokenizer.vocab_size) if self.tokenizer.decode(i).lower().strip() in self.label_words]  # all possible class tokens

    def find_last(self, text: str) -> str:
        for word in reversed(text.replace(':', ' ').split()):
            if word.lower() in self.label_words:
                return word.lower()

    def calc_auroc(self, preds_score: list[float], labels: list[int]) -> float:
        from sklearn.metrics import roc_auc_score
        try:
            auroc = roc_auc_score(labels, preds_score)
            logging.info(f'AUROC: {auroc}')
            return auroc
        except Exception as e:
            logging.error(f'Failed to calculate AUROC: {e}')
            return 0.5


    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        total, correct = 0, 0
        logits_for_auroc = []
        labels_for_auroc = []
        for pred, label_batched in zip(eval_preds.predictions, eval_preds.label_ids):
            pred: "GenerateDecoderOnlyOutput"
            pred_tokens_batched: np.ndarray = pred.sequences  # noqa
            if pred.scores is not None:
                logits_batched = pred.scores
                for pred_tokens, label, logits in zip(pred_tokens_batched, label_batched, logits_batched):
                    pred_str = self.tokenizer.decode([x for x in pred_tokens if x > 0], skip_special_tokens=True)
                    label = self.tokenizer.decode([x for x in label if x > 0], skip_special_tokens=True)
                    label = self.find_last(label)
                    logits_for_auroc.append(logits)
                    labels_for_auroc.append(1 if label == 'yes' else 0)
                    try:
                        correct += int(self.find_last(pred_str) == label)
                        total += 1
                    except Exception as e:
                        logging.error(f'Failed to evaluate sample due to {e}:\n'
                              f'Prediction: {pred_str}\n'
                              f'Label: {label}')
            else:
                for pred_tokens, label in zip(pred_tokens_batched, label_batched):
                    pred_str = self.tokenizer.decode([x for x in pred_tokens if x > 0], skip_special_tokens=True)
                    label = self.tokenizer.decode([x for x in label if x > 0], skip_special_tokens=True)
                    label = self.find_last(label)
                    try:
                        correct += int(self.find_last(pred_str) == label)
                        total += 1
                    except Exception as e:
                        logging.error(f'Failed to evaluate sample due to {e}:\n'
                              f'Prediction: {pred_str}\n'
                              f'Label: {label}')

        print(f'acc {correct / total if total else 0.}')
        self.score_dict = {'acc': (correct / total if total else 0.)}
        if logits_for_auroc:
            self.score_dict['auroc'] = self.calc_auroc(logits_for_auroc, labels_for_auroc)

        if compute_result:
            return self._dump()
