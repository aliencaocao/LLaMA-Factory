from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer

import torch


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
        self.label_words = label_words
        self.label_tokens = [i for i in range(self.tokenizer.vocab_size) if self.tokenizer.decode(i).lower().strip() in self.label_words]  # all possible class tokens
        self.label_tokens_tensor = torch.tensor(yes_no_tokens, dtype=torch.int64)

    def find_last(self, text: str) -> str:
        for word in reversed(text.replace(':', ' ').split()):
            if word.lower() in self.label_tokens:
                return word.lower()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        total, correct = 0, 0
        labels_for_auroc = []
        for pred, label in zip(eval_pred.predictions, eval_pred.label_ids):
            pred = self.tokenizer.decode(pred, skip_special_tokens=True)
            label = self.tokenizer.decode([x for x in label if x > 0], skip_special_tokens=True)
            label = self.find_last(label)
            labels_for_auroc.append(1 if label == 'yes' else 0)
            try:
                correct += int(find_last(pred) == label)
                total += 1
            except:
                print(f'Failed to evaluate sample:\n'
                      f'Prediction: {pred}\n'
                      f'Label: {label}')
        print(f'acc {correct / total if total else 0.}')
        self.score_dict = {'acc': (correct / total if total else 0.)}

        if compute_result:
            return self._dump()
