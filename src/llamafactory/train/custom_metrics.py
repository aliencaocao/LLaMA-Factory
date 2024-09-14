from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer

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
        self.label_tokens_tensor = torch.tensor(self.label_tokens, dtype=torch.int64)

    def find_last(self, text: str) -> str:
        for word in reversed(text.replace(':', ' ').split()):
            if word.lower() in self.label_words:
                return word.lower()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        total, correct = 0, 0
        for pred, label in zip(eval_preds.predictions, eval_preds.label_ids):
            pred = self.tokenizer.decode([x for x in pred if x > 0], skip_special_tokens=True)
            label = self.tokenizer.decode([x for x in label if x > 0], skip_special_tokens=True)
            label = self.find_last(label)
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
