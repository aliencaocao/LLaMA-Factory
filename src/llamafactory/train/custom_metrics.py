from dataclasses import dataclass
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


@dataclass
class CustomMetric:
    """
    All custom metric has access to the tokenizer.
    """

    tokenizer: "PreTrainedTokenizer"

    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {}
        return result

    def __post_init__(self):
        self._dump()
