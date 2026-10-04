import json
import math
import os
import re
import string
from collections import Counter
from typing import Any, Dict, List, Optional


def _normalize_answer(value: Any) -> str:
    text = str(value).lower()
    text = "".join(character for character in text if character not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _matches(prediction: Any, reference: Any) -> float:
    if isinstance(reference, list):
        return float(prediction in reference)
    return float(prediction == reference)


def _exact_match(prediction: Any, reference: Any) -> float:
    references = reference if isinstance(reference, list) else [reference]
    normalized = _normalize_answer(prediction)
    return float(any(normalized == _normalize_answer(item) for item in references))


def _f1(prediction: Any, reference: Any) -> float:
    references = reference if isinstance(reference, list) else [reference]
    prediction_tokens = _normalize_answer(prediction).split()
    best = 0.0
    for item in references:
        reference_tokens = _normalize_answer(item).split()
        common = Counter(prediction_tokens) & Counter(reference_tokens)
        matches = sum(common.values())
        if matches == 0:
            score = 0.0
        else:
            precision = matches / max(1, len(prediction_tokens))
            recall = matches / max(1, len(reference_tokens))
            score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


class OnlineMetricTracker:
    """Accumulate metrics from predictions made before each online update."""

    def __init__(
        self,
        metric_name: str,
        *,
        interval: int = 20,
        output_path: Optional[str] = None,
        reset_output: bool = True,
    ) -> None:
        self.metric_name = metric_name.lower()
        self.interval = max(1, int(interval))
        self.output_path = output_path
        self.rows: List[Dict[str, Any]] = []
        self.scores: List[float] = []
        self._pending_index: Optional[int] = None
        if output_path and reset_output:
            directory = os.path.dirname(output_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(output_path, "w", encoding="utf-8"):
                pass

    def _score(self, prediction: Any, reference: Any, metric_name: str) -> float:
        if metric_name == "accuracy":
            return _matches(prediction, reference)
        if metric_name == "em":
            return _exact_match(prediction, reference)
        if metric_name == "f1":
            return _f1(prediction, reference)
        raise ValueError(f"unsupported online metric: {metric_name}")

    def begin_step(
        self,
        *,
        stream_step: int,
        prediction: Any,
        reference: Any,
        dataset_name: str,
        segment_id: int,
        metric_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self._pending_index is not None:
            self.finish_step()
        active_metric = (metric_name or self.metric_name).lower()
        score = self._score(prediction, reference, active_metric)
        self.scores.append(score)
        count = len(self.scores)
        row = {
            "stream_step": int(stream_step),
            "dataset_name": dataset_name,
            "segment_id": int(segment_id),
            "metric_name": active_metric,
            "prediction": prediction,
            "reference": reference,
            "pre_update_metric": score,
            "cumulative_metric": sum(self.scores) / count,
            "loss_before_update": None,
            "learning_rate": None,
        }
        self.rows.append(row)
        self._pending_index = len(self.rows) - 1
        return row

    def finish_step(
        self,
        *,
        loss: Optional[Any] = None,
        learning_rate: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._pending_index is None:
            return
        row = self.rows[self._pending_index]
        if loss is not None:
            row["loss_before_update"] = _to_number(loss)
        if learning_rate is not None:
            row["learning_rate"] = float(learning_rate)
        if extra:
            row.update({key: _to_number(value) for key, value in extra.items()})
        if self.output_path:
            with open(self.output_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        self._pending_index = None

    def add_last_step_extra(self, extra: Dict[str, Any]) -> None:
        if not self.rows:
            return
        self.rows[-1].update({key: _to_number(value) for key, value in extra.items()})

    @property
    def cumulative_metric(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    def final_metrics(self) -> Dict[str, Any]:
        self.finish_step()
        return {
            "online_cumulative_metric": self.cumulative_metric,
            f"online_{self.metric_name}": self.cumulative_metric,
            "online_samples": len(self.scores),
        }


def _to_number(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (RuntimeError, ValueError):
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    return value
