import copy
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Iterator, List, Optional, Sequence


@dataclass(frozen=True)
class StreamMetadata:
    stream_step: int
    source_index: int
    dataset_name: str
    task_type: str
    segment_id: int


class AnnotatedSample:
    """Metadata wrapper for source samples that do not permit new attributes."""

    def __init__(self, source: Any, metadata: StreamMetadata, task: Any = None) -> None:
        self._online_source = source
        self._online_metadata = metadata
        self._online_task = task

    def __getattr__(self, name: str) -> Any:
        return getattr(self._online_source, name)


def _annotate_sample(sample: Any, metadata: StreamMetadata, task: Any = None) -> Any:
    annotated = copy.copy(sample)
    try:
        setattr(annotated, "_online_metadata", metadata)
        if task is not None:
            setattr(annotated, "_online_task", task)
        return annotated
    except (AttributeError, TypeError):
        return AnnotatedSample(sample, metadata, task)


def get_stream_metadata(sample: Any, default_step: int = 0) -> StreamMetadata:
    metadata = getattr(sample, "_online_metadata", None)
    if metadata is not None:
        return metadata
    return StreamMetadata(
        stream_step=default_step,
        source_index=default_step,
        dataset_name="unknown",
        task_type="unknown",
        segment_id=0,
    )


class OnlineStream(Sequence):
    """A deterministic, one-pass view over task samples."""

    def __init__(
        self,
        samples: Sequence[Any],
        *,
        dataset_name: str,
        task_type: str,
        max_samples: Optional[int] = 2000,
        shuffle: bool = False,
        seed: int = 42,
        indices: Optional[Sequence[int]] = None,
        segment_id: int = 0,
        task: Any = None,
    ) -> None:
        available = len(samples)
        limit = available if max_samples is None or max_samples <= 0 else min(max_samples, available)
        if indices is None:
            selected = list(range(available))
            if shuffle:
                random.Random(seed).shuffle(selected)
            selected = selected[:limit]
        else:
            selected = [int(index) for index in indices]
            if any(index < 0 or index >= available for index in selected):
                raise IndexError("stream index is outside the source dataset")
            selected = selected[:limit]

        self.indices = selected
        self.dataset_name = dataset_name
        self.task_type = task_type
        self.segment_id = segment_id
        self.samples = [
            _annotate_sample(
                samples[source_index],
                StreamMetadata(
                    stream_step=stream_step,
                    source_index=source_index,
                    dataset_name=dataset_name,
                    task_type=task_type,
                    segment_id=segment_id,
                ),
                task=task,
            )
            for stream_step, source_index in enumerate(selected)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Any:
        return self.samples[index]

    def __iter__(self) -> Iterator[Any]:
        return iter(self.samples)

    def save_indices(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "dataset_name": self.dataset_name,
            "task_type": self.task_type,
            "segment_id": self.segment_id,
            "indices": self.indices,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


def build_online_stream(
    samples: Sequence[Any],
    *,
    dataset_name: str,
    task_type: str,
    max_samples: Optional[int] = 2000,
    shuffle: bool = False,
    seed: int = 42,
    indices: Optional[Sequence[int]] = None,
    indices_path: Optional[str] = None,
    segment_id: int = 0,
    task: Any = None,
) -> List[Any]:
    stream = OnlineStream(
        samples,
        dataset_name=dataset_name,
        task_type=task_type,
        max_samples=max_samples,
        shuffle=shuffle,
        seed=seed,
        indices=indices,
        segment_id=segment_id,
        task=task,
    )
    if indices_path:
        stream.save_indices(indices_path)
    return list(stream)
