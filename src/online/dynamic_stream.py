import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .stream import OnlineStream, StreamMetadata


@dataclass
class DynamicStream:
    samples: List[Any]
    switch_indices: List[int]
    segments: List[Dict[str, Any]]

    def save_manifest(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {"switch_indices": self.switch_indices, "segments": self.segments},
                handle,
                indent=2,
            )


def build_dynamic_stream(
    dataset_names: Sequence[str],
    samples_by_dataset: Mapping[str, Sequence[Any]],
    *,
    tasks_by_dataset: Optional[Mapping[str, Any]] = None,
    task_types: Optional[Mapping[str, str]] = None,
    max_samples_per_dataset: int = 2000,
    shuffle_within_segment: bool = False,
    seed: int = 42,
    manifest_path: Optional[str] = None,
) -> DynamicStream:
    """Concatenate deterministic dataset segments and record every boundary."""
    samples: List[Any] = []
    switch_indices: List[int] = []
    segments: List[Dict[str, Any]] = []

    for segment_id, dataset_name in enumerate(dataset_names):
        if dataset_name not in samples_by_dataset:
            raise KeyError(f"missing samples for dynamic dataset {dataset_name!r}")
        if segment_id > 0:
            switch_indices.append(len(samples))
        task_type = (task_types or {}).get(dataset_name, "unknown")
        task = (tasks_by_dataset or {}).get(dataset_name)
        segment = OnlineStream(
            samples_by_dataset[dataset_name],
            dataset_name=dataset_name,
            task_type=task_type,
            max_samples=max_samples_per_dataset,
            shuffle=shuffle_within_segment,
            seed=seed + segment_id,
            segment_id=segment_id,
            task=task,
        )
        start = len(samples)
        for local_step, sample in enumerate(segment):
            metadata = getattr(sample, "_online_metadata")
            setattr(
                sample,
                "_online_metadata",
                StreamMetadata(
                    stream_step=start + local_step,
                    source_index=metadata.source_index,
                    dataset_name=metadata.dataset_name,
                    task_type=metadata.task_type,
                    segment_id=metadata.segment_id,
                ),
            )
            samples.append(sample)
        segments.append(
            {
                "dataset_name": dataset_name,
                "task_type": task_type,
                "segment_id": segment_id,
                "start_index": start,
                "end_index": len(samples),
                "num_samples": len(segment),
                "source_indices": segment.indices,
            }
        )

    result = DynamicStream(samples=samples, switch_indices=switch_indices, segments=segments)
    if manifest_path:
        result.save_manifest(manifest_path)
    return result
