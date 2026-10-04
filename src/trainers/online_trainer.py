import os
from typing import Any, Dict, Optional

from torch.utils.data import SequentialSampler

from src.online.metrics import OnlineMetricTracker
from src.online.stream import get_stream_metadata
from src.optim.muon_da import MuonDA


class OnlineTrainerMixin:
    """Shared protocol hooks for prequential, one-update-per-sample training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._online_tracker: Optional[OnlineMetricTracker] = None
        self._online_pending_loss = None

    def _get_train_sampler(self, *args, **kwargs):
        if getattr(self.args, "online_mode", False):
            return SequentialSampler(self.train_dataset)
        return super()._get_train_sampler(*args, **kwargs)

    def create_optimizer(self):
        if not getattr(self.args, "use_muon", False):
            return super().create_optimizer()
        if self.optimizer is None:
            self.optimizer = MuonDA(
                self.model.named_parameters(),
                lr=float(self.args.learning_rate),
                momentum=float(getattr(self.args, "muon_beta", 0.95)),
                weight_decay=float(getattr(self.args, "weight_decay", 0.0)),
                adam_betas=(
                    float(getattr(self.args, "adam_beta1", 0.9)),
                    float(getattr(self.args, "adam_beta2", 0.999)),
                ),
                adam_eps=float(
                    getattr(self.args, "adam_epsilon", getattr(self.args, "adam_eps", 1e-8))
                ),
                drift_aware=bool(getattr(self.args, "drift_aware", False)),
                da_beta_min=float(getattr(self.args, "da_beta_min", 0.50)),
                da_beta_max=float(getattr(self.args, "da_beta_max", 0.99)),
                da_eps=float(getattr(self.args, "da_eps", 1e-8)),
                da_reduction_chunk_elements=int(
                    getattr(self.args, "da_reduction_chunk_elements", 1_048_576)
                ),
                da_momentum=bool(getattr(self.args, "da_momentum", True)),
                da_hard_reset=bool(getattr(self.args, "da_hard_reset", False)),
                da_use_gg_ns=bool(getattr(self.args, "da_use_gg_ns", True)),
                gg_ns_eta_g=float(getattr(self.args, "gg_ns_eta_g", 0.50)),
                gg_ns_pre_steps=int(getattr(self.args, "gg_ns_pre_steps", 3)),
                ns_steps=int(getattr(self.args, "muon_ns_steps", 5)),
                log_callback=self.online_optimizer_callback,
            )
        return self.optimizer

    def _validate_online_protocol(self) -> None:
        if not getattr(self.args, "online_mode", False):
            return
        if int(self.args.per_device_train_batch_size) != 1:
            raise ValueError("online_mode requires per_device_train_batch_size=1")
        if int(self.args.gradient_accumulation_steps) != 1:
            raise ValueError("online_mode requires gradient_accumulation_steps=1")
        if int(getattr(self.args, "replay_buffer_size", 0)) != 0:
            raise ValueError("online_mode does not permit replay")

    def _ensure_online_tracker(self) -> Optional[OnlineMetricTracker]:
        if not getattr(self.args, "online_mode", False):
            return None
        self._validate_online_protocol()
        if self._online_tracker is None:
            metric_name = getattr(getattr(self, "framework", None), "task", None)
            metric_name = getattr(metric_name, "metric_name", "accuracy")
            output_path = None
            if getattr(self.args, "save_online_curve", True):
                output_path = os.path.join(self.args.output_dir, "online_curve.jsonl")
            self._online_tracker = OnlineMetricTracker(
                metric_name,
                interval=getattr(self.args, "online_metric_interval", 20),
                output_path=output_path,
                reset_output=int(getattr(self.state, "global_step", 0)) == 0,
            )
        return self._online_tracker

    def online_pre_update(self, model=None) -> None:
        tracker = self._ensure_online_tracker()
        if tracker is None or not getattr(self.args, "online_eval_before_update", True):
            return
        raw_samples = getattr(self, "raw_train_samples", None)
        if raw_samples is None:
            raise RuntimeError("online trainer requires raw_train_samples")
        index = int(self.state.global_step)
        if index >= len(raw_samples):
            raise RuntimeError("online stream would replay a previously consumed sample")
        sample = raw_samples[index]
        active_model = model if model is not None else self.model
        was_training = bool(active_model.training)
        prediction = self.framework.one_step_pred([], sample)
        if was_training:
            active_model.train()
        metadata = get_stream_metadata(sample, default_step=index)
        sample_task = getattr(sample, "_online_task", self.framework.task)
        tracker.begin_step(
            stream_step=metadata.stream_step,
            prediction=prediction.predicted_candidate,
            reference=prediction.correct_candidate,
            dataset_name=metadata.dataset_name,
            segment_id=metadata.segment_id,
            metric_name=getattr(sample_task, "metric_name", "accuracy"),
        )

    def online_finish_step(
        self,
        loss: Any = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        tracker = self._ensure_online_tracker()
        if tracker is None:
            return
        try:
            learning_rate = float(self._get_learning_rate())
        except (AttributeError, TypeError, ValueError):
            learning_rate = float(getattr(self.args, "learning_rate", 0.0))
        tracker.finish_step(loss=loss, learning_rate=learning_rate, extra=extra)

    def online_optimizer_callback(self, summary: Dict[str, float]) -> None:
        self.online_finish_step(loss=self._online_pending_loss, extra=summary)
        self._online_pending_loss = None

    def get_online_metrics(self) -> Dict[str, Any]:
        tracker = self._ensure_online_tracker()
        return tracker.final_metrics() if tracker is not None else {}
