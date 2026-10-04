import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch

from .da_controller import DAController, DAResult, summarize_da_results


@torch.no_grad()
def zeropower_via_newton_schulz(matrix: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Compute the Muon polar direction with stable float32 NS iterations."""
    if matrix.ndim != 2:
        raise ValueError("Muon transform requires a two-dimensional tensor")
    original_dtype = matrix.dtype
    value = matrix.float()
    transposed = value.shape[0] > value.shape[1]
    if transposed:
        value = value.mT
    value = value / (torch.linalg.vector_norm(value) + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = value @ value.mT
        value = a * value + (b * gram + c * gram @ gram) @ value
    if transposed:
        value = value.mT
    return value.to(original_dtype)



@torch.no_grad()
def gradient_guided_newton_schulz(
    matrix: torch.Tensor,
    gradient: torch.Tensor,
    steps: int = 5,
    pre_steps: int = 3,
    eta_g: float = 0.50,
    eps: float = 1e-8,
    return_stats: bool = False,
):
    """Run 3/4 Muon NS steps, one gradient steering step, and a final Muon NS step."""
    if matrix.ndim != 2 or gradient.ndim != 2:
        raise ValueError("GG-NS requires two-dimensional tensors")
    if matrix.shape != gradient.shape:
        raise ValueError("matrix and gradient must have the same shape")
    if steps != 5:
        raise ValueError("GG-NS requires the Muon baseline configuration steps=5")
    if pre_steps not in (3, 4):
        raise ValueError("GG-NS pre_steps must be 3 or 4")
    if not 0.0 <= eta_g <= 1.0:
        raise ValueError("GG-NS eta_g must be in [0, 1]")
    if eps <= 0:
        raise ValueError("GG-NS epsilon must be positive")

    original_dtype = matrix.dtype
    value = matrix.float()
    current_gradient = gradient.float().to(device=value.device)
    # Match the standard NS orientation: work with rows <= columns and
    # restore the caller's shape before returning.
    transposed = value.shape[0] > value.shape[1]
    if transposed:
        value = value.mT
        current_gradient = current_gradient.mT
    rows = value.shape[0]
    identity = torch.eye(rows, device=value.device, dtype=value.dtype)
    value = value / (torch.linalg.vector_norm(value) + 1e-7)
    normalized_gradient = current_gradient / (torch.linalg.vector_norm(current_gradient) + eps)

    def muon_ns_step(current):
        a, b, c = 3.4445, -4.7750, 2.0315
        gram = current @ current.mT
        next_value = a * current + (b * gram + c * gram @ gram) @ current
        correction = next_value - current
        return next_value, correction

    last_correction = None
    for _ in range(pre_steps):
        value, last_correction = muon_ns_step(value)
    pre_steering_value = value
    last_pre_correction_norm = torch.linalg.vector_norm(last_correction)

    # For the internal wide orientation, project onto the tangent space of
    # the row-orthogonal manifold: T = G - sym(G X^T) X.
    cross = normalized_gradient @ pre_steering_value.mT
    symmetric = 0.5 * (cross + cross.mT)
    tangent = normalized_gradient - symmetric @ pre_steering_value
    tangent_norm = torch.linalg.vector_norm(tangent)
    gamma = eta_g * last_pre_correction_norm / (tangent_norm + eps)
    steered = pre_steering_value + gamma * tangent
    result, final_ns_correction = muon_ns_step(steered)

    gradient_norm = torch.linalg.vector_norm(normalized_gradient)
    tangent_alignment = torch.sum(normalized_gradient * tangent) / (
        gradient_norm * tangent_norm + eps
    )
    pre_residual = torch.linalg.vector_norm(identity - steered @ steered.mT)
    # Compute this in the internal rows <= columns orientation so the Gram
    # matrix stays bounded by the smaller matrix dimension.
    post_residual = torch.linalg.vector_norm(identity - result @ result.mT)
    stats = {
        "gg_ns_pre_steps": float(pre_steps),
        "gg_ns_eta": float(eta_g),
        "gg_ns_gamma": float(gamma.item()),
        "gg_ns_steering_norm": float((gamma * tangent_norm).item()),
        "gg_ns_tangent_alignment": float(tangent_alignment.item()),
        "gg_ns_pre_correction_norm": float(last_pre_correction_norm.item()),
        "gg_ns_pre_retraction_residual": float(pre_residual.item()),
        "gg_ns_post_retraction_residual": float(post_residual.item()),
        "gg_ns_final_correction_norm": float(
            torch.linalg.vector_norm(final_ns_correction).item()
        ),
    }
    if transposed:
        result = result.mT
    result = result.to(original_dtype)
    if not return_stats:
        return result
    return result, stats


class MuonDA(torch.optim.Optimizer):
    """Muon for matrices plus AdamW fallback, driven by gradients or explicit signals."""

    def __init__(
        self,
        named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
        *,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        adam_betas: Tuple[float, float] = (0.9, 0.999),
        adam_eps: float = 1e-8,
        drift_aware: bool = False,
        da_beta_min: float = 0.50,
        da_beta_max: float = 0.99,
        da_eps: float = 1e-8,
        da_reduction_chunk_elements: int = 1_048_576,
        da_momentum: bool = True,
        da_hard_reset: bool = False,
        da_use_gg_ns: bool = True,
        gg_ns_eta_g: float = 0.50,
        gg_ns_pre_steps: int = 3,
        ns_steps: int = 5,
        log_callback: Optional[Callable[[Dict[str, float]], None]] = None,
    ) -> None:
        named = [(name, parameter) for name, parameter in named_parameters if parameter.requires_grad]
        if not named:
            raise ValueError("MuonDA received no trainable parameters")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if drift_aware and int(ns_steps) != 5:
            raise ValueError("GG-NS requires the Muon baseline configuration ns_steps=5")
        if drift_aware and not 0.0 <= gg_ns_eta_g <= 1.0:
            raise ValueError("GG-NS eta_g must be in [0, 1]")
        if drift_aware and int(gg_ns_pre_steps) not in (3, 4):
            raise ValueError("GG-NS pre_steps must be 3 or 4")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            adam_betas=adam_betas,
            adam_eps=adam_eps,
        )
        super().__init__([parameter for _, parameter in named], defaults)
        self._names = {id(parameter): name for name, parameter in named}
        self.drift_aware = bool(drift_aware)
        self.da_momentum = bool(da_momentum)
        self.da_hard_reset = bool(da_hard_reset)
        self.da_use_gg_ns = bool(da_use_gg_ns)
        self.ns_steps = int(ns_steps)
        self.gg_ns_eta_g = float(gg_ns_eta_g)
        self.gg_ns_pre_steps = int(gg_ns_pre_steps)
        self.controller = DAController(
            beta_min=da_beta_min,
            beta_max=da_beta_max,
            eps=da_eps,
            reduction_chunk_elements=da_reduction_chunk_elements,
        )
        self.log_callback = log_callback
        self.last_da_summary: Dict[str, float] = {}

    def train(self) -> None:
        return None

    def eval(self) -> None:
        return None

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        signals = []
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    signals.append((self._names[id(parameter)], parameter, parameter.grad))
        self.step_from_signals(signals)
        return loss

    @torch.no_grad()
    def step_from_signals(
        self,
        named_signals: Iterable[Tuple[str, torch.nn.Parameter, torch.Tensor]],
    ) -> Dict[str, float]:
        group = self.param_groups[0]
        da_results: List[DAResult] = []
        gg_stats: List[Dict[str, float]] = []
        momentum_norms: List[float] = []
        for name, parameter, signal in named_signals:
            if signal is None:
                continue
            if parameter.ndim == 2:
                result, momentum_norm, gg_result = self._step_muon_matrix(parameter, signal, group)
                if gg_result is not None:
                    gg_stats.append(gg_result)
                momentum_norms.append(momentum_norm)
                if result is not None:
                    da_results.append(result)
            else:
                self._step_adamw_fallback(name, parameter, signal, group)

        summary = summarize_da_results(da_results)
        if gg_stats:
            for key in gg_stats[0]:
                values = [item[key] for item in gg_stats if key in item]
                if values:
                    summary[f"mean_{key}"] = sum(values) / len(values)
        if momentum_norms:
            summary["momentum_norm"] = sum(momentum_norms) / len(momentum_norms)
        self.last_da_summary = summary
        if self.log_callback is not None:
            self.log_callback(summary)
        return summary

    def _step_muon_matrix(self, parameter, signal, group):
        state = self.state[parameter]
        if "momentum" not in state:
            state["momentum"] = torch.zeros_like(parameter)
        momentum = state["momentum"]
        current = signal.to(device=parameter.device, dtype=parameter.dtype)
        da_result = None
        beta = float(group["momentum"])
        gg_result = None
        momentum_updated = False

        if self.drift_aware:
            if self.da_momentum:
                da_result = self.controller.update_momentum_(current, momentum)
                momentum_updated = True
            else:
                da_result = self.controller.compute_one_matrix(current, momentum)
            cosine = float(da_result.cosine.item())
            if not self.da_momentum and self.da_hard_reset and cosine <= 0.0:
                beta = 0.0

        if not momentum_updated:
            momentum.mul_(beta).add_(current, alpha=1.0 - beta)
        if self.drift_aware and self.da_use_gg_ns:
            direction, gg_result = gradient_guided_newton_schulz(
                momentum,
                current,
                steps=self.ns_steps,
                pre_steps=self.gg_ns_pre_steps,
                eta_g=self.gg_ns_eta_g,
                eps=self.controller.eps,
                return_stats=True,
            )
        else:
            direction = zeropower_via_newton_schulz(momentum, steps=self.ns_steps)
        rows, columns = parameter.shape
        direction = direction * math.sqrt(max(1.0, rows / columns))
        if group["weight_decay"]:
            parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
        parameter.add_(direction, alpha=-group["lr"])
        momentum_norm = torch.linalg.vector_norm(momentum, dtype=torch.float32)
        return da_result, float(momentum_norm.item()), gg_result

    def _step_adamw_fallback(self, name, parameter, signal, group):
        state = self.state[parameter]
        if "step" not in state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(parameter)
            state["exp_avg_sq"] = torch.zeros_like(parameter)
        state["step"] += 1
        beta1, beta2 = group["adam_betas"]
        gradient = signal.to(device=parameter.device, dtype=parameter.dtype)
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
        bias_correction1 = 1.0 - beta1 ** state["step"]
        bias_correction2 = 1.0 - beta2 ** state["step"]
        denominator = exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
        denominator.add_(group["adam_eps"])
        if group["weight_decay"] and not _exclude_weight_decay(name):
            parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
        parameter.addcdiv_(exp_avg, denominator, value=-group["lr"] / bias_correction1)


def _exclude_weight_decay(name: str) -> bool:
    lowered = name.lower()
    return "bias" in lowered or "layer_norm" in lowered or "layernorm" in lowered
