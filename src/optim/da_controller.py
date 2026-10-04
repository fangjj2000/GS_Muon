from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch


@dataclass
class DAResult:
    cosine: torch.Tensor
    mismatch: torch.Tensor
    beta: torch.Tensor
    signal_norm: torch.Tensor
    previous_momentum_norm: torch.Tensor
    mean_row_cosine: torch.Tensor
    mean_column_cosine: torch.Tensor
    min_row_cosine: torch.Tensor
    min_column_cosine: torch.Tensor
    reset_fraction: torch.Tensor


class DAController:
    """Stateless joint row-column temporal-retention controller."""

    def __init__(
        self,
        beta_min: float = 0.50,
        beta_max: float = 0.99,
        beta_reference: float = 0.95,
        cosine_temperature: float = 0.10,
        eps: float = 1e-8,
        reduction_chunk_elements: int = 1_048_576,
    ) -> None:
        if not 0.0 <= beta_min <= beta_max < 1.0:
            raise ValueError("momentum bounds must satisfy 0 <= beta_min <= beta_max < 1")
        if cosine_temperature <= 0:
            raise ValueError("cosine_temperature must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if reduction_chunk_elements <= 0:
            raise ValueError("reduction_chunk_elements must be positive")
        # These legacy shaping arguments remain accepted for CLI/checkpoint
        # compatibility. The row-column rule is beta_max times reliability.
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.beta_reference = float(beta_reference)
        self.cosine_temperature = float(cosine_temperature)
        self.eps = float(eps)
        self.reduction_chunk_elements = int(reduction_chunk_elements)

    @torch.no_grad()
    def _compute_row_column(
        self,
        current_signal: torch.Tensor,
        previous_momentum: torch.Tensor,
    ) -> Tuple[DAResult, torch.Tensor, torch.Tensor]:
        if current_signal.shape != previous_momentum.shape:
            raise ValueError("current signal and previous momentum must have the same shape")
        if current_signal.ndim != 2:
            raise ValueError("row-column retention requires two-dimensional tensors")
        if current_signal.numel() == 0:
            raise ValueError("row-column retention does not support empty tensors")

        rows, columns = current_signal.shape
        stats_kwargs = {"device": current_signal.device, "dtype": torch.float32}
        row_dot = torch.empty(rows, **stats_kwargs)
        row_signal_sq = torch.empty(rows, **stats_kwargs)
        row_momentum_sq = torch.empty(rows, **stats_kwargs)
        column_dot = torch.zeros(columns, **stats_kwargs)
        column_signal_sq = torch.zeros(columns, **stats_kwargs)
        column_momentum_sq = torch.zeros(columns, **stats_kwargs)

        # Casting complete model matrices to fp32 would add two full-size
        # buffers. Process a fixed element budget to bound temporary memory.
        chunk_rows = max(1, self.reduction_chunk_elements // max(1, columns))
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            signal_chunk = current_signal[start:end].float()
            momentum_chunk = previous_momentum[start:end].float()

            product = signal_chunk * momentum_chunk
            row_dot[start:end] = product.sum(dim=1)
            column_dot.add_(product.sum(dim=0))
            del product

            squared = signal_chunk.square()
            row_signal_sq[start:end] = squared.sum(dim=1)
            column_signal_sq.add_(squared.sum(dim=0))
            del squared

            squared = momentum_chunk.square()
            row_momentum_sq[start:end] = squared.sum(dim=1)
            column_momentum_sq.add_(squared.sum(dim=0))
            del squared, signal_chunk, momentum_chunk

        row_cosine = row_dot / (
            row_signal_sq.sqrt() * row_momentum_sq.sqrt() + self.eps
        )
        column_cosine = column_dot / (
            column_signal_sq.sqrt() * column_momentum_sq.sqrt() + self.eps
        )
        row_cosine.clamp_(min=-1.0, max=1.0)
        column_cosine.clamp_(min=-1.0, max=1.0)
        row_reliability = row_cosine.clamp(min=0.0)
        column_reliability = column_cosine.clamp(min=0.0)

        # These vectors factor the implicit geometric-mean retention matrix.
        row_scale = row_reliability.sqrt()
        column_scale = column_reliability.sqrt()
        mean_retention = row_scale.mean() * column_scale.mean()
        mean_cosine = 0.5 * (row_cosine.mean() + column_cosine.mean())
        active_fraction = (
            (row_reliability > 0).float().mean()
            * (column_reliability > 0).float().mean()
        )
        result = DAResult(
            cosine=mean_cosine,
            mismatch=1.0 - mean_retention,
            beta=self.beta_max * mean_retention,
            signal_norm=row_signal_sq.sum().sqrt(),
            previous_momentum_norm=row_momentum_sq.sum().sqrt(),
            mean_row_cosine=row_cosine.mean(),
            mean_column_cosine=column_cosine.mean(),
            min_row_cosine=row_cosine.min(),
            min_column_cosine=column_cosine.min(),
            reset_fraction=1.0 - active_fraction,
        )
        return result, row_scale, column_scale

    @torch.no_grad()
    def compute_one_matrix(
        self,
        current_signal: torch.Tensor,
        previous_momentum: torch.Tensor,
    ) -> DAResult:
        result, _, _ = self._compute_row_column(current_signal, previous_momentum)
        return result

    @torch.no_grad()
    def update_momentum_(
        self,
        current_signal: torch.Tensor,
        previous_momentum: torch.Tensor,
    ) -> DAResult:
        result, row_scale, column_scale = self._compute_row_column(
            current_signal, previous_momentum
        )
        row_scale = row_scale.to(dtype=previous_momentum.dtype)
        column_scale = column_scale.to(dtype=previous_momentum.dtype)
        # M_new = G + beta_max * sqrt(row_rho) * sqrt(column_rho) * (M_old - G).
        # Separate broadcasts avoid materializing an m-by-n beta tensor.
        previous_momentum.sub_(current_signal)
        previous_momentum.mul_(row_scale[:, None])
        previous_momentum.mul_(column_scale[None, :])
        previous_momentum.mul_(self.beta_max)
        previous_momentum.add_(current_signal)
        return result


def summarize_da_results(results: Iterable[DAResult]) -> Dict[str, float]:
    entries = list(results)
    if not entries:
        return {}

    def stack(attribute: str) -> torch.Tensor:
        return torch.stack(
            [getattr(entry, attribute).float().cpu() for entry in entries]
        )

    cosines = stack("cosine")
    mismatches = stack("mismatch")
    betas = stack("beta")
    signal_norms = stack("signal_norm")
    momentum_norms = stack("previous_momentum_norm")
    summary = {
        "mean_cosine": cosines.mean().item(),
        "median_cosine": cosines.median().item(),
        "min_cosine": cosines.min().item(),
        "mean_row_cosine": stack("mean_row_cosine").mean().item(),
        "mean_column_cosine": stack("mean_column_cosine").mean().item(),
        "min_row_cosine": stack("min_row_cosine").min().item(),
        "min_column_cosine": stack("min_column_cosine").min().item(),
        "mean_beta": betas.mean().item(),
        "median_beta": betas.median().item(),
        "reset_fraction": stack("reset_fraction").mean().item(),
        "mean_mismatch": mismatches.mean().item(),
        "mean_signal_norm": signal_norms.mean().item(),
        "mean_previous_momentum_norm": momentum_norms.mean().item(),
    }
    return summary
