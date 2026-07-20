"""Pruning for the student detector backbone: L1 masks AND channel removal.

Two levers, measured side by side in ``docs/results/phase4_cpu_pruning.md``:

* ``apply_pruning`` / ``remove_pruning`` — element-wise weight masks via
  ``torch.nn.utils.prune.l1_unstructured``. Measured consequence: masks leave
  raw ONNX size and dense CPU latency unchanged; only gzip-compressed size
  shrinks.
* ``channel_prune_conv_chain`` — true structured pruning: the lowest-L1
  output channels are REMOVED, so parameters, ONNX size, and FLOPs actually
  drop. Supports straight-line Conv2d chains with pooled Linear heads (the CI
  stand-in's shape); branching architectures need a dependency-graph pruner
  (torch-pruning) and fail closed here.

Two-phase workflow (mirrors the ML literature):
    1. ``apply_pruning`` — introduce pruning masks (model keeps its shape,
       masks zero out selected weights during forward).
    2. ``remove_pruning`` — bake masks into weights permanently and drop the
       masks. The model stays the same size; the zeros become permanent.

Lazy-imports torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PruneConfig:
    """Pruning knobs."""

    amount: float = 0.20         # fraction of weights to mask (0 = none, 1 = all)
    method: str = "l1"           # "l1" (L1 magnitude) or "random"
    target_module_types: tuple[str, ...] = ("Conv2d",)  # class names to target

    def __post_init__(self) -> None:
        if not (0.0 <= self.amount < 1.0):
            raise ValueError(f"amount must be in [0, 1), got {self.amount}")
        if self.method not in ("l1", "random"):
            raise ValueError(f"method must be 'l1' or 'random', got {self.method!r}")


@dataclass(frozen=True)
class PruneResult:
    """Audit log for one pruning pass."""

    n_parameters_before: int
    n_parameters_after: int
    n_zero_parameters: int
    sparsity: float           # fraction of zero weights (including pruned channels)
    modules_pruned: int
    amount_requested: float

    @property
    def compression_ratio(self) -> float:
        return self.n_parameters_before / max(self.n_parameters_after, 1)

    def as_dict(self) -> dict:
        return {
            "n_parameters_before": self.n_parameters_before,
            "n_parameters_after": self.n_parameters_after,
            "n_zero_parameters": self.n_zero_parameters,
            "sparsity": round(self.sparsity, 4),
            "modules_pruned": self.modules_pruned,
            "amount_requested": self.amount_requested,
            "compression_ratio": round(self.compression_ratio, 3),
        }


def _count_params(model: Any) -> int:
    return sum(p.numel() for p in model.parameters())


def _count_zeros(model: Any) -> int:
    zeros = 0
    for p in model.parameters():
        zeros += int((p.data == 0).sum().item())
    return zeros


def apply_pruning(
    model: Any,
    config: PruneConfig | None = None,
) -> PruneResult:
    """Apply L1-magnitude element-wise mask pruning to ``model`` in-place.

    After this call the model still has the same number of parameters but
    selected weights are zeroed out by masks. This does NOT reduce FLOPs or
    dense latency (measured null result in
    ``docs/results/phase4_cpu_pruning.md``); ``remove_pruning(model)`` only
    bakes the zeros in.

    Args:
        model: a ``torch.nn.Module`` (the student).
        config: pruning knobs. Defaults to 20% L1 on Conv2d layers.

    Returns:
        ``PruneResult`` bookkeeping.
    """
    try:
        import torch.nn.utils.prune as prune
    except ImportError as e:  # pragma: no cover
        raise ImportError("apply_pruning requires PyTorch.") from e

    config = config or PruneConfig()
    before = _count_params(model)
    modules_pruned = 0

    for _name, module in model.named_modules():
        if type(module).__name__ not in config.target_module_types:
            continue
        if not hasattr(module, "weight") or module.weight is None:
            continue

        # Prune the "weight" parameter.
        if config.method == "l1":
            prune.l1_unstructured(module, name="weight", amount=config.amount)
        else:
            prune.random_unstructured(module, name="weight", amount=config.amount)
        modules_pruned += 1

    after = _count_params(model)
    zeros = _count_zeros(model)
    total = max(after, 1)
    sparsity = zeros / total

    return PruneResult(
        n_parameters_before=before,
        n_parameters_after=after,
        n_zero_parameters=zeros,
        sparsity=sparsity,
        modules_pruned=modules_pruned,
        amount_requested=config.amount,
    )


def remove_pruning(model: Any) -> None:
    """Make pruning permanent by baking masks into weights.

    After this call:
        - Masks are removed from all modules.
        - Pruned weights are permanently zero.
        - The model is a standard dense module ready for ONNX export.

    Note: this does NOT actually shrink the model; it just removes the mask
    overhead. True channel-removal (reducing output_channels) requires
    architecture surgery not yet implemented here — that's the Phase-7 stretch
    item if INT8 alone doesn't hit the latency target.
    """
    try:
        import torch.nn.utils.prune as prune
    except ImportError as e:  # pragma: no cover
        raise ImportError("remove_pruning requires PyTorch.") from e

    for _, module in model.named_modules():
        if prune.is_pruned(module):
            prune.remove(module, "weight")


@dataclass(frozen=True)
class ChannelPruneResult:
    """Audit log for one structured channel-removal pass."""

    n_parameters_before: int
    n_parameters_after: int
    amount_requested: float
    channels_kept: dict[str, int]  # module name -> out_channels kept

    @property
    def param_reduction_pct(self) -> float:
        return 100.0 * (1 - self.n_parameters_after / max(self.n_parameters_before, 1))

    def as_dict(self) -> dict:
        return {
            "n_parameters_before": self.n_parameters_before,
            "n_parameters_after": self.n_parameters_after,
            "param_reduction_pct": round(self.param_reduction_pct, 2),
            "amount_requested": self.amount_requested,
            "channels_kept": dict(self.channels_kept),
        }


def channel_prune_conv_chain(model: Any, amount: float, min_channels: int = 1) -> tuple[Any, ChannelPruneResult]:
    """True structured pruning: REMOVE the lowest-L1 output channels.

    Unlike ``apply_pruning`` (element-wise masks; measured null effect on dense
    size/latency in ``docs/results/phase4_cpu_pruning.md``), this rebuilds each
    ``Conv2d`` with fewer output channels, slices the next conv's input
    channels to match, and slices the columns of any ``Linear`` head fed by
    the (1x1-pooled, flattened) final conv features. The returned model is
    genuinely smaller: fewer parameters, smaller ONNX export, fewer FLOPs.

    ponytail: supports straight-line Conv2d chains ending in a 1x1 pool +
    Flatten + Linear heads (the shape of ``make_tiny_model``). Branching
    architectures (residuals, transformers, RT-DETR proper) need a
    dependency-graph pruner such as torch-pruning — fails closed on mismatch.

    Args:
        model: source ``torch.nn.Module`` (left untouched; a pruned deepcopy
            is returned).
        amount: fraction of output channels to remove from every Conv2d,
            in [0, 1).
        min_channels: floor on kept channels per conv.

    Returns:
        ``(pruned_model, ChannelPruneResult)``.

    Raises:
        ValueError: if ``amount`` is out of range, the model has no Conv2d, or
            a Linear head's in_features does not match the final conv width
            (i.e. the architecture is not a plain chain — fail closed rather
            than silently corrupt weights).
    """
    if not (0.0 <= amount < 1.0):
        raise ValueError(f"amount must be in [0, 1), got {amount}")
    try:
        import copy

        import torch
        from torch import nn
    except ImportError as e:  # pragma: no cover
        raise ImportError("channel_prune_conv_chain requires PyTorch.") from e

    model = copy.deepcopy(model)
    before = _count_params(model)

    convs = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError("model has no Conv2d modules to channel-prune")

    def _set_module(root: Any, dotted: str, new: Any) -> None:
        *parents, leaf = dotted.split(".")
        obj = root
        for p in parents:
            obj = getattr(obj, p)
        setattr(obj, leaf, new)

    channels_kept: dict[str, int] = {}
    keep_prev: torch.Tensor | None = None  # input-channel indices kept so far
    with torch.no_grad():
        for name, conv in convs:
            w = conv.weight.data
            if keep_prev is not None:
                if int(w.shape[1]) != int(keep_prev_orig):
                    raise ValueError(
                        f"conv {name!r} in_channels={w.shape[1]} does not follow the "
                        f"previous conv ({keep_prev_orig} channels) — not a plain chain"
                    )
                w = w[:, keep_prev]
            n_out = w.shape[0]
            n_keep = max(min_channels, round(n_out * (1 - amount)))
            scores = w.abs().sum(dim=(1, 2, 3))  # L1 norm per output filter
            keep = torch.sort(torch.topk(scores, n_keep).indices).values

            new_conv = nn.Conv2d(
                in_channels=w.shape[1],
                out_channels=n_keep,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=conv.groups,
                bias=conv.bias is not None,
                padding_mode=conv.padding_mode,
            )
            new_conv.weight.data = w[keep].clone()
            if conv.bias is not None:
                new_conv.bias.data = conv.bias.data[keep].clone()
            _set_module(model, name, new_conv)
            channels_kept[name] = n_keep
            keep_prev, keep_prev_orig = keep, n_out

        # Linear heads consume the final conv's channels via 1x1 pool + flatten:
        # slice their input columns to the kept channels. Fail closed otherwise.
        final_orig = int(keep_prev_orig)
        for name, lin in [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]:
            if lin.in_features != final_orig:
                raise ValueError(
                    f"linear {name!r} in_features={lin.in_features} != final conv "
                    f"width {final_orig} — not a pooled-flatten head; refusing to prune"
                )
            new_lin = nn.Linear(len(keep_prev), lin.out_features, bias=lin.bias is not None)
            new_lin.weight.data = lin.weight.data[:, keep_prev].clone()
            if lin.bias is not None:
                new_lin.bias.data = lin.bias.data.clone()
            _set_module(model, name, new_lin)

    return model, ChannelPruneResult(
        n_parameters_before=before,
        n_parameters_after=_count_params(model),
        amount_requested=amount,
        channels_kept=channels_kept,
    )


def pruning_summary(result: PruneResult) -> str:
    return (
        f"  Modules pruned   : {result.modules_pruned}\n"
        f"  Amount requested : {result.amount_requested:.0%}\n"
        f"  Sparsity         : {result.sparsity:.2%}\n"
        f"  Zero params      : {result.n_zero_parameters:,} / {result.n_parameters_after:,}\n"
        f"  Compression      : {result.compression_ratio:.2f}× (mask only; remove for export)"
    )
