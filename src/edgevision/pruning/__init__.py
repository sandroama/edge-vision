"""Pruning of student-backbone layers: L1 masks and channel removal.

Phase 4 modules:
    structured_prune — L1 / random element-wise *mask* pruning via
    torch.nn.utils.prune, plus true channel-removal surgery
    (``channel_prune_conv_chain``) for straight-line conv chains.
"""

from edgevision.pruning.structured_prune import (
    ChannelPruneResult,
    PruneConfig,
    PruneResult,
    apply_pruning,
    channel_prune_conv_chain,
    pruning_summary,
    remove_pruning,
)

__all__ = [
    "ChannelPruneResult",
    "PruneConfig",
    "PruneResult",
    "apply_pruning",
    "channel_prune_conv_chain",
    "pruning_summary",
    "remove_pruning",
]
