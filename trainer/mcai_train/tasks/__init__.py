"""Task definitions: reward shaping and goal logic over decoded observations."""
from mcai_train.tasks.gather_wood import (
    GatherWoodReward,
    LOG_ITEM_NAMES,
    NO_LOG_DIST,
    WoodShapedReward,
    log_block_ids,
    log_item_ids,
    nearest_log_dist,
    total_item_count,
    wood_count,
)

__all__ = [
    "GatherWoodReward",
    "WoodShapedReward",
    "LOG_ITEM_NAMES",
    "NO_LOG_DIST",
    "log_block_ids",
    "log_item_ids",
    "nearest_log_dist",
    "total_item_count",
    "wood_count",
]
