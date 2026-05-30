"""Task definitions: reward shaping and goal logic over decoded observations."""
from mcai_train.tasks.gather_wood import (
    GatherWoodReward,
    LOG_ITEM_NAMES,
    log_item_ids,
    wood_count,
)

__all__ = [
    "GatherWoodReward",
    "LOG_ITEM_NAMES",
    "log_item_ids",
    "wood_count",
]
