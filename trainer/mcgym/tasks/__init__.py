"""tasks, each defines reward shaping and goal logic over decoded observations"""
from .parkour import Parkour
from .task import Task
from .wood import Wood

__all__ = ["Parkour", "Task", "Wood"]
