"""Tasks, each defines reward shaping and goal logic over decoded observations."""
from .task import Task
from .wood import Wood

__all__ = ["Task", "Wood"]
