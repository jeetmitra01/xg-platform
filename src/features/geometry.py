import math
from dataclasses import dataclass

@dataclass(frozen=True)
class PitchConfig:
    goal_x: float = 120.0
    goal_y: float = 40.0
    left_post_y: float = 36.0
    right_post_y: float = 44.0

PITCH = PitchConfig()

def shot_distance(x: float, y: float, cfg: PitchConfig = PITCH) -> float:
    return math.sqrt((cfg.goal_x - x) ** 2 + (cfg.goal_y - y) ** 2)

def shot_angle(x: float, y: float, cfg: PitchConfig = PITCH) -> float:
    """
    Angle between the lines from (x,y) to each goalpost.
    Result in radians in [0, pi].
    """
    dx = cfg.goal_x - x
    dy_left = cfg.left_post_y - y
    dy_right = cfg.right_post_y - y
    a1 = math.atan2(dy_right, dx)
    a2 = math.atan2(dy_left, dx)
    return abs(a1 - a2)
