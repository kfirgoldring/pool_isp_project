"""
Physics_Engine.py — Pure geometry helpers for billiards shot planning.
No state, no GUI, no OpenCV, no Qt.
All coordinates are in table centimetres (origin = top-left pocket).
"""
import math
from typing import List, Tuple, Optional

from config import BALL_RADIUS_CM, TABLE_WIDTH_CM, TABLE_HEIGHT_CM


def is_path_blocked(
    a_cm: Tuple[float, float],
    b_cm: Tuple[float, float],
    obstacles: List[Tuple[float, float]],
    ball_radius_cm: float = BALL_RADIUS_CM,
    buffer_cm: float = 0.0,
) -> bool:
    """
    Return True if any ball in *obstacles* physically blocks the straight
    line segment a_cm → b_cm.

    A ball at position P blocks the segment when its perpendicular distance
    to the segment is less than 2 * ball_radius_cm + buffer_cm AND the
    closest point on the segment to P lies between the endpoints (t ∈ [0, 1]).

    Parameters
    ----------
    a_cm, b_cm      : endpoints of the segment in table cm.
    obstacles       : list of (x, y) ball centre positions to test.
    ball_radius_cm  : radius of a ball (default = BALL_RADIUS_CM from config).
    buffer_cm       : extra clearance margin added to the block distance.
                      Use 0.5 for the cue path to account for calibration error.
    """
    dx = b_cm[0] - a_cm[0]
    dy = b_cm[1] - a_cm[1]
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return False

    block_dist = 2.0 * ball_radius_cm + buffer_cm

    for ox, oy in obstacles:
        apx = ox - a_cm[0]
        apy = oy - a_cm[1]

        # Scalar projection of AP onto AB, normalized to [0,1]
        t = (apx * dx + apy * dy) / L2
        t = max(0.0, min(1.0, t))

        # Closest point on segment to obstacle centre
        cx = a_cm[0] + t * dx
        cy = a_cm[1] + t * dy

        perp = math.hypot(ox - cx, oy - cy)
        if perp < block_dist:
            return True

    return False


def find_bank_target_paths(
    target_cm: Tuple[float, float],
    pocket_cm: Tuple[float, float],
    obstacles: List[Tuple[float, float]],
    table_width_cm:  float = TABLE_WIDTH_CM,
    table_height_cm: float = TABLE_HEIGHT_CM,
) -> List[List[Tuple[float, float]]]:
    """
    Compute all valid 1-cushion bank paths for the target ball to reach
    pocket_cm by bouncing off one of the four table cushions.

    Uses the mirror-image / reflection method:
      - Reflect pocket_cm across a wall to get a virtual pocket.
      - The straight line from target_cm to the virtual pocket intersects
        the wall at the correct bounce point.
      - Verify the bounce point is within the wall's extent and that
        neither sub-segment (target→wall, wall→pocket) is obstructed.

    Parameters
    ----------
    target_cm       : current position of the target ball in table cm.
    pocket_cm       : destination pocket position in table cm.
    obstacles       : list of (x, y) ball centres that may block the path
                      (should exclude the cue ball and the target ball itself).
    table_width_cm  : table width (default from config).
    table_height_cm : table height (default from config).

    Returns
    -------
    List of valid 3-point paths, each [[target_cm, wall_hit_cm, pocket_cm]].
    May be empty if no unobstructed bank is possible.
    """
    tx, ty = target_cm
    px, py = pocket_cm
    valid_paths: List[List[Tuple[float, float]]] = []

    # Define the 4 walls as (wall_name, wall_coord, is_vertical, reflected_pocket_fn)
    walls = [
        # Left wall  x = 0
        ('left',   0.0,              True,  (-px,                       py)),
        # Right wall x = table_width
        ('right',  table_width_cm,   True,  (2.0 * table_width_cm - px, py)),
        # Top wall   y = 0
        ('top',    0.0,              False, (px,                        -py)),
        # Bottom wall y = table_height
        ('bottom', table_height_cm,  False, (px,  2.0 * table_height_cm - py)),
    ]

    for wall_name, wall_coord, is_vertical, (rpx, rpy) in walls:
        # Direction from target to reflected pocket
        ddx = rpx - tx
        ddy = rpy - ty

        # Compute t at wall intersection: for vertical wall, solve x = wall_coord
        if is_vertical:
            if abs(ddx) < 1e-9:
                continue  # ray parallel to wall
            t = (wall_coord - tx) / ddx
        else:
            if abs(ddy) < 1e-9:
                continue  # ray parallel to wall
            t = (wall_coord - ty) / ddy

        if t < 1e-6:
            # Wall hit is behind the target (wrong direction for reflection)
            continue

        # Wall hit point
        wx = tx + t * ddx
        wy = ty + t * ddy

        # Validate: hit point must lie within the wall extent
        if is_vertical:
            if wy < -1e-6 or wy > table_height_cm + 1e-6:
                continue
        else:
            if wx < -1e-6 or wx > table_width_cm + 1e-6:
                continue

        wall_hit = (wx, wy)

        # Degenerate check: target and wall_hit are the same point
        if math.hypot(wx - tx, wy - ty) < 1e-6:
            continue

        # Obstacle clearance checks
        if is_path_blocked(target_cm, wall_hit, obstacles):
            continue
        if is_path_blocked(wall_hit, pocket_cm, obstacles):
            continue

        valid_paths.append([target_cm, wall_hit, pocket_cm])

    return valid_paths
