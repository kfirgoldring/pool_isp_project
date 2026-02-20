"""
trajectory.py — Pure shot-geometry functions for the billiards golf game.

No state, no GUI, no OpenCV, no Qt.
All coordinates are in table centimetres (origin = top-left pocket).
Table is 122 cm × 61 cm with 6 pockets at corners and mid-long-edges.
"""
import math
from typing import Dict, List, Optional, Tuple

# ─── Table constants ──────────────────────────────────────────────────────────
TABLE_WIDTH_CM  = 122.0
TABLE_HEIGHT_CM = 61.0
BALL_RADIUS_CM  = 2.875   # standard pool ball ≈ 57.15 mm diameter / 2 = 2.875 cm
                           # (derived from BALL_RADIUS_TOP_DOWN=23px / TABLE_DISPLAY_SCALE=8)

# Pocket positions derived from table dimensions — 6 pockets at corners and mid-edges.
POCKET_POSITIONS_CM: List[Tuple[float, float]] = [
    (k * TABLE_WIDTH_CM / 2, j * TABLE_HEIGHT_CM)
    for j in (0, 1) for k in (0, 1, 2)
]
# Expands to: [(0,0), (61,0), (122,0), (0,61), (61,61), (122,61)]


# ─── Public API ───────────────────────────────────────────────────────────────

def calculate_path(
    cue_cm:    Tuple[float, float],
    target_cm: Tuple[float, float],
    pocket_cm: Optional[Tuple[float, float]] = None,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Ghost-ball geometry for a single billiards shot.

    Parameters
    ----------
    cue_cm    : (x, y) of cue ball in table centimetres.
    target_cm : (x, y) of target ball in table centimetres.
    pocket_cm : (x, y) of destination pocket in table centimetres.
                Pass None (default) to auto-select the best pocket.

    Returns
    -------
    (cue_path, target_path)
      cue_path    : [cue_position_cm, ghost_ball_contact_point_cm]
      target_path : [target_position_cm, pocket_position_cm]
    Both lists contain exactly 2 (x, y) tuples.
    Returns ([], []) if cue and target coincide or geometry fails.

    Algorithm
    ---------
    To pocket the target ball:
      1. Determine the unit vector from target towards the pocket.
      2. The ghost-ball position is target - unit_vec * 2 * BALL_RADIUS_CM.
         (This is where the cue ball centre must be at the moment of contact.)
      3. The cue travels from its current position to the ghost-ball position.
      4. The target ball travels from its position to the pocket.
    """
    cx, cy = cue_cm
    tx, ty = target_cm

    if pocket_cm is None:
        shot_dir = (tx - cx, ty - cy)
        pocket_cm = _choose_best_pocket(target_cm, shot_dir)

    px, py = pocket_cm

    # Vector from target to pocket
    dx, dy = px - tx, py - ty
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return [], []

    # Unit vector target → pocket
    ux, uy = dx / dist, dy / dist

    # Ghost-ball position: step back from target by 2 ball diameters
    gx = tx - ux * 2 * BALL_RADIUS_CM
    gy = ty - uy * 2 * BALL_RADIUS_CM

    cue_path    = [(cx, cy), (gx, gy)]
    target_path = [(tx, ty), (px, py)]
    return cue_path, target_path


def suggest_best_shot(
    cue_cm:          Tuple[float, float],
    remaining_balls: List[Dict],
) -> Tuple[Optional[Dict], List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Select the easiest ball to pocket from the cue ball's current position.

    Parameters
    ----------
    cue_cm          : (x, y) of cue ball in table centimetres.
    remaining_balls : list of ball dicts, each must have 'center_cm' and 'color'.
                      Only balls whose 'center_cm' is not None are considered.

    Returns
    -------
    (best_ball_dict, cue_path, target_path)
      best_ball_dict : the ball dict with the easiest shot, or None if no candidates.
      cue_path       : ghost-ball cue path in table cm (2 points).
      target_path    : target-to-pocket path in table cm (2 points).

    Scoring heuristic
    -----------------
    For every (ball, pocket) pair compute the angle between:
      - the shot vector (cue → ball)
      - the target-to-pocket vector (ball → pocket)
    The pair with the smallest angle requires the least deflection and is easiest.
    """
    best_ball: Optional[Dict] = None
    best_cue_path: List[Tuple[float, float]] = []
    best_target_path: List[Tuple[float, float]] = []
    best_angle = float('inf')

    cx, cy = cue_cm

    for ball in remaining_balls:
        bpos = ball.get('center_cm')
        if bpos is None:
            continue
        bx, by = bpos

        shot_dx, shot_dy = bx - cx, by - cy
        shot_dist = math.hypot(shot_dx, shot_dy)
        if shot_dist < 1e-6:
            continue

        for pocket in POCKET_POSITIONS_CM:
            px, py = pocket
            to_pocket_dx = px - bx
            to_pocket_dy = py - by
            pocket_dist = math.hypot(to_pocket_dx, to_pocket_dy)
            if pocket_dist < 1e-6:
                continue

            # Angle between shot direction and target→pocket direction
            cosang = (shot_dx * to_pocket_dx + shot_dy * to_pocket_dy) / (shot_dist * pocket_dist)
            cosang = max(-1.0, min(1.0, cosang))
            angle = math.acos(cosang)

            if angle < best_angle:
                best_angle = angle
                best_ball = ball
                best_cue_path, best_target_path = calculate_path(cue_cm, bpos, pocket)

    return best_ball, best_cue_path, best_target_path


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _choose_best_pocket(
    target_cm:  Tuple[float, float],
    shot_dir:   Tuple[float, float],
) -> Tuple[float, float]:
    """
    Pick the pocket whose direction from target_cm best aligns with shot_dir.

    Parameters
    ----------
    target_cm : (x, y) of the target ball in table cm.
    shot_dir  : (dx, dy) direction vector from cue to target (need not be unit).

    Returns
    -------
    (x, y) of the best matching pocket in table cm.

    Algorithm (extracted and converted to cm from gui/app.py _choose_pocket(),
    lines 1014-1050):
      For each pocket compute the angle between shot_dir and the vector
      (target → pocket). Return the pocket with the minimum angle.
    """
    tx, ty = target_cm
    sx, sy = shot_dir
    shot_norm = math.hypot(sx, sy)
    if shot_norm < 1e-6:
        # Degenerate: cue and target coincide — return nearest pocket
        return min(
            POCKET_POSITIONS_CM,
            key=lambda p: (p[0] - tx) ** 2 + (p[1] - ty) ** 2,
        )

    best_pocket = POCKET_POSITIONS_CM[0]
    best_angle  = float('inf')

    for pocket in POCKET_POSITIONS_CM:
        px, py = pocket
        tdx, tdy = px - tx, py - ty
        pocket_norm = math.hypot(tdx, tdy)
        if pocket_norm < 1e-6:
            continue

        cosang = (sx * tdx + sy * tdy) / (shot_norm * pocket_norm)
        cosang = max(-1.0, min(1.0, cosang))
        angle  = math.acos(cosang)

        if angle < best_angle:
            best_angle  = angle
            best_pocket = pocket

    return best_pocket
