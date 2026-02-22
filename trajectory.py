"""
trajectory.py â€" Pure shot-geometry functions for the billiards golf game.
No state, no GUI, no OpenCV, no Qt.
All coordinates are in table centimetres (origin = top-left pocket).
Table is 122 cm Ã— 61 cm with 6 pockets at corners and mid-long-edges.
"""
import math
from typing import Dict, List, Optional, Tuple
from config import BALL_RADIUS_CM, TABLE_HEIGHT_CM, TABLE_WIDTH_CM
from Physics_Engine import is_path_blocked, find_bank_target_paths

# Pocket positions derived from table dimensions and of-set 
import math
from typing import List, Tuple

def compute_pocket_positions(table_width_cm: float, table_height_cm: float, offset_cm: float) -> List[Tuple[float, float]]:
    """
    Computes pocket positions with axis-aligned offsets.
    Logic: If it's a side pocket (the middle index of xs), offset only the axis towards the center.
    """
    pockets = []
    
    # Define the X and Y coordinates with their respective offsets
    # Corners are offset by offset_cm, middle X is exactly at center
    xs = [offset_cm, table_width_cm / 2, table_width_cm - offset_cm]
    ys = [offset_cm, table_height_cm - offset_cm]
    
    for y in ys:
        for x in xs:
            pockets.append((x, y))
            
    return pockets

# Adjust offset as needed
POCKET_POSITIONS_CM: List[Tuple[float, float]] = compute_pocket_positions(
    TABLE_WIDTH_CM, TABLE_HEIGHT_CM, offset_cm=2.0 )

'''POCKET_POSITIONS_CM: List[Tuple[float, float]] = [
    (k * TABLE_WIDTH_CM / 2, j * TABLE_HEIGHT_CM)
    for j in (0, 1) for k in (0, 1, 2)
]'''
# Expands to: [(0,0), (61,0), (122,0), (0,61), (61,61), (122,61)]

# Squared distance threshold for excluding the cue/target ball from the obstacle list.
# Avoids fragile exact float equality after homography conversion.
_EXCLUDE_DIST_SQ: float = (BALL_RADIUS_CM * 0.5) ** 2


# â"€â"€â"€ Public API â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
def calculate_path(
    cue_cm:    Tuple[float, float],
    target_cm: Tuple[float, float],
    pocket_cm: Optional[Tuple[float, float]] = None,
    obstacles: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Ghost-ball geometry for a single billiards shot.

    Parameters
    ----------
    cue_cm    : (x, y) of cue ball in table centimetres.
    target_cm : (x, y) of target ball in table centimetres.
    pocket_cm : (x, y) of destination pocket in table centimetres.
                Pass None (default) to auto-select the best pocket.
    obstacles : list of (x, y) ball centres that may block the path.
                Used only when pocket_cm is None (for pocket auto-selection).

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
        pocket_cm = _choose_best_pocket(target_cm, shot_dir, obstacles)
    px, py = pocket_cm
    # Vector from target to pocket
    dx, dy = px - tx, py - ty
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return [], []
    # Unit vector target â†' pocket
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
    all_balls:       Optional[List[Dict]] = None,
) -> Tuple[Optional[Dict], List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Select the easiest ball to pocket from the cue ball's current position.

    Parameters
    ----------
    cue_cm          : (x, y) of cue ball in table centimetres.
    remaining_balls : list of ball dicts, each must have 'center_cm' and 'color'.
                      Only balls whose 'center_cm' is not None are considered.
    all_balls       : full list of all ball dicts on the table (used to build
                      per-shot obstacle lists for blocking checks). If None,
                      obstacle detection is skipped (backward-compatible).

    Returns
    -------
    (best_ball_dict, cue_path, target_path)
      best_ball_dict : the ball dict with the easiest shot, or None if no candidates.
      cue_path       : ghost-ball cue path in table cm (2 points).
      target_path    : target-to-pocket path in table cm (2 or 3 points).
                       3-point paths indicate a 1-cushion bank shot.

    Scoring heuristic
    -----------------
    For every (ball, pocket) pair:
      • Direct shot (unblocked cue path + unblocked target path):
          score = 0.6 * angle_score + 0.2 * dist_ct_score + 0.2 * dist_tp_score
      • Bank shot (target bounces off 1 cushion, needed when target→pocket blocked):
          score = same formula  +  0.2 penalty
      Angle score is quadratic (angle/π)² to penalise large cut angles harshly.
      A near-miss clearance penalty (up to +0.1) is added for cue paths that
      barely clear an obstacle.
      Candidates where the cue→ghost path is blocked are skipped entirely.
    The pair with the lowest score wins.
    """
    best_ball: Optional[Dict] = None
    best_cue_path: List[Tuple[float, float]] = []
    best_target_path: List[Tuple[float, float]] = []
    best_score = float('inf')

    cx, cy = cue_cm

    # Build a lookup of all ball centres for obstacle checking
    all_centers: List[Tuple[float, float]] = []
    if all_balls is not None:
        for b in all_balls:
            c = b.get('center_cm')
            if c is not None:
                all_centers.append(c)

    candidates: List[Dict] = []
    for ball in remaining_balls:
        bpos = ball.get('center_cm')
        if bpos is None:
            continue
        bx, by = bpos

        shot_dx, shot_dy = bx - cx, by - cy
        shot_dist = math.hypot(shot_dx, shot_dy)
        if shot_dist < 1e-6:
            continue

        # Obstacles for this ball: all balls except the cue and this target.
        # Use a distance threshold instead of exact equality to be robust against
        # floating-point rounding introduced by homography conversion.
        obstacles = [
            c for c in all_centers
            if (c[0]-cue_cm[0])**2 + (c[1]-cue_cm[1])**2 > _EXCLUDE_DIST_SQ
            and (c[0]-bpos[0])**2 + (c[1]-bpos[1])**2 > _EXCLUDE_DIST_SQ
        ]

        for pocket in POCKET_POSITIONS_CM:
            px, py = pocket
            to_pocket_dx = px - bx
            to_pocket_dy = py - by
            pocket_dist = math.hypot(to_pocket_dx, to_pocket_dy)
            if pocket_dist < 1e-6:
                continue

            # Ghost ball for a direct shot to this pocket
            pocket_norm = math.hypot(to_pocket_dx, to_pocket_dy)
            ux, uy = to_pocket_dx / pocket_norm, to_pocket_dy / pocket_norm
            ghost_cm = (bx - ux * 2 * BALL_RADIUS_CM, by - uy * 2 * BALL_RADIUS_CM)

            # If the cue can't reach the ghost ball, skip this (ball, pocket) pair.
            # buffer_cm=0.5 adds a small real-world safety margin for calibration error.
            if obstacles and is_path_blocked(cue_cm, ghost_cm, obstacles, buffer_cm=0.5):
                continue

            # Angle between shot direction and target->pocket direction
            cosang = (shot_dx * to_pocket_dx + shot_dy * to_pocket_dy) / (shot_dist * pocket_dist)
            cosang = max(-1.0, min(1.0, cosang))
            angle = math.acos(cosang)

            tp_blocked = bool(obstacles) and is_path_blocked(bpos, pocket, obstacles)

            if not tp_blocked:
                # Direct shot
                candidates.append({
                    'ball':        ball,
                    'bpos':        bpos,
                    'pocket':      pocket,
                    'angle':       angle,
                    'dist_ct':     shot_dist,
                    'dist_tp':     pocket_dist,
                    'bank_penalty': 0.0,
                    'cue_path':    [(cx, cy), ghost_cm],
                    'target_path': [bpos, pocket],
                    'obstacles':   obstacles,
                })
            else:
                # Try 1-cushion bank shots for the target→pocket segment
                bank_paths = find_bank_target_paths(bpos, pocket, obstacles)
                for bank_path in bank_paths:
                    wall_hit = bank_path[1]
                    wdx = wall_hit[0] - bx
                    wdy = wall_hit[1] - by
                    wdist = math.hypot(wdx, wdy)
                    if wdist < 1e-9:
                        continue
                    bank_ux, bank_uy = wdx / wdist, wdy / wdist
                    bank_ghost = (bx - bank_ux * 2 * BALL_RADIUS_CM,
                                  by - bank_uy * 2 * BALL_RADIUS_CM)

                    # Cue must also reach the bank ghost position
                    if obstacles and is_path_blocked(cue_cm, bank_ghost, obstacles, buffer_cm=0.5):
                        continue

                    # Angle for bank: between (cue→target) and (target→wall_hit)
                    bank_cosang = (shot_dx * wdx + shot_dy * wdy) / (shot_dist * wdist)
                    bank_cosang = max(-1.0, min(1.0, bank_cosang))
                    bank_angle = math.acos(bank_cosang)

                    # Total path length for target: target→wall + wall→pocket
                    dist_wall_to_pocket = math.hypot(
                        pocket[0] - wall_hit[0], pocket[1] - wall_hit[1]
                    )
                    bank_tp_dist = wdist + dist_wall_to_pocket

                    candidates.append({
                        'ball':         ball,
                        'bpos':         bpos,
                        'pocket':       pocket,
                        'angle':        bank_angle,
                        'dist_ct':      shot_dist,
                        'dist_tp':      bank_tp_dist,
                        'bank_penalty': 0.2,
                        'cue_path':     [(cx, cy), bank_ghost],
                        'target_path':  bank_path,
                        'obstacles':    obstacles,
                    })

    if not candidates:
        return None, [], []

    min_ct = min(c['dist_ct'] for c in candidates)
    max_ct = max(c['dist_ct'] for c in candidates)
    min_tp = min(c['dist_tp'] for c in candidates)
    max_tp = max(c['dist_tp'] for c in candidates)

    ct_range = max_ct - min_ct
    tp_range = max_tp - min_tp

    # Weights should sum to 1.0
    w_angle = 0.6  # weight of angle between shot and target-to-pocket
    w_ct    = 0.2  # weight of distance cue-target
    w_tp    = 0.2  # weight of distance target-pocket

    for c in candidates:
        # Quadratic angle penalty: small cuts stay easy, large cuts penalised harshly.
        # A 30° cut scores 0.028 (was 0.167 linear). A 75° cut scores 0.174 (was 0.42).
        angle_norm  = c['angle'] / math.pi  # 0..1
        angle_score = angle_norm ** 2

        ct_score = 0.0 if ct_range < 1e-6 else (c['dist_ct'] - min_ct) / ct_range
        tp_score = 0.0 if tp_range < 1e-6 else (c['dist_tp'] - min_tp) / tp_range

        score = (w_angle * angle_score
                 + w_ct * ct_score
                 + w_tp * tp_score
                 + c['bank_penalty'])

        # Near-miss penalty: add up to +0.1 when the cue path barely clears an obstacle.
        cue_clr = _min_clearance(cue_cm, c['cue_path'][1], c['obstacles'])
        if 0 < cue_clr < BALL_RADIUS_CM:
            score += 0.1 * (1.0 - cue_clr / BALL_RADIUS_CM)

        if score < best_score:
            best_score     = score
            best_ball      = c['ball']
            best_cue_path  = c['cue_path']
            best_target_path = c['target_path']

    return best_ball, best_cue_path, best_target_path


def _min_clearance(
    a_cm: Tuple[float, float],
    b_cm: Tuple[float, float],
    obstacles: List[Tuple[float, float]],
) -> float:
    """
    Return the smallest signed clearance between any obstacle and segment a→b.

    Clearance = (perpendicular distance from obstacle centre to segment) - 2*BALL_RADIUS_CM.
    Positive means the path is clear. Negative means it is blocked.
    Returns +inf when there are no obstacles.
    """
    if not obstacles:
        return float('inf')
    dx = b_cm[0] - a_cm[0]
    dy = b_cm[1] - a_cm[1]
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return float('inf')
    min_clr = float('inf')
    for ox, oy in obstacles:
        apx = ox - a_cm[0]
        apy = oy - a_cm[1]
        t    = max(0.0, min(1.0, (apx * dx + apy * dy) / L2))
        perp = math.hypot(ox - (a_cm[0] + t * dx), oy - (a_cm[1] + t * dy))
        clr  = perp - 2.0 * BALL_RADIUS_CM
        if clr < min_clr:
            min_clr = clr
    return min_clr


def _choose_best_pocket(
    target_cm:  Tuple[float, float],
    shot_dir:   Tuple[float, float],
    obstacles:  Optional[List[Tuple[float, float]]] = None,
) -> Tuple[float, float]:
    """
    Pick the pocket whose direction from target_cm best aligns with shot_dir,
    preferring pockets whose target→pocket path is not obstructed.

    Parameters
    ----------
    target_cm : (x, y) of the target ball in table cm.
    shot_dir  : (dx, dy) direction vector from cue to target (need not be unit).
    obstacles : optional list of (x, y) ball centres that may block the path.
                When provided, unblocked pockets are preferred; all pockets are
                still considered as a fallback if all paths are blocked.

    Returns
    -------
    (x, y) of the best matching pocket in table cm.

    Algorithm:
      For each pocket compute the angle between shot_dir and (target -> pocket).
      Among unblocked pockets, return the one with the minimum angle.
      If all pockets are blocked (or no obstacle list given), fall back to the
      globally best-aligned pocket regardless of blocking.
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

    best_pocket       = POCKET_POSITIONS_CM[0]
    best_angle        = float('inf')
    best_clear_pocket = None
    best_clear_angle  = float('inf')

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

        if obstacles is not None:
            if not is_path_blocked(target_cm, pocket, obstacles):
                if angle < best_clear_angle:
                    best_clear_angle  = angle
                    best_clear_pocket = pocket

    # Prefer a clear pocket when one exists
    if best_clear_pocket is not None:
        return best_clear_pocket
    return best_pocket

