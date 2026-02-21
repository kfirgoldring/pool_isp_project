"""
golf_game.py — Game logic and state machine for the billiards golf game.

No OpenCV, no camera, no GUI.  All coordinates are in table centimetres.

Golf mode: pocket all colored balls in as few strokes as possible using
the white cue ball.

State machine
-------------
  WAITING_FOR_SHOT   — watching for the next stroke
  SHOT_IN_PROGRESS   — balls are moving after a stroke
  RESOLVING          — confirming which balls were pocketed
  GAME_OVER          — all colored balls have been pocketed

Transitions
-----------
  WAITING → IN_PROGRESS:
    Any ball moves more than MOVEMENT_THRESHOLD_CM between consecutive ticks.
    A snapshot of positions is saved; stroke count is NOT incremented yet.

  IN_PROGRESS → RESOLVING:
    All balls have been stationary for STATIONARY_FRAMES_REQUIRED consecutive frames.

  RESOLVING → WAITING  (or GAME_OVER):
    Compare current ball set to the snapshot taken at shot start.
    • If zero net movement (false trigger, e.g. camera noise / person passing):
        → WAITING, do NOT increment stroke count.
    • If at least one ball moved:
        → Increment stroke_count.
        → Balls that disappeared AND were near a pocket at snapshot time
          are marked as pocketed.
        → Transition to GAME_OVER if remaining_balls is empty,
          otherwise WAITING.
"""

from typing import Dict, List, Optional, Tuple

# ─── Table & game constants ───────────────────────────────────────────────────

TABLE_WIDTH_CM  = 122.0
TABLE_HEIGHT_CM = 61.0

# 6 pockets at corners and mid-long-edges (derived from table dimensions).
POCKET_POSITIONS_CM: List[Tuple[float, float]] = [
    (k * TABLE_WIDTH_CM / 2, j * TABLE_HEIGHT_CM)
    for j in (0, 1) for k in (0, 1, 2)
]
# → [(0,0), (61,0), (122,0), (0,61), (61,61), (122,61)]

POCKET_RADIUS_CM: float = 8.0   # a ball within this distance of a pocket is "pocketed"

# A ball that moves more than this between ticks is considered "in motion".
# With the tracker's 1 cm dead-zone, smoothed positions are rock-stable when
# balls are stationary.  2 cm comfortably exceeds residual jitter while still
# catching even the gentlest real tap (which moves a ball several cm).
MOVEMENT_THRESHOLD_CM: float = 2.0

# Number of consecutive frames all balls must be stationary to end IN_PROGRESS.
# At ~30 fps (33 ms/tick), 60 frames ≈ 2 seconds of complete immobility.
STATIONARY_FRAMES_REQUIRED: int = 60

# ─── State constants ──────────────────────────────────────────────────────────

STATE_SETUP       = 'SETUP'
STATE_WAITING     = 'WAITING_FOR_SHOT'
STATE_IN_PROGRESS = 'SHOT_IN_PROGRESS'
STATE_RESOLVING   = 'RESOLVING'
STATE_GAME_OVER   = 'GAME_OVER'


# ─── GolfGame class ───────────────────────────────────────────────────────────

class GolfGame:
    """
    Tracks all game state for one golf-mode session.

    Usage (called by main.py every tick):
        game = GolfGame(['orange', 'yellow', 'blue', 'bordeaux'])
        ...
        game.update(balls)              # advance state machine
        game.select_target('yellow')   # from user click
        cue   = game.get_cue_ball(balls)
        tgt   = game.get_target_ball(balls)
    """

    def __init__(self, colored_ball_colors: List[str]) -> None:
        """
        Parameters
        ----------
        colored_ball_colors : color strings of the non-cue balls to pocket.
            Must match the color strings returned by Ball_Detection
            ('orange', 'yellow', 'blue', 'bordeaux', …).
        """
        self.state: str = STATE_SETUP
        self.stroke_count: int = 0
        self.remaining_balls: List[str] = list(colored_ball_colors)
        self.pocketed_balls: List[str] = []
        self.selected_target_color: Optional[str] = None
        self._snapshot_before_shot: Dict[str, Tuple[float, float]] = {}
        self._prev_positions: Dict[str, Tuple[float, float]] = {}
        self._stationary_frames: int = 0

    # ── Public methods ────────────────────────────────────────────────────────

    def reset(self, colored_ball_colors: List[str]) -> None:
        """Reset all game state for a new round."""
        self.state = STATE_WAITING
        self.stroke_count = 0
        self.remaining_balls = list(colored_ball_colors)
        self.pocketed_balls = []
        self.selected_target_color = None
        self._snapshot_before_shot = {}
        self._prev_positions = {}
        self._stationary_frames = 0

    def update(self, balls: List[Dict]) -> None:
        """
        Advance the state machine for one tick.

        Parameters
        ----------
        balls : list of ball dicts from main.py's adapter layer.
            Each dict must contain 'center_cm' (or None) and 'color'.
            The cue ball has 'is_cue': True.
        """
        if self.state in (STATE_SETUP, STATE_GAME_OVER):
            return

        current_positions = _extract_positions(balls)

        if self.state == STATE_WAITING:
            self._handle_waiting(current_positions)

        elif self.state == STATE_IN_PROGRESS:
            self._handle_in_progress(current_positions)

        elif self.state == STATE_RESOLVING:
            self._handle_resolving(current_positions)

        self._prev_positions = current_positions

    def select_target(self, color: str) -> bool:
        """
        Set the currently selected target ball by color.

        Called by main.py when the user clicks on a ball.
        No-op if the same color is already selected.
        Only accepts colors that are still in remaining_balls.

        Returns True if the selection was accepted, False otherwise.
        """
        if color == self.selected_target_color:
            return True   # already selected — no-op
        if color not in self.remaining_balls:
            return False
        self.selected_target_color = color
        return True

    def get_cue_ball(self, balls: List[Dict]) -> Optional[Dict]:
        """Return the cue ball dict from the provided list, or None."""
        for ball in balls:
            if ball.get('is_cue'):
                return ball
        return None

    def get_target_ball(self, balls: List[Dict]) -> Optional[Dict]:
        """
        Return the dict of the currently selected target ball, or None.
        Matches by self.selected_target_color against ball['color'].
        """
        if self.selected_target_color is None:
            return None
        for ball in balls:
            if ball.get('color') == self.selected_target_color and not ball.get('is_cue'):
                return ball
        return None

    def is_game_over(self) -> bool:
        """True when no colored balls remain to be pocketed."""
        return len(self.remaining_balls) == 0

    # ── Private state-machine handlers ───────────────────────────────────────

    def _handle_waiting(self, current: Dict[str, Tuple[float, float]]) -> None:
        """WAITING: detect stroke onset."""
        if not self._prev_positions:
            # First tick — just record positions.
            return

        if _any_ball_moved(self._prev_positions, current, MOVEMENT_THRESHOLD_CM):
            # Stroke detected: save snapshot, enter IN_PROGRESS.
            # Stroke count is NOT incremented here — only confirmed in RESOLVING.
            self._snapshot_before_shot = dict(self._prev_positions)
            self._stationary_frames = 0
            self.state = STATE_IN_PROGRESS

    def _handle_in_progress(self, current: Dict[str, Tuple[float, float]]) -> None:
        """IN_PROGRESS: wait for all balls to stop."""
        if _any_ball_moved(self._prev_positions, current, MOVEMENT_THRESHOLD_CM):
            self._stationary_frames = 0
        else:
            self._stationary_frames += 1
            if self._stationary_frames >= STATIONARY_FRAMES_REQUIRED:
                self.state = STATE_RESOLVING

    def _handle_resolving(self, current: Dict[str, Tuple[float, float]]) -> None:
        """RESOLVING: count strokes and identify pocketed balls."""
        snapshot = self._snapshot_before_shot

        # Check for false trigger (zero net movement — e.g. camera noise).
        if not _any_ball_moved(snapshot, current, MOVEMENT_THRESHOLD_CM):
            # Nothing actually happened; discard this "shot".
            self.state = STATE_WAITING
            return

        # Real shot confirmed: increment stroke.
        self.stroke_count += 1

        # Identify pocketed balls: any colored ball present in the pre-shot
        # snapshot but missing from current detection was pocketed.
        for color in list(snapshot.keys()):
            if color == 'white':
                continue
            if color in self.remaining_balls and color not in current:
                self.pocketed_balls.append(color)
                self.remaining_balls.remove(color)
                if self.selected_target_color == color:
                    self.selected_target_color = None

        # Transition.
        if self.is_game_over():
            self.state = STATE_GAME_OVER
        else:
            self.state = STATE_WAITING


# ─── Module-level helpers ─────────────────────────────────────────────────────

def _extract_positions(balls: List[Dict]) -> Dict[str, Tuple[float, float]]:
    """
    Build a color → (x_cm, y_cm) map from the ball list.
    Balls with None center_cm are skipped.
    """
    positions: Dict[str, Tuple[float, float]] = {}
    for ball in balls:
        cm = ball.get('center_cm')
        color = ball.get('color')
        if cm is not None and color:
            positions[color] = (float(cm[0]), float(cm[1]))
    return positions


def _any_ball_moved(
    prev:    Dict[str, Tuple[float, float]],
    current: Dict[str, Tuple[float, float]],
    threshold_cm: float,
) -> bool:
    """
    Return True if any ball that appears in BOTH dicts moved more than threshold_cm.
    Balls that appear in only one dict (appeared/disappeared) are ignored here;
    they are handled in RESOLVING.
    """
    for color, (cx, cy) in current.items():
        if color not in prev:
            continue
        px, py = prev[color]
        dist_sq = (cx - px) ** 2 + (cy - py) ** 2
        if dist_sq > threshold_cm ** 2:
            return True
    return False


def _near_any_pocket(pos_cm: Tuple[float, float]) -> bool:
    """Return True if pos_cm is within POCKET_RADIUS_CM of any pocket."""
    x, y = pos_cm
    for px, py in POCKET_POSITIONS_CM:
        if (x - px) ** 2 + (y - py) ** 2 <= POCKET_RADIUS_CM ** 2:
            return True
    return False
