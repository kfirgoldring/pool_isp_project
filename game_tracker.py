"""
game_tracker.py — Unified game + ball-tracking state machine.

Merges the old 3-state BallTracker and 5-state GolfGame into one clean
module with two active states (TRACKING / DISTURBED) plus an initial
WAITING_FOR_BALLS phase and a terminal GAME_OVER.

State diagram
─────────────
    WAITING_FOR_BALLS ──start_game()──► TRACKING
         ▲                                │  ▲
         │ reset()             motion ────┘  │
         │                       │           │
         │                       ▼           │
         │                   DISTURBED ──────┘
         │                    30 frames calm
         │
    GAME_OVER ◄── all colored balls pocketed

Ball dict format (unchanged throughout the system):
    {
        'center':    (int, int),            # camera pixel coords
        'center_cm': (float, float) | None, # table-cm coords
        'color':     str,
        'is_cue':    bool,
        'radius':    int,
    }
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ─── State constants ─────────────────────────────────────────────────────────

ST_WAITING_FOR_BALLS = 'WAITING_FOR_BALLS'
ST_TRACKING          = 'TRACKING'
ST_DISTURBED         = 'DISTURBED'
ST_GAME_OVER         = 'GAME_OVER'

# ─── Table & game constants ─────────────────────────────────────────────────

TABLE_WIDTH_CM  = 122.0
TABLE_HEIGHT_CM = 61.0

POCKET_POSITIONS_CM: List[Tuple[float, float]] = [
    (k * TABLE_WIDTH_CM / 2, j * TABLE_HEIGHT_CM)
    for j in (0, 1) for k in (0, 1, 2)
]

POCKET_RADIUS_CM: float = 8.0

# ─── Threshold constants ────────────────────────────────────────────────────

# Minimum displacement (cm) between confirmed and averaged position to
# register as a real stroke rather than detection jitter.
SHOT_THRESHOLD_CM: float = 2.0

# Frame-diff motion detection: fraction of pixels whose intensity change
# exceeds MOTION_DIFF_THRESH before the frame counts as "has motion".
MOTION_PIXEL_FRAC:  float = 0.05
MOTION_DIFF_THRESH: int   = 30

# Consecutive calm frames required before leaving DISTURBED (~1 s at 30 fps).
CALM_FRAMES_REQUIRED: int = 30

# After returning from DISTURBED, build tracks for this many ticks before
# checking for strokes.  Keeps a single noisy frame from causing a false shot.
REACQUISITION_TICKS: int = 5

# Stroke penalty applied when the cue ball is pocketed.
CUE_BALL_PENALTY: int = 3

# Consecutive STABLE-TRACKING frames white must be absent before declaring a
# real pocket.  Does NOT count during the reacquisition window so that motion
# blur (track loss during a fast shot) cannot trigger a false penalty.
# (~333 ms at 30 fps)
CUE_POCKET_CONFIRM_FRAMES: int = 10

# Consecutive STABLE-TRACKING frames a colored ball must be absent before
# registering it as pocketed.  Guards against touching-ball detection failure
# where two adjacent balls merge into one blob for a few frames.
# (~167 ms at 30 fps)
BALL_POCKET_CONFIRM_FRAMES: int = 5


class GameTracker:
    """Unified game-state + ball-tracking controller.

    Usage (called by main.py every tick)::

        gt = GameTracker()
        ...
        if gt.needs_detection:
            raw = _detect_balls(frame, ...)
        else:
            raw = []
        balls = gt.update(frame, raw)
    """

    def __init__(
        self,
        match_radius_cm:       float = 5.0,
        ema_alpha:             float = 0.4,
        confirm_frames:        int   = 3,
        stale_frames:          int   = 5,
        dead_zone_cm:          float = 1.0,
        tracking_detect_every: int   = 5,
    ):
        # ── Tracking parameters ──────────────────────────────────────────────
        self._match_radius:      float = match_radius_cm
        self._alpha:             float = ema_alpha
        self._confirm_frames:    int   = confirm_frames
        self._stale_frames:      int   = stale_frames
        self._dead_zone:         float = dead_zone_cm
        self._tracking_interval: int   = max(1, tracking_detect_every)

        # ── State ────────────────────────────────────────────────────────────
        self.state: str = ST_WAITING_FOR_BALLS

        # ── Game bookkeeping ─────────────────────────────────────────────────
        self.stroke_count: int              = 0
        self.remaining_balls: List[str]     = []
        self.pocketed_balls:  List[str]     = []
        self.selected_target_color: Optional[str] = None
        self.cue_ball_pocketed: bool        = False   # True after a cue-ball pocket, cleared next stroke

        # ── Internal tracking state ──────────────────────────────────────────
        self._tracks: List[Dict]            = []
        self._confirmed_balls: List[Dict]   = []
        self._confirmed_pos: Dict[str, Tuple[float, float]] = {}

        # Snapshot taken right before entering DISTURBED so we can compare
        # after calm returns.
        self._pre_disturb_pos: Dict[str, Tuple[float, float]] = {}

        # Motion detection
        self._ref_frame: Optional[np.ndarray] = None
        self._calm_count: int = 0

        # Re-acquisition counter (counts down after DISTURBED -> TRACKING)
        self._reacq_remaining: int = 0

        # Tick counter for detection throttling
        self._tick_count: int = 0
        self._cue_ref_pos: Optional[Tuple[float, float]] = None
        self._last_known_cue_pos: Optional[Tuple[float, float]] = None

        # How many consecutive STABLE-TRACKING ticks white has been absent.
        # Paused during the reacquisition window (_reacq_remaining > 0) so that
        # motion blur (track loss on a fast shot) cannot trigger a false penalty.
        self._cue_hidden_frames: int = 0

        # Per-color consecutive-absence counters for colored balls.
        # Guards against touching-ball merges causing a false pocket.
        self._ball_hidden_frames: Dict[str, int] = {}

    # ── Public properties ────────────────────────────────────────────────────

    @property
    def needs_detection(self) -> bool:
        """True when the caller should run ball_detection this tick.

        WAITING_FOR_BALLS: every tick (need to find balls ASAP).
        TRACKING (re-acquiring): every tick (need fresh data).
        TRACKING (normal): every Nth tick (drift correction only).
        DISTURBED / GAME_OVER: never.
        """
        if self.state == ST_WAITING_FOR_BALLS:
            return True
        if self.state == ST_TRACKING and self.cue_ball_pocketed:
            return True   # actively search for white every tick until it returns
        if self.state == ST_TRACKING:
            if self._reacq_remaining > 0:
                return True
            return self._tick_count % self._tracking_interval == 0
        if self.state == ST_DISTURBED and self.cue_ball_pocketed:
            return True   # keep searching for white while table is disturbed
        return False

    # ── Lifecycle methods ────────────────────────────────────────────────────

    def start_game(
        self,
        colored_ball_colors: List[str],
        initial_balls: List[Dict],
        cue_ref_pos: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Transition WAITING_FOR_BALLS -> TRACKING.

        Called by the UI when the user presses *Start Game* after all 5 balls
        have been detected.
        """
        self.state = ST_TRACKING
        self.stroke_count = 0
        self.remaining_balls = list(colored_ball_colors)
        self.pocketed_balls = []
        self.selected_target_color = None
        self.cue_ball_pocketed = False
        self._confirmed_balls = list(initial_balls)
        self._confirmed_pos = _extract_positions(initial_balls)
        self._pre_disturb_pos = {}
        self._tracks = []
        self._ref_frame = None
        self._calm_count = 0
        self._reacq_remaining = 0
        self._tick_count = 0
        self._cue_ref_pos = cue_ref_pos
        self._last_known_cue_pos = None
        self._cue_hidden_frames = 0
        self._ball_hidden_frames = {}

    def reset(self) -> None:
        """Return to WAITING_FOR_BALLS and clear all game state."""
        self.state = ST_WAITING_FOR_BALLS
        self.stroke_count = 0
        self.remaining_balls = []
        self.pocketed_balls = []
        self.selected_target_color = None
        self.cue_ball_pocketed = False
        self._confirmed_balls = []
        self._confirmed_pos = {}
        self._pre_disturb_pos = {}
        self._tracks = []
        self._ref_frame = None
        self._calm_count = 0
        self._reacq_remaining = 0
        self._tick_count = 0
        self._cue_ref_pos = None
        self._last_known_cue_pos = None
        self._cue_hidden_frames = 0
        self._ball_hidden_frames = {}

    # ── Target / ball lookup helpers ─────────────────────────────────────────

    def select_target(self, color: str) -> bool:
        """Set the selected target ball by colour.  Returns True if accepted."""
        if color == self.selected_target_color:
            return True
        if color not in self.remaining_balls:
            return False
        self.selected_target_color = color
        return True

    @staticmethod
    def get_cue_ball(balls: List[Dict]) -> Optional[Dict]:
        for ball in balls:
            if ball.get('is_cue'):
                return ball
        return None

    def get_target_ball(self, balls: List[Dict]) -> Optional[Dict]:
        if self.selected_target_color is None:
            return None
        for ball in balls:
            if ball.get('color') == self.selected_target_color and not ball.get('is_cue'):
                return ball
        return None

    # ── Main per-tick entry point ────────────────────────────────────────────

    def update(self, frame: np.ndarray, raw_balls: List[Dict]) -> List[Dict]:
        """Advance one tick.  Returns the ball list the UI / trajectory code
        should use this frame.

        Parameters
        ----------
        frame     : current camera frame (always provided).
        raw_balls : detections from ball_detection, or ``[]`` when detection
                    was skipped (DISTURBED / GAME_OVER).
        """
        self._tick_count += 1

        if self.state == ST_WAITING_FOR_BALLS:
            return self._tick_waiting(raw_balls)

        if self.state == ST_GAME_OVER:
            return list(self._confirmed_balls)

        if self.state == ST_TRACKING:
            return self._tick_tracking(frame, raw_balls)

        if self.state == ST_DISTURBED:
            return self._tick_disturbed(frame, raw_balls)

        return list(self._confirmed_balls)

    # ── Per-state tick handlers ──────────────────────────────────────────────

    def _tick_waiting(self, raw_balls: List[Dict]) -> List[Dict]:
        """WAITING_FOR_BALLS: run tracking so the UI can see detected balls,
        but don't apply any game logic."""
        if raw_balls:
            self._do_tracking(raw_balls, create_new=True)
        return self._emit_confirmed()

    def _tick_tracking(self, frame: np.ndarray, raw_balls: List[Dict]) -> List[Dict]:
        """TRACKING: detect motion, update tracks, check for strokes.

        Strokes are only detected through the DISTURBED -> re-acquisition
        path.  Balls are assumed static in TRACKING; the EMA is used purely
        to refine accuracy, not to detect movement.  Comparing cumulative
        EMA drift against a fixed snapshot would cause false positives from
        detection noise.
        """

        # 1. Cheap motion check — may transition to DISTURBED
        if self._check_motion(frame):
            self._pre_disturb_pos = dict(self._confirmed_pos)
            self.state = ST_DISTURBED
            self._calm_count = 0
            self._cue_hidden_frames = 0   # reset occlusion counter on new disturbance
            return list(self._confirmed_balls)

        # 2. Feed detections into EMA tracker
        if raw_balls:
            self._do_tracking(raw_balls, create_new=True)

        # 3a. Track white ball occlusion — FIX #5 (motion blur).
        #     Only count while the tracker is stable (not during reacquisition).
        #     During reacq, the EMA is still warming up so white being absent
        #     is expected and must NOT be counted toward the pocket threshold.
        current_snapshot = self._emit_confirmed()
        white_now_visible = any(b.get('color') == 'white' for b in current_snapshot)
        if not self.cue_ball_pocketed:
            if not white_now_visible and self._reacq_remaining == 0:
                self._cue_hidden_frames += 1
                # Threshold crossed: declare pocket immediately.
                if self._cue_hidden_frames == CUE_POCKET_CONFIRM_FRAMES:
                    self.cue_ball_pocketed = True
                    self.stroke_count += CUE_BALL_PENALTY
                    self._last_known_cue_pos = self._confirmed_pos.get('white')
            elif white_now_visible:
                self._cue_hidden_frames = 0

        # 3b. Track per-color absence — FIX #6 (touching balls).
        #     Only count while stable (not during reacq) for the same reason.
        if self._reacq_remaining == 0:
            for color in list(self._confirmed_pos.keys()):
                if color == 'white':
                    continue
                color_visible = any(b.get('color') == color for b in current_snapshot)
                if not color_visible:
                    self._ball_hidden_frames[color] = \
                        self._ball_hidden_frames.get(color, 0) + 1
                else:
                    self._ball_hidden_frames[color] = 0

        # 4. During re-acquisition (just returned from DISTURBED), build
        #    tracks without checking for strokes yet.
        if self._reacq_remaining > 0:
            self._reacq_remaining -= 1
            if self._reacq_remaining == 0:
                self._check_for_stroke(current_snapshot, self._pre_disturb_pos)

        self._confirmed_balls = current_snapshot
        return list(self._confirmed_balls)

    def _tick_disturbed(self, frame: np.ndarray, raw_balls: List[Dict]) -> List[Dict]:
        """DISTURBED: freeze output, monitor for calm.

        When cue_ball_pocketed is True we continue tracking so that white
        can be found before the DISTURBED cycle resolves.
        """
        if self.cue_ball_pocketed and raw_balls:
            self._do_tracking(raw_balls, create_new=True)

        has_motion = self._check_motion(frame)

        if has_motion:
            self._calm_count = 0
        else:
            self._calm_count += 1
            if self._calm_count >= CALM_FRAMES_REQUIRED:
                self.state = ST_TRACKING
                self._tracks = []
                self._ref_frame = None
                self._calm_count = 0
                self._reacq_remaining = REACQUISITION_TICKS

        return list(self._confirmed_balls)

    # ── Stroke / pocket logic ────────────────────────────────────────────────

    def _check_for_stroke(
        self,
        current_snapshot: List[Dict],
        reference_pos: Dict[str, Tuple[float, float]],
    ) -> None:
        """Compare *current_snapshot* to *reference_pos* and register a stroke
        if any ball moved significantly or was pocketed."""
        current_pos = _extract_positions(current_snapshot)
        # Scratch-return: cue was pocketed before, white is now confirmed back.
        if self.cue_ball_pocketed and 'white' in current_pos:
            self._register_stroke(current_snapshot, current_pos)
            return
        if not reference_pos:
            return
        if _any_ball_moved(reference_pos, current_pos, SHOT_THRESHOLD_CM) or \
           _any_ball_disappeared(reference_pos, current_pos):
            self._register_stroke(current_snapshot, current_pos)

    def _register_stroke(
        self,
        new_snapshot: List[Dict],
        new_pos: Dict[str, Tuple[float, float]],
    ) -> None:
        """Record a stroke: increment count, update confirmed positions,
        pocket any colored balls that disappeared, check for game-over.

        Cue-ball pocket detection is handled entirely in _tick_tracking via
        the _cue_hidden_frames threshold.  _register_stroke only clears the
        flag (on scratch-return) or leaves it as-is for colored balls.
        """
        self.stroke_count += 1
        self._cue_hidden_frames = 0
        # Capture before resetting so the touching-ball guard below can read
        # the pre-stroke absence counts (same pattern as cue_hidden_frames).
        ball_hidden_snapshot = self._ball_hidden_frames
        self._ball_hidden_frames = {}  # reset for the next stroke

        # Scratch-return confirmation: cue came back -> clear the pocketed flag.
        # The penalty was already applied when cue_ball_pocketed was first set.
        if self.cue_ball_pocketed and 'white' in new_pos:
            self.cue_ball_pocketed = False

        # FIX #6 (touching balls): only pocket a colored ball if it has been
        # absent for BALL_POCKET_CONFIRM_FRAMES consecutive stable-tracking ticks.
        # This prevents a single-frame detection failure (e.g. two balls touching
        # and merging into one blob) from registering a false pocket.
        for color in list(self._confirmed_pos.keys()):
            if color == 'white':
                continue
            if color in self.remaining_balls and color not in new_pos:
                hidden = ball_hidden_snapshot.get(color, 0)
                if hidden >= BALL_POCKET_CONFIRM_FRAMES:
                    self.pocketed_balls.append(color)
                    self.remaining_balls.remove(color)
                    if self.selected_target_color == color:
                        self.selected_target_color = None

        self._confirmed_balls = list(new_snapshot)
        self._confirmed_pos = dict(new_pos)

        if len(self.remaining_balls) == 0:
            self.state = ST_GAME_OVER

    # ── Motion detection ─────────────────────────────────────────────────────

    def _check_motion(self, frame: np.ndarray) -> bool:
        """Cheap pixel-diff motion detector.  Returns True if motion detected.

        The reference frame is only replaced when motion IS detected (or on
        first call).  This keeps the baseline stable during calm periods so
        the diff accumulates real changes rather than just comparing two
        consecutive 33 ms frames, and avoids a full-frame copy every tick.
        """
        if self._ref_frame is None:
            self._ref_frame = frame.copy()
            return False

        diff = cv2.absdiff(frame, self._ref_frame)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) if diff.ndim == 3 else diff
        changed = int(np.count_nonzero(gray > MOTION_DIFF_THRESH))
        total = gray.shape[0] * gray.shape[1]
        has_motion = total > 0 and changed / total > MOTION_PIXEL_FRAC

        if has_motion:
            self._ref_frame = frame.copy()
        return has_motion

    # ── EMA ball tracking ────────────────────────────────────────────────────

    def _do_tracking(self, raw_balls: List[Dict], create_new: bool) -> None:
        """Greedy nearest-neighbour matching with EMA position smoothing."""
        n_existing = len(self._tracks)
        matched_tracks: set = set()
        matched_dets:   set = set()

        for di, det in enumerate(raw_balls):
            det_cm = det.get('center_cm')
            if det_cm is None:
                continue
            best_ti   = -1
            best_dist = self._match_radius
            for ti in range(n_existing):
                if ti in matched_tracks:
                    continue
                trk = self._tracks[ti]
                dx = det_cm[0] - trk['smooth_cm'][0]
                dy = det_cm[1] - trk['smooth_cm'][1]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_ti   = ti
            if best_ti >= 0:
                matched_tracks.add(best_ti)
                matched_dets.add(di)
                trk = self._tracks[best_ti]
                if best_dist > self._dead_zone:
                    a   = self._alpha
                    old = trk['smooth_cm']
                    trk['smooth_cm'] = (
                        old[0] * (1 - a) + det_cm[0] * a,
                        old[1] * (1 - a) + det_cm[1] * a,
                    )
                trk['center'] = det['center']
                trk['color']  = det['color']
                trk['is_cue'] = det.get('is_cue', False)
                trk['radius'] = det.get('radius', 18)
                trk['age']   += 1
                trk['stale']  = 0

        if create_new:
            # FIX #7 (foreign objects): build the set of colors we actually
            # expect to see on the table right now.  In WAITING_FOR_BALLS all
            # standard colors are valid; during play only remaining + white.
            if self.state == ST_WAITING_FOR_BALLS:
                allowed_colors = {'white', 'red', 'blue', 'green', 'yellow'}
            else:
                allowed_colors = set(self.remaining_balls) | {'white'}

            for di, det in enumerate(raw_balls):
                if di in matched_dets:
                    continue
                det_cm = det.get('center_cm')
                if det_cm is None:
                    continue
                det_color = det.get('color', '')

                # Reject detections whose color is not expected (chalk, cloth, etc.)
                if det_color not in allowed_colors:
                    continue

                # One-track-per-color cap: there is exactly one physical ball
                # per color; a second detection of the same color is a false positive.
                if any(t['color'] == det_color for t in self._tracks):
                    continue

                # De-duplication guard: skip if spatially too close to an existing
                # track of any color (avoids ghost doubles at same position).
                too_close = False
                for trk in self._tracks:
                    dx = det_cm[0] - trk['smooth_cm'][0]
                    dy = det_cm[1] - trk['smooth_cm'][1]
                    if (dx * dx + dy * dy) ** 0.5 < self._match_radius:
                        too_close = True
                        break
                if too_close:
                    continue

                self._tracks.append({
                    'smooth_cm': det_cm,
                    'center':    det['center'],
                    'color':     det_color,
                    'is_cue':    det.get('is_cue', False),
                    'radius':    det.get('radius', 18),
                    'age':       1,
                    'stale':     0,
                })

        for ti in range(n_existing):
            if ti not in matched_tracks:
                self._tracks[ti]['stale'] += 1

        self._tracks = [t for t in self._tracks if t['stale'] < self._stale_frames]

    def _emit_confirmed(self) -> List[Dict]:
        """Return tracks that have been seen for at least *confirm_frames*."""
        return [
            {
                'center':    t['center'],
                'center_cm': t['smooth_cm'],
                'color':     t['color'],
                'is_cue':    t['is_cue'],
                'radius':    t['radius'],
            }
            for t in self._tracks
            if t['age'] >= self._confirm_frames
        ]


# ─── Module-level helpers ────────────────────────────────────────────────────

def _extract_positions(balls: List[Dict]) -> Dict[str, Tuple[float, float]]:
    """Build a colour -> (x_cm, y_cm) map, skipping balls with no coords."""
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
    """True if any ball present in both dicts moved more than *threshold_cm*."""
    for color, (cx, cy) in current.items():
        if color not in prev:
            continue
        px, py = prev[color]
        if (cx - px) ** 2 + (cy - py) ** 2 > threshold_cm ** 2:
            return True
    return False


def _any_ball_disappeared(
    old_pos: Dict[str, Tuple[float, float]],
    new_pos: Dict[str, Tuple[float, float]],
) -> bool:
    """True if any ball present in *old_pos* is absent from *new_pos*."""
    for color in old_pos:
        if color not in new_pos:
            return True
    return False
