"""
main.py — Orchestrator for the Billiards Golf Game.

This is the only file that imports from all other modules.

Pipeline (called every 33 ms via QTimer):
  1. app.grab_frame()               — raw camera frame
  2. detect_balls_with_color(...)   — ball positions in camera pixels
  3. _adapt_detections(...)         — convert to internal dict format with center_cm
  4. game.update(balls)             — advance state machine
  5. app.consume_pending_clicks()   — process user ball-selection clicks
  6. calculate_path / suggest_best_shot — compute trajectory in table cm
  7. app.render(...)                — draw overlays and display

Coordinate system rule:
  This file is the only place that converts camera pixels → table cm
  (via _adapt_detections, using cam_H).
  Everything passed to game.update() and trajectory functions is in table cm.
"""

import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── Project modules ───────────────────────────────────────────────────────────
from golf_game import (
    GolfGame,
    STATE_WAITING,
)
from trajectory import calculate_path, suggest_best_shot

# ── Table constants (keep in sync with golf_game.py / trajectory.py) ─────────
TABLE_WIDTH_CM  = 122.0
TABLE_HEIGHT_CM = 61.0

# ── Ball Detection import (Person 1's module) ─────────────────────────────────
# detect_balls_with_color(frame_bgr, table_corners, ref_path, ...) → np.ndarray (N,3)
# Each row: [x_cam_px, y_cam_px, color_str]
try:
    from Ball_Detection import detect_balls_with_color as _detect_balls_with_color
    DETECTION_AVAILABLE = True
    print('[main] Ball_Detection module loaded.')
except (ImportError, AttributeError) as _e:
    DETECTION_AVAILABLE = False
    print(f'[main] Ball_Detection not available ({_e}); using mock detection.')


# ─────────────────────────────────────────────────────────────────────────────
# Temporal ball tracker — stabilises detections across video frames
# ─────────────────────────────────────────────────────────────────────────────

_TRK_TRACKING  = 'TRACKING'
_TRK_DISTURBED = 'DISTURBED'
_TRK_SETTLING  = 'SETTLING'

# Minimum ball displacement (cm) to consider a real shot vs. noise.
_SHOT_THRESHOLD_CM = 5.0

# Frame-diff motion threshold: fraction of table-area pixels that must change
# to trigger a disturbance from the TRACKING (idle) state.
_MOTION_PIXEL_FRAC = 0.02
_MOTION_DIFF_THRESH = 30   # per-pixel intensity threshold for "changed"


class _BallTracker:
    """
    Three-state tracker that avoids running the expensive Ball_Detection
    pipeline while the table is idle.

    TRACKING  — table is stable.  Output is a frozen snapshot.
                A cheap frame-diff checks for motion; if motion is detected
                the tracker transitions to DISTURBED.
                Ball_Detection is NOT called in this state.

    DISTURBED — something is moving on the table (hand, cue, shot).
                Ball_Detection is NOT called.  A cheap frame-diff watches
                for motion to *stop*.  After ``quiet_frames`` consecutive
                quiet frames the tracker transitions to SETTLING.
                Output stays frozen to the pre-disturbance snapshot.

    SETTLING  — disturbance cleared; re-acquiring ball positions.
                Ball_Detection IS called; new tracks are built.
                After ``settle_frames`` clean frames the new positions are
                compared to the pre-disturbance snapshot.
                  • Significant change  → real shot.  New snapshot becomes
                    active (the game sees a position jump and counts a stroke).
                  • No significant change → false alarm.  Old snapshot is
                    kept (the game sees nothing).
    """

    def __init__(
        self,
        match_radius_cm:  float = 5.0,
        ema_alpha:        float = 0.4,
        confirm_frames:   int   = 3,
        stale_frames:     int   = 5,
        dead_zone_cm:     float = 1.0,
        settle_frames:    int   = 10,
        quiet_frames:     int   = 5,
    ):
        self._tracks:          List[Dict] = []
        self._match_radius:    float      = match_radius_cm
        self._alpha:           float      = ema_alpha
        self._confirm_frames:  int        = confirm_frames
        self._stale_frames:    int        = stale_frames
        self._dead_zone:       float      = dead_zone_cm
        self._settle_required: int        = settle_frames
        self._quiet_required:  int        = quiet_frames

        self._state:           str        = _TRK_SETTLING
        self._stable_snapshot: List[Dict] = []
        self._pre_disturb_snapshot: List[Dict] = []
        self._expected_count:  int        = 0
        self._settle_count:    int        = 0
        self._quiet_count:     int        = 0
        self._ref_frame: Optional[np.ndarray] = None

    # ── public: called every tick from _run_tick ────────────────────────────

    @property
    def needs_detection(self) -> bool:
        """True when the orchestrator should run Ball_Detection this tick."""
        return self._state == _TRK_SETTLING

    def check_motion(self, frame: np.ndarray) -> None:
        """Cheap frame-diff motion check (TRACKING and DISTURBED states).

        TRACKING  → DISTURBED  when motion is detected.
        DISTURBED → SETTLING   after ``quiet_frames`` consecutive quiet frames.
        """
        if self._ref_frame is None:
            self._ref_frame = frame.copy()
            self._quiet_count = 0
            return

        diff = cv2.absdiff(frame, self._ref_frame)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) if diff.ndim == 3 else diff
        changed = int(np.count_nonzero(gray > _MOTION_DIFF_THRESH))
        total   = gray.shape[0] * gray.shape[1]
        has_motion = total > 0 and changed / total > _MOTION_PIXEL_FRAC

        if self._state == _TRK_TRACKING:
            if has_motion:
                self._pre_disturb_snapshot = list(self._stable_snapshot)
                self._state = _TRK_DISTURBED
                self._settle_count = 0
                self._quiet_count = 0
                self._ref_frame = None
        elif self._state == _TRK_DISTURBED:
            if has_motion:
                self._quiet_count = 0
                self._ref_frame = frame.copy()
            else:
                self._quiet_count += 1
                if self._quiet_count >= self._quiet_required:
                    self._state = _TRK_SETTLING
                    self._settle_count = 0
                    self._ref_frame = None

    def update(self, raw_balls: List[Dict]) -> List[Dict]:
        """Process raw detections (SETTLING only)."""

        if self._state != _TRK_SETTLING:
            return list(self._stable_snapshot)

        self._do_tracking(raw_balls, create_new=True)

        self._settle_count += 1
        if self._settle_count >= self._settle_required:
            new_snapshot = self._emit_confirmed()
            if self._snapshot_changed(self._pre_disturb_snapshot, new_snapshot):
                self._stable_snapshot = new_snapshot
            self._expected_count = len(self._stable_snapshot)
            self._state = _TRK_TRACKING
            self._ref_frame = None
            return list(self._stable_snapshot)

        return list(self._stable_snapshot)

    def get_snapshot(self) -> List[Dict]:
        """Return the current frozen ball snapshot (used during TRACKING)."""
        return list(self._stable_snapshot)

    # ── internal ────────────────────────────────────────────────────────────

    def _snapshot_changed(
        self, old_snap: List[Dict], new_snap: List[Dict],
    ) -> bool:
        """True if any ball moved significantly or disappeared (pocketed)."""
        if not old_snap:
            return bool(new_snap)

        old_by_color = {b['color']: b['center_cm'] for b in old_snap}
        new_by_color = {b['color']: b['center_cm'] for b in new_snap}

        for color in old_by_color:
            if color == 'white':
                continue
            if color not in new_by_color:
                return True

        for color, (nx, ny) in new_by_color.items():
            if color in old_by_color:
                ox, oy = old_by_color[color]
                if ((nx - ox) ** 2 + (ny - oy) ** 2) > _SHOT_THRESHOLD_CM ** 2:
                    return True

        return False

    def _do_tracking(self, raw_balls: List[Dict], create_new: bool) -> List[Dict]:
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
                dx  = det_cm[0] - trk['smooth_cm'][0]
                dy  = det_cm[1] - trk['smooth_cm'][1]
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
            for di, det in enumerate(raw_balls):
                if di in matched_dets:
                    continue
                det_cm = det.get('center_cm')
                if det_cm is None:
                    continue
                self._tracks.append({
                    'smooth_cm': det_cm,
                    'center':    det['center'],
                    'color':     det['color'],
                    'is_cue':    det.get('is_cue', False),
                    'radius':    det.get('radius', 18),
                    'age':       1,
                    'stale':     0,
                })

        for ti in range(n_existing):
            if ti not in matched_tracks:
                self._tracks[ti]['stale'] += 1

        self._tracks = [
            t for t in self._tracks if t['stale'] < self._stale_frames
        ]

        return self._emit_confirmed()

    def _emit_confirmed(self) -> List[Dict]:
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


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        from PyQt5.QtWidgets import QApplication
    except ImportError:
        print('ERROR: PyQt5 is required. Install with:  pip install PyQt5')
        sys.exit(1)

    from app import BilliardsApp

    app_qt = QApplication(sys.argv)
    app_qt.setApplicationName('Billiards Golf Game')

    COLORED_BALLS = ['orange', 'yellow', 'blue', 'bordeaux']
    game    = GolfGame(COLORED_BALLS)
    tracker = _BallTracker()

    _ctx: Dict = {}

    def on_tick() -> None:
        window = _ctx.get('window')
        if window is None:
            return
        _run_tick(window, game, tracker)

    window = BilliardsApp(tick_callback=on_tick)
    _ctx['window'] = window
    window.show()

    sys.exit(app_qt.exec_())


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame orchestration tick
# ─────────────────────────────────────────────────────────────────────────────

def _run_tick(window, game: GolfGame, tracker: _BallTracker) -> None:
    """
    Single orchestration tick, called every 33 ms.

    Detection is only active when the tracker is NOT in TRACKING state.
    During TRACKING the table is idle — a cheap frame-diff watches for
    motion and the frozen snapshot is reused.
    """

    # 1. Grab raw camera frame
    frame = window.grab_frame()
    if frame is None:
        return

    det_cfg = getattr(window, 'detection_config', None)

    if window.use_mock:
        balls = _detect_balls(
            frame, window.cam_H, window.table_corners,
            window.ref_path, True, det_cfg,
        )
    elif tracker.needs_detection:
        # SETTLING — run the expensive detection pipeline
        raw_balls = _detect_balls(
            frame, window.cam_H, window.table_corners,
            window.ref_path, False, det_cfg,
        )
        balls = tracker.update(raw_balls)
    else:
        # TRACKING or DISTURBED — cheap frame-diff only
        tracker.check_motion(frame)
        balls = tracker.get_snapshot()

    # 2. Advance game state machine
    game.update(balls)

    # 5. Drain pending clicks (pocket selection is handled inside app.py)
    window.consume_pending_clicks()

    # 6. Compute trajectory only when waiting AND at least 2 balls visible
    #    (need cue ball + at least one target)
    cue_path:    List[Tuple[float, float]] = []
    target_path: List[Tuple[float, float]] = []

    if game.state == STATE_WAITING and len(balls) >= 2:
        cue_ball = game.get_cue_ball(balls)
        if cue_ball is not None and cue_ball.get('center_cm') is not None:
            cue_cm = cue_ball['center_cm']
            target_ball = game.get_target_ball(balls)

            if target_ball is not None and target_ball.get('center_cm') is not None:
                cue_path, target_path = calculate_path(
                    cue_cm,
                    target_ball['center_cm'],
                    window.selected_pocket_cm,
                )
            else:
                remaining_dicts = [
                    b for b in balls
                    if not b.get('is_cue')
                    and b.get('center_cm') is not None
                    and b.get('color') in game.remaining_balls
                ]
                if remaining_dicts:
                    best_ball, cue_path, target_path = suggest_best_shot(
                        cue_cm, remaining_dicts
                    )
                    if best_ball is not None:
                        game.select_target(best_ball['color'])

    # 7. Render
    window.render(
        frame          = frame,
        balls          = balls,
        cue_path       = cue_path,
        target_path    = target_path,
        game_state     = game.state,
        stroke_count   = game.stroke_count,
        remaining      = game.remaining_balls,
        selected_color = game.selected_target_color,
        tracker_state  = tracker._state if not window.use_mock else _TRK_TRACKING,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Ball detection  (adapter between Ball_Detection.py output and internal format)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_balls(
    frame:         np.ndarray,
    cam_H:         Optional[np.ndarray],
    table_corners: Optional[np.ndarray],
    ref_path:      Optional[str] = None,
    use_mock:      bool = False,
    config:        object = None,
) -> List[Dict]:
    """
    Run ball detection and convert to internal dict format.

    Returns only actually-detected balls.  Mock balls are generated only
    when ``use_mock`` is True (user selected "no camera" in setup).

    Returns list of dicts:
        {
            'center':    (int, int),            # camera pixel coords
            'center_cm': (float, float) | None, # table cm coords, or None
            'color':     str,
            'is_cue':    bool,
            'radius':    int,
        }
    """
    if DETECTION_AVAILABLE and table_corners is not None and ref_path is not None:
        try:
            raw = _detect_balls_with_color(
                frame_bgr     = frame,
                table_corners = table_corners,
                ref_path      = ref_path,
                table_size_cm = (TABLE_WIDTH_CM, TABLE_HEIGHT_CM),
                config        = config,
            )
            if raw is not None and len(raw) > 0:
                return _adapt_detections(raw, cam_H)
        except Exception as exc:
            print(f'[main] Detection error: {exc}')
        return []

    if use_mock:
        return _mock_detect_balls(frame, cam_H)

    return []


def _adapt_detections(
    raw:   np.ndarray,
    cam_H: Optional[np.ndarray],
) -> List[Dict]:
    """
    Convert detect_balls_with_color() numpy output → internal dict list.
    Applies cam_H to produce center_cm for each ball.

    Parameters
    ----------
    raw   : shape (N, 3) array [[x_cam_px, y_cam_px, color_str], ...]
    cam_H : (3,3) float32 homography camera → table cm, or None
    """
    balls: List[Dict] = []
    for row in raw:
        x_cam = float(row[0])
        y_cam = float(row[1])
        color = str(row[2]).lower().strip()

        center_cm: Optional[Tuple[float, float]] = None
        if cam_H is not None:
            pt        = np.array([[[x_cam, y_cam]]], dtype=np.float32)
            t         = cv2.perspectiveTransform(pt, cam_H)
            center_cm = (float(t[0][0][0]), float(t[0][0][1]))

        balls.append({
            'center':    (int(x_cam), int(y_cam)),
            'center_cm': center_cm,
            'color':     color,
            'is_cue':    color == 'white',
            'radius':    18,   # Ball_Detection doesn't expose per-ball radius
        })
    return balls


def _mock_detect_balls(
    frame: np.ndarray,
    cam_H: Optional[np.ndarray],
) -> List[Dict]:
    """
    Generate mock ball positions for demo / testing when Ball_Detection is
    unavailable or when table corners have not been calibrated yet.
    Positions are fixed fractions of the frame size.
    """
    h, w = frame.shape[:2]
    raw_positions = [
        (0.50, 0.50, 'white',    True),
        (0.35, 0.38, 'orange',   False),
        (0.65, 0.33, 'yellow',   False),
        (0.28, 0.62, 'blue',     False),
        (0.72, 0.60, 'bordeaux', False),
    ]
    balls: List[Dict] = []
    for fx, fy, color, is_cue in raw_positions:
        px = int(w * fx)
        py = int(h * fy)
        center_cm: Optional[Tuple[float, float]] = None
        if cam_H is not None:
            pt        = np.array([[[float(px), float(py)]], ], dtype=np.float32)
            t         = cv2.perspectiveTransform(pt, cam_H)
            center_cm = (float(t[0][0][0]), float(t[0][0][1]))
        balls.append({
            'center':    (px, py),
            'center_cm': center_cm,
            'color':     color,
            'is_cue':    is_cue,
            'radius':    18,
        })
    return balls


# ─────────────────────────────────────────────────────────────────────────────
# Helper: find which ball color is at a table-cm click position
# ─────────────────────────────────────────────────────────────────────────────

def _find_ball_color_at(
    x_cm:      float,
    y_cm:      float,
    balls:     List[Dict],
    tolerance: float = 3.5,   # cm — slightly larger than BALL_RADIUS_CM=2.875
) -> Optional[str]:
    """
    Return the color of the ball whose center_cm is within tolerance of (x_cm, y_cm),
    or None if no ball is close enough.
    """
    for ball in balls:
        cm = ball.get('center_cm')
        if cm is None:
            continue
        dx = x_cm - cm[0]
        dy = y_cm - cm[1]
        if dx * dx + dy * dy <= tolerance ** 2:
            return ball.get('color')
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Script entry
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    main()
