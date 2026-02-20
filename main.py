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

    # Golf mode: pocket all colored balls.
    # Color strings must match Ball_Detection output.
    COLORED_BALLS = ['orange', 'yellow', 'blue', 'bordeaux']
    game = GolfGame(COLORED_BALLS)

    # Mutable container so the on_tick closure can reference window
    # before it is fully constructed.
    _ctx: Dict = {}

    def on_tick() -> None:
        window = _ctx.get('window')
        if window is None:
            return
        _run_tick(window, game)

    window = BilliardsApp(tick_callback=on_tick)
    _ctx['window'] = window
    window.show()

    sys.exit(app_qt.exec_())


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame orchestration tick
# ─────────────────────────────────────────────────────────────────────────────

def _run_tick(window, game: GolfGame) -> None:
    """
    Single orchestration tick, called every 33 ms.
    Order: grab → detect → adapt → game update → click → trajectory → render.
    """

    # 1. Grab raw camera frame
    frame = window.grab_frame()
    if frame is None:
        return

    # 2. Detect balls (raw frame, no warping done here)
    balls = _detect_balls(frame, window.cam_H, window.table_corners)

    # 3. Advance game state machine
    game.update(balls)

    # 4. Process user ball-selection clicks (only in WAITING state)
    if game.state == STATE_WAITING:
        for x_cm, y_cm in window.consume_pending_clicks():
            color = _find_ball_color_at(x_cm, y_cm, balls)
            if color and color != 'white':
                game.select_target(color)
    else:
        # Drain clicks so they don't accumulate during IN_PROGRESS/RESOLVING
        window.consume_pending_clicks()

    # 5. Compute trajectory (only when waiting for next shot)
    cue_path:    List[Tuple[float, float]] = []
    target_path: List[Tuple[float, float]] = []

    if game.state == STATE_WAITING:
        cue_ball = game.get_cue_ball(balls)
        if cue_ball is not None and cue_ball.get('center_cm') is not None:
            cue_cm = cue_ball['center_cm']
            target_ball = game.get_target_ball(balls)

            if target_ball is not None and target_ball.get('center_cm') is not None:
                # User has selected a specific target ball
                cue_path, target_path = calculate_path(
                    cue_cm,
                    target_ball['center_cm'],
                    window.selected_pocket_cm,   # None → auto-select pocket
                )
            else:
                # No target selected — auto-suggest the optimal shot
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

    # 6. Render
    window.render(
        frame         = frame,
        balls         = balls,
        cue_path      = cue_path,
        target_path   = target_path,
        game_state    = game.state,
        stroke_count  = game.stroke_count,
        remaining     = game.remaining_balls,
        selected_color = game.selected_target_color,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Ball detection  (adapter between Ball_Detection.py output and internal format)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_balls(
    frame:         np.ndarray,
    cam_H:         Optional[np.ndarray],
    table_corners: Optional[np.ndarray],
) -> List[Dict]:
    """
    Run ball detection and convert to internal dict format.

    Returns list of dicts:
        {
            'center':    (int, int),            # camera pixel coords
            'center_cm': (float, float) | None, # table cm coords, or None
            'color':     str,
            'is_cue':    bool,
            'radius':    int,
        }
    """
    if DETECTION_AVAILABLE and table_corners is not None:
        try:
            raw = _detect_balls_with_color(
                frame_bgr     = frame,
                table_corners = table_corners,
                ref_path      = 'ref.jpeg',
                table_size_cm = (TABLE_WIDTH_CM, TABLE_HEIGHT_CM),
            )
            if raw is not None and len(raw) > 0:
                return _adapt_detections(raw, cam_H)
        except Exception as exc:
            print(f'[main] Detection error: {exc}')

    # Fallback: mock ball positions
    return _mock_detect_balls(frame, cam_H)


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
