"""
main.py — Orchestrator for the Billiards Golf Game.

This is the only file that imports from all other modules.

Pipeline (called every 33 ms via QTimer):
  1. app.grab_frame()               — raw camera frame
  2. detect_balls_with_color(...)   — ball positions in camera pixels
  3. _adapt_detections(...)         — convert to internal dict format with center_cm
  4. gt.update(frame, raw_balls)    — advance unified state machine
  5. app.consume_pending_clicks()   — process user ball-selection clicks
  6. suggest_best_shot              — compute trajectory in table cm
  7. app.render(...)                — draw overlays and display

Coordinate system rule:
  This file is the only place that converts camera pixels → table cm
  (via _adapt_detections, using cam_H).
  Everything passed to gt.update() and trajectory functions is in table cm.
"""

import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── Project modules ───────────────────────────────────────────────────────────
from game_tracker import GameTracker, ST_TRACKING, TABLE_WIDTH_CM, TABLE_HEIGHT_CM
from trajectory import suggest_best_shot

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
# Font loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_opticue_fonts(app) -> None:
    """Load OptiCue custom fonts and set the application default font."""
    import os
    from PyQt5.QtGui import QFontDatabase, QFont
    font_dir = os.path.join(os.path.dirname(__file__), 'assets', 'fonts')
    for fname in [
        'PlayfairDisplay-Bold.ttf', 'PlayfairDisplay-Black.ttf',
        'PlayfairDisplay-Italic.ttf', 'Lora-Regular.ttf', 'Lora-Italic.ttf',
        'DMSans-Regular.ttf', 'DMSans-Medium.ttf',
    ]:
        path = os.path.join(font_dir, fname)
        if os.path.exists(path):
            QFontDatabase.addApplicationFont(path)
        else:
            print(f'[fonts] WARNING: {fname} not found at {path}')
    app.setFont(QFont('DM Sans', 10))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import Qt
    except ImportError:
        print('ERROR: PyQt5 is required. Install with:  pip install PyQt5')
        sys.exit(1)

    from app import BilliardsApp

    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app_qt = QApplication(sys.argv)
    app_qt.setApplicationName('Billiards Golf Game')

    # Load custom fonts before window creation so they are available in the stylesheet
    _load_opticue_fonts(app_qt)

    # Gentle DPI-based font scaling: only scales font-size px values in the global
    # stylesheet, leaving all widget geometry unchanged so layouts cannot overflow.
    import re, app as _app_mod
    _screen_dpi = app_qt.primaryScreen().logicalDotsPerInch()
    _font_scale = min(_screen_dpi / 96.0, 1.5)   # cap at 1.5× to prevent overflow
    if _font_scale > 1.05:
        _app_mod._APP_STYLE = re.sub(
            r'font-size:\s*(\d+)px',
            lambda m: f'font-size: {round(int(m.group(1)) * _font_scale)}px',
            _app_mod._APP_STYLE,
        )

    gt = GameTracker()

    _ctx: Dict = {}

    def on_tick() -> None:
        window = _ctx.get('window')
        if window is None:
            return
        _run_tick(window, gt)

    window = BilliardsApp(tick_callback=on_tick)
    window.game = gt
    window._last_balls = []
    _ctx['window'] = window

    window.show()

    sys.exit(app_qt.exec_())


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame orchestration tick
# ─────────────────────────────────────────────────────────────────────────────

def _run_tick(window, gt: GameTracker) -> None:
    """Single orchestration tick, called every 33 ms."""

    frame = window.grab_frame()
    if frame is None:
        return

    # Run detection only when the state machine needs it
    raw_balls: List[Dict] = []
    if window.use_mock:
        raw_balls = _detect_balls(
            frame, window.cam_H, window.table_corners,
            window.ref_path, True,
        )
    elif gt.needs_detection:
        raw_balls = _detect_balls(
            frame, window.cam_H, window.table_corners,
            window.ref_path, False,
        )

    balls = gt.update(frame, raw_balls)
    window._last_balls = balls

    window.consume_pending_clicks()

    # Compute trajectory only while tracking with at least 2 balls visible
    cue_path:    List[Tuple[float, float]] = []
    target_path: List[Tuple[float, float]] = []

    if gt.state == ST_TRACKING and len(balls) >= 2:
        cue_ball = gt.get_cue_ball(balls)
        if cue_ball is not None and cue_ball.get('center_cm') is not None:
            cue_cm = cue_ball['center_cm']
            remaining_dicts = [
                b for b in balls
                if not b.get('is_cue')
                and b.get('center_cm') is not None
                and b.get('color') in gt.remaining_balls
            ]
            if remaining_dicts:
                best_ball, cue_path, target_path = suggest_best_shot(
                    cue_cm, remaining_dicts, all_balls=balls
                )
                if best_ball is not None:
                    gt.select_target(best_ball['color'])
                else:
                    gt.selected_target_color = None

    window.render(
        frame          = frame,
        balls          = balls,
        cue_path       = cue_path,
        target_path    = target_path,
        game_state     = gt.state,
        stroke_count   = gt.stroke_count,
        remaining      = gt.remaining_balls,
        selected_color = gt.selected_target_color,
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
