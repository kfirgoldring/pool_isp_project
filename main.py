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

import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── Project modules ───────────────────────────────────────────────────────────
from core.game_tracker import GameTracker, ST_TRACKING, ST_DISTURBED, TABLE_WIDTH_CM, TABLE_HEIGHT_CM
from core.trajectory import suggest_best_shot

# ── Ball Detection import ─────────────────────────────────────────────────────
# detect_balls_with_color(frame_bgr, table_corners, ref_path, ...) → np.ndarray (N,3)
# Each row: [x_cam_px, y_cam_px, color_str]
try:
    from services.ball_detection_test import detect_balls_with_color as _detect_balls_with_color
    DETECTION_AVAILABLE = True
    print('[main] ball_detection_test (Hough) module loaded.')
except (ImportError, AttributeError) as _e:
    DETECTION_AVAILABLE = False
    print(f'[main] ball_detection_test not available ({_e}).')


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
        _run_tick(window, gt, _ctx)

    window = BilliardsApp(tick_callback=on_tick)
    window.game = gt
    window._last_balls = []
    _ctx['window'] = window

    window.show()

    sys.exit(app_qt.exec_())


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame orchestration tick
# ─────────────────────────────────────────────────────────────────────────────

def _run_tick(window, gt: GameTracker, ctx: Optional[Dict] = None) -> None:
    """Single orchestration tick, called every 33 ms."""

    frame = window.grab_frame()
    if frame is None:
        return

    if window.table_corners is not None:
        gt.set_table_corners(window.table_corners)
    _sync_tracker_reference(window, gt, ctx)

    raw_balls: List[Dict] = []
    if gt.needs_detection:
        raw_balls = _detect_balls(
            frame, window.cam_H, window.table_corners,
            window.ref_path,
        )

    balls = gt.update(frame, raw_balls)
    window._last_balls = balls

    window.consume_pending_clicks()

    # Compute trajectory only while tracking with at least 2 balls visible
    cue_path:    List[Tuple[float, float]] = []
    target_path: List[Tuple[float, float]] = []

    if gt.state in (ST_TRACKING, ST_DISTURBED) and len(balls) >= 2:
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
# Tracker reference sync
# ─────────────────────────────────────────────────────────────────────────────

def _sync_tracker_reference(window, gt: GameTracker, ctx: Optional[Dict]) -> None:
    """Keep GameTracker's reference image in sync with window.ref_path.

    The image is loaded only when path/mtime changes to avoid disk IO per tick.
    """
    if ctx is None:
        return

    ref_path = getattr(window, 'ref_path', None)
    mtime: Optional[float] = None
    if ref_path:
        try:
            mtime = os.path.getmtime(ref_path)
        except OSError:
            mtime = None

    if ref_path == ctx.get('ref_path') and mtime == ctx.get('ref_mtime'):
        return

    ctx['ref_path'] = ref_path
    ctx['ref_mtime'] = mtime

    if not ref_path or mtime is None:
        gt.set_ref_image(None)
        return

    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        print(f'[main] WARNING: failed to load reference image at {ref_path}')
        gt.set_ref_image(None)
        return

    gt.set_ref_image(ref_img)


# ─────────────────────────────────────────────────────────────────────────────
# Ball detection  (adapter between ball_detection.py output and internal format)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_balls(
    frame:         np.ndarray,
    cam_H:         Optional[np.ndarray],
    table_corners: Optional[np.ndarray],
    ref_path:      Optional[str] = None,
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
            'radius':    18,   # ball_detection doesn't expose per-ball radius
        })
    return balls


# Script entry
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    main()
