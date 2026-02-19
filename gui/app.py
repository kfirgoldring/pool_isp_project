"""
Billiards Assistance System - GUI Module
=========================================
PyQt5-based control window + OpenCV projection overlay.

Supports two output modes:
  - Screen Mode:     Camera feed with overlays shown on the operator's monitor.
  - Projection Mode: Black-background overlay drawn in projector coordinates,
                     displayed fullscreen on a second monitor / projector.

Standalone usage (mock data, no camera required):
    python gui/app.py

Integration usage:
    from gui import BilliardsApp, draw_overlay
"""

import sys
import os
import time
import numpy as np
import cv2

# ── Make parent directory importable (config.py, Ball_Detection.py, etc.) ────
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import config

# ── Optional module imports (graceful fallback to mock data) ─────────────────
try:
    from Ball_Detection import detect_balls as _detect_balls_impl
    DETECTION_AVAILABLE = True
except (ImportError, AttributeError):
    DETECTION_AVAILABLE = False

try:
    from Physics_Engine import calculate_path as _calculate_path_impl
    PHYSICS_AVAILABLE = True
except (ImportError, AttributeError):
    PHYSICS_AVAILABLE = False

# ── PyQt5 imports ─────────────────────────────────────────────────────────────
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFrame, QSizePolicy, QStatusBar,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette

from typing import Dict, List, Optional, Tuple

# ═════════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════════

MODE_SCREEN     = 'screen'
MODE_PROJECTION = 'projection'

# Billiards ball color name → OpenCV BGR
BALL_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    'white':  (255, 255, 255),
    'yellow': (0,   230, 255),
    'blue':   (210, 100,   0),
    'red':    (0,     0, 210),
    'purple': (170,   0, 120),
    'orange': (0,   130, 255),
    'green':  (0,   160,  50),
    'maroon': (0,    30, 120),
    'black':  (40,   40,  40),
    'gray':   (128, 128, 128),
}

COLOR_TRAJECTORY = (0, 255, 150)   # bright green-cyan  — cue ball path
COLOR_TARGET_PATH = (0, 165, 255)  # orange             — target ball → pocket
COLOR_SELECTION  = (0, 255, 255)   # cyan selection ring
COLOR_POCKET     = (200, 200, 200) # light grey pocket marker
COLOR_POCKET_SEL = (0, 220, 255)   # highlighted selected pocket

POCKET_CLICK_RADIUS = 24  # px tolerance for clicking a pocket marker

# Standard billiards table: 6 pockets in table-cm coordinates.
# (0,0) = top-left corner; TABLE_WIDTH_CM × TABLE_HEIGHT_CM = 122 × 61 cm.
POCKET_POSITIONS_TABLE = [
    {'name': 'top-left',      'pos': (0.0,                      0.0)},
    {'name': 'top-center',    'pos': (config.TABLE_WIDTH_CM / 2, 0.0)},
    {'name': 'top-right',     'pos': (config.TABLE_WIDTH_CM,     0.0)},
    {'name': 'bottom-left',   'pos': (0.0,                      config.TABLE_HEIGHT_CM)},
    {'name': 'bottom-center', 'pos': (config.TABLE_WIDTH_CM / 2, config.TABLE_HEIGHT_CM)},
    {'name': 'bottom-right',  'pos': (config.TABLE_WIDTH_CM,     config.TABLE_HEIGHT_CM)},
]

# Fallback pocket positions as fractions of frame size (used when no calibration).
_POCKET_FRACTIONS = [
    (0.03, 0.04), (0.50, 0.02), (0.97, 0.04),
    (0.03, 0.96), (0.50, 0.98), (0.97, 0.96),
]

# Dark-theme stylesheet applied to the whole control window
_APP_STYLE = """
QMainWindow, QWidget {
    background-color: #1a1a2e;
    color: #e0e0e0;
    font-family: 'Segoe UI', Arial, sans-serif;
}
QLabel {
    color: #e0e0e0;
}
QPushButton {
    background-color: #16213e;
    color: #e0e0e0;
    border: 1px solid #0f3460;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 13px;
    min-height: 32px;
}
QPushButton:hover {
    background-color: #0f3460;
    border-color: #e94560;
}
QPushButton:pressed {
    background-color: #e94560;
}
QPushButton#btn_mode_active {
    background-color: #e94560;
    border-color: #e94560;
    color: #fff;
}
QFrame#divider {
    color: #0f3460;
}
QStatusBar {
    background-color: #16213e;
    color: #aaa;
}
"""

# ═════════════════════════════════════════════════════════════════════════════
# ClickableLabel — QLabel that emits a signal on left mouse click
# ═════════════════════════════════════════════════════════════════════════════

class ClickableLabel(QLabel):
    """A QLabel that emits (x, y) in label pixel coords when left-clicked."""
    clicked = pyqtSignal(int, int)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(event.x(), event.y())
        super().mousePressEvent(event)


# ═════════════════════════════════════════════════════════════════════════════
# ProjectorWindow — fullscreen overlay on the projector / second monitor
# ═════════════════════════════════════════════════════════════════════════════

class ProjectorWindow(QWidget):
    """
    Overlay window for the projector / second monitor.
    Goes fullscreen on the second monitor when one is available.
    If only one monitor is present, shows as a regular window so the
    control panel remains accessible.
    Pressing Escape or M switches back to Screen mode via close_callback.
    """

    def __init__(self, close_callback=None):
        super().__init__()
        self._close_callback = close_callback
        self.setWindowTitle('Billiards - Projector Output  [Esc = back to Screen]')
        self.setStyleSheet('background-color: black;')

        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet('background-color: black;')

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

    def update_frame(self, frame: np.ndarray):
        """Update the displayed overlay. frame is a BGR numpy array."""
        pixmap = _bgr_to_pixmap(frame)
        self._label.setPixmap(
            pixmap.scaled(self.width(), self.height(),
                          Qt.KeepAspectRatioByExpanding,
                          Qt.SmoothTransformation)
        )

    def show_on_best_screen(self):
        """
        Go fullscreen on the second monitor if one is connected.
        Fall back to a regular (non-fullscreen) window on the primary monitor
        so the control panel stays reachable.
        """
        screens = QApplication.screens()
        if len(screens) >= 2:
            # Place fullscreen on the second screen (the projector)
            geom = screens[1].geometry()
            self.move(geom.left(), geom.top())
            self.resize(geom.width(), geom.height())
            self.showFullScreen()
        else:
            # Single monitor: show as a normal window alongside the control panel
            self.showNormal()
            self.resize(800, 500)
            self.move(100, 100)

    def keyPressEvent(self, event):
        """Escape / M / Q → switch back to screen mode."""
        if event.key() in (Qt.Key_Escape, Qt.Key_M, Qt.Key_Q):
            self._return_to_screen()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """Window closed (e.g. via title-bar X) → switch back to screen mode."""
        self._return_to_screen()
        event.accept()

    def _return_to_screen(self):
        if self._close_callback is not None:
            self._close_callback()


# ═════════════════════════════════════════════════════════════════════════════
# BilliardsApp — main PyQt5 control window
# ═════════════════════════════════════════════════════════════════════════════

class BilliardsApp(QMainWindow):
    """
    Main GUI application for the billiards assistance system.

    Parameters
    ----------
    camera_index : int
        OpenCV camera index. Ignored when use_mock=True.
    use_mock : bool
        If True, bypass camera and use synthetic mock data.
    """

    def __init__(self, camera_index: int = config.CAMERA_INDEX,
                 use_mock: bool = False):
        super().__init__()

        self.camera_index = camera_index
        self.use_mock     = use_mock

        # ── Rendering state ─────────────────────────────────────────────────
        self.mode      = MODE_SCREEN
        self.paused    = False

        # ── Calibration homographies ─────────────────────────────────────────
        self.cam_H       : Optional[np.ndarray] = None   # camera → table
        self.proj_H_inv  : Optional[np.ndarray] = None   # table  → projector

        # ── Ball / path state ────────────────────────────────────────────────
        self.current_balls  : List[Dict] = []
        self.cue_ball       : Optional[Dict] = None
        self.selected_ball  : Optional[Dict] = None
        self.selected_pocket: Optional[Dict] = None   # pocket dict with 'name','pos','cam_pos'
        # Dual paths in CAMERA PIXEL coordinates.
        # cue_path:    cue ball → ghost-ball contact point
        # target_path: target ball → selected pocket
        self.cue_path       : List[Tuple] = []
        self.target_path    : List[Tuple] = []

        # ── Camera frame ─────────────────────────────────────────────────────
        self.current_frame  : Optional[np.ndarray] = None
        self.frozen_frame   : Optional[np.ndarray] = None
        self._cap           = None

        # ── Projector window ─────────────────────────────────────────────────
        self.proj_window    : Optional[ProjectorWindow] = None

        # ── Build UI ─────────────────────────────────────────────────────────
        self._build_ui()
        self._load_calibrations()
        self._start_camera()
        self._start_timer()

    # ─────────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle('Billiards Assistance System')
        self.setMinimumSize(900, 560)
        self.setStyleSheet(_APP_STYLE)

        central = QWidget()
        self.setCentralWidget(central)

        # ── Main horizontal layout: camera | sidebar ─────────────────────────
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # ── Left: camera feed label ───────────────────────────────────────────
        self.camera_label = ClickableLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet(
            'background-color: #0a0a1a; border: 2px solid #0f3460; border-radius: 4px;'
        )
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_label.setText('Initialising camera...')
        self.camera_label.clicked.connect(self._on_camera_click)
        main_layout.addWidget(self.camera_label, stretch=4)

        # ── Right: sidebar ────────────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(200)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(4, 4, 4, 4)
        sidebar_layout.setSpacing(6)

        # Title
        title = QLabel('Billiards\nAssistance')
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(13)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet('color: #e94560; padding: 4px 0;')
        sidebar_layout.addWidget(title)

        sidebar_layout.addWidget(self._make_divider())

        # Status labels
        status_header = QLabel('STATUS')
        status_header.setStyleSheet('color: #888; font-size: 10px; letter-spacing: 1px;')
        sidebar_layout.addWidget(status_header)

        self.lbl_mode   = self._make_status_label('Mode: Screen')
        self.lbl_balls  = self._make_status_label('Balls: —')
        self.lbl_cue    = self._make_status_label('Cue: —')
        self.lbl_target = self._make_status_label('Target: none')
        self.lbl_pocket = self._make_status_label('Pocket: none')
        self.lbl_detect = self._make_status_label(
            'Detection: mock' if not DETECTION_AVAILABLE else 'Detection: real'
        )
        sidebar_layout.addWidget(self.lbl_mode)
        sidebar_layout.addWidget(self.lbl_balls)
        sidebar_layout.addWidget(self.lbl_cue)
        sidebar_layout.addWidget(self.lbl_target)
        sidebar_layout.addWidget(self.lbl_pocket)
        sidebar_layout.addWidget(self.lbl_detect)

        sidebar_layout.addWidget(self._make_divider())

        # Action buttons
        btn_header = QLabel('CONTROLS')
        btn_header.setStyleSheet('color: #888; font-size: 10px; letter-spacing: 1px;')
        sidebar_layout.addWidget(btn_header)

        self.btn_capture = QPushButton('Capture Frame')
        self.btn_reset   = QPushButton('Reset Selection')
        self.btn_mode    = QPushButton('Switch to Projection')

        self.btn_capture.setToolTip('Freeze / unfreeze the camera feed (Space)')
        self.btn_reset.setToolTip('Clear selected target ball (R)')
        self.btn_mode.setToolTip('Toggle Screen / Projection output mode (M)')

        self.btn_capture.clicked.connect(self._on_capture)
        self.btn_reset.clicked.connect(self._on_reset)
        self.btn_mode.clicked.connect(self._on_toggle_mode)

        sidebar_layout.addWidget(self.btn_capture)
        sidebar_layout.addWidget(self.btn_reset)
        sidebar_layout.addWidget(self.btn_mode)

        sidebar_layout.addStretch()

        # Instructions
        hint = QLabel(
            '① Click a ball\n   to select target.\n'
            '② Click a pocket ◆\n   to set destination.\n'
            'Right-click to clear.'
        )
        hint.setStyleSheet('color: #666; font-size: 11px;')
        hint.setAlignment(Qt.AlignCenter)
        hint.setWordWrap(True)
        sidebar_layout.addWidget(hint)

        main_layout.addWidget(sidebar, stretch=0)

        # ── Status bar ────────────────────────────────────────────────────────
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage('Ready — click a ball to select a target.')

    def _make_status_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet('font-size: 12px; padding: 1px 0;')
        return lbl

    def _make_divider(self) -> QFrame:
        line = QFrame()
        line.setObjectName('divider')
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet('color: #0f3460;')
        return line

    # ─────────────────────────────────────────────────────────────────────────
    # Calibration loading
    # ─────────────────────────────────────────────────────────────────────────

    def _load_calibrations(self):
        """Load homography matrices from disk. Missing files → warn only."""
        cam_path  = config.CAMERA_CALIB_FILE
        proj_path = config.PROJECTOR_CALIB_FILE

        if os.path.exists(cam_path):
            self.cam_H = np.load(cam_path).astype(np.float32)
            print(f'[GUI] Camera homography loaded from {cam_path}')
        else:
            print(f'[GUI] WARNING: No camera calibration at {cam_path}')
            print( '[GUI]   Overlays will use raw camera pixel coordinates.')

        if os.path.exists(proj_path):
            raw_H         = np.load(proj_path).astype(np.float32)
            self.proj_H_inv = np.linalg.inv(raw_H).astype(np.float32)
            print(f'[GUI] Projector homography loaded from {proj_path}')
        else:
            print(f'[GUI] WARNING: No projector calibration at {proj_path}')
            print( '[GUI]   Projection mode will show an error overlay.')

    # ─────────────────────────────────────────────────────────────────────────
    # Camera setup
    # ─────────────────────────────────────────────────────────────────────────

    def _start_camera(self):
        if self.use_mock:
            print('[GUI] Mock mode — no camera opened.')
            return
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            print(f'[GUI] WARNING: Cannot open camera {self.camera_index}. Falling back to mock.')
            self.use_mock = True

    def _start_timer(self):
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start(33)  # ~30 fps

    # ─────────────────────────────────────────────────────────────────────────
    # Main timer callback
    # ─────────────────────────────────────────────────────────────────────────

    def _on_timer(self):
        # ── Grab frame ───────────────────────────────────────────────────────
        if not self.paused:
            frame = self._grab_frame()
            if frame is not None:
                self.current_frame = frame
                self.frozen_frame  = frame.copy()
        else:
            frame = self.frozen_frame

        if frame is None:
            return

        # ── Detect balls ─────────────────────────────────────────────────────
        balls = self._detect_or_mock(frame)
        self.current_balls = balls
        self.cue_ball = next((b for b in balls if b.get('is_cue')), None)

        # Keep selected_ball reference alive across frames by re-matching
        if self.selected_ball is not None:
            self.selected_ball = self._rematch_selection(self.selected_ball, balls)

        # ── Compute paths ────────────────────────────────────────────────────
        pockets_cam = self._get_pocket_camera_positions(frame)
        self.cue_path, self.target_path = self._get_paths(frame, pockets_cam)

        # ── Render and display ───────────────────────────────────────────────
        annotated = self._render_frame(
            frame.copy(), balls, pockets_cam,
            self.cue_path, self.target_path
        )
        self.camera_label.setPixmap(
            _bgr_to_pixmap(annotated).scaled(
                self.camera_label.width(), self.camera_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

        # ── Update projector window if open ──────────────────────────────────
        if self.proj_window is not None and self.proj_window.isVisible():
            proj_canvas = self._render_projection(
                balls, pockets_cam, self.cue_path, self.target_path
            )
            self.proj_window.update_frame(proj_canvas)

        # ── Refresh sidebar status ───────────────────────────────────────────
        self._update_status_panel()

    # ─────────────────────────────────────────────────────────────────────────
    # Frame acquisition
    # ─────────────────────────────────────────────────────────────────────────

    def _grab_frame(self) -> Optional[np.ndarray]:
        if self.use_mock:
            return _make_mock_frame()
        if self._cap is None or not self._cap.isOpened():
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    # ─────────────────────────────────────────────────────────────────────────
    # Ball detection
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_or_mock(self, frame: np.ndarray) -> List[Dict]:
        """Run real detection if available; otherwise return mock balls."""
        if not self.use_mock and DETECTION_AVAILABLE:
            try:
                result = _detect_balls_impl(frame)
                if result:
                    return result
            except Exception as exc:
                print(f'[GUI] Detection error: {exc}')

        # Mock: 6 balls scaled to current frame size
        h, w = frame.shape[:2]
        return [
            {'center': (int(w * 0.50), int(h * 0.50)),
             'radius': 18, 'color': 'white',  'is_cue': True},
            {'center': (int(w * 0.35), int(h * 0.38)),
             'radius': 16, 'color': 'red',    'is_cue': False},
            {'center': (int(w * 0.65), int(h * 0.33)),
             'radius': 16, 'color': 'blue',   'is_cue': False},
            {'center': (int(w * 0.28), int(h * 0.62)),
             'radius': 16, 'color': 'yellow', 'is_cue': False},
            {'center': (int(w * 0.72), int(h * 0.60)),
             'radius': 16, 'color': 'green',  'is_cue': False},
            {'center': (int(w * 0.55), int(h * 0.72)),
             'radius': 16, 'color': 'purple', 'is_cue': False},
        ]

    def _rematch_selection(self, prev: Dict, balls: List[Dict]) -> Optional[Dict]:
        """Re-find the previously selected ball in the new detection by proximity."""
        if not balls:
            return None
        px, py = prev['center']
        best, best_d = None, float('inf')
        for b in balls:
            if b.get('is_cue'):
                continue
            cx, cy = b['center']
            d = (cx - px) ** 2 + (cy - py) ** 2
            if d < best_d:
                best_d = d
                best = b
        # Accept if within 50px (2500 squared) — reasonable for 30fps
        return best if best_d < 2500 else None

    # ─────────────────────────────────────────────────────────────────────────
    # Pocket positions
    # ─────────────────────────────────────────────────────────────────────────

    def _get_pocket_camera_positions(self, frame: np.ndarray) -> List[Dict]:
        """
        Return the 6 pocket dicts, each with a 'cam_pos' (x, y) in camera pixels.
        Uses the camera homography if available; falls back to frame-fraction positions.
        """
        h, w = frame.shape[:2]
        pockets = []
        for i, p in enumerate(POCKET_POSITIONS_TABLE):
            cam_pos = self._table_to_camera(p['pos'])
            if cam_pos is None:
                fx, fy = _POCKET_FRACTIONS[i]
                cam_pos = (int(w * fx), int(h * fy))
            pockets.append({**p, 'cam_pos': cam_pos})
        return pockets

    # ─────────────────────────────────────────────────────────────────────────
    # Path calculation
    # ─────────────────────────────────────────────────────────────────────────

    def _get_paths(self, frame: np.ndarray,
                   pockets_cam: List[Dict]) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Return (cue_path, target_path) — both as lists of (x, y) in CAMERA PIXELS.

        cue_path   : cue ball → ghost-ball contact point (where cue must hit target).
        target_path: target ball → selected pocket.

        Falls back to ghost-ball geometry when the physics module is unavailable.
        Tries physics module with pocket kwarg first, then without.
        """
        if self.cue_ball is None or self.selected_ball is None:
            return [], []

        cue_cam    = self.cue_ball['center']
        target_cam = self.selected_ball['center']
        pocket_cam = (
            self.selected_pocket['cam_pos'] if self.selected_pocket else None
        )

        # ── Try real physics module ───────────────────────────────────────────
        if not self.use_mock and PHYSICS_AVAILABLE:
            try:
                cue_t    = self._camera_to_table(cue_cam)
                target_t = self._camera_to_table(target_cam)
                if cue_t and target_t:
                    cue_d    = {**self.cue_ball,    'center': cue_t}
                    target_d = {**self.selected_ball,'center': target_t}
                    pocket_d = None
                    if self.selected_pocket:
                        pocket_d = {**self.selected_pocket,
                                    'center': self.selected_pocket['pos']}

                    # Try 3-arg version (with pocket) first
                    try:
                        result = _calculate_path_impl(cue_d, target_d, pocket_d)
                    except TypeError:
                        result = _calculate_path_impl(cue_d, target_d)

                    # Physics module may return a tuple (cue_path, target_path)
                    # or just a single list for the cue path.
                    if isinstance(result, tuple) and len(result) == 2:
                        cp_t, tp_t = result
                        cp  = [self._table_to_camera(p) for p in cp_t]
                        tp  = [self._table_to_camera(p) for p in tp_t]
                        return (
                            [p for p in cp if p is not None],
                            [p for p in tp if p is not None],
                        )
                    else:
                        # Single list → that's the cue path
                        cp = [self._table_to_camera(p) for p in result]
                        cue_path = [p for p in cp if p is not None]
                        target_path = (
                            [target_cam, pocket_cam] if pocket_cam else []
                        )
                        return cue_path, target_path
            except Exception as exc:
                print(f'[GUI] Physics error: {exc}')

        # ── Geometric fallback ────────────────────────────────────────────────
        # Ghost-ball: the cue ball must arrive at a position that is one
        # ball-diameter behind the target along the target→pocket direction.
        ball_r = self.selected_ball.get('radius', 16)
        cue_path: List[Tuple] = []
        target_path: List[Tuple] = []

        if pocket_cam:
            tx, ty = target_cam
            px, py = pocket_cam
            dx, dy = px - tx, py - ty
            dist = (dx**2 + dy**2) ** 0.5
            if dist > 0:
                ux, uy = dx / dist, dy / dist
                # Ghost ball center = target center shifted away from pocket
                ghost = (int(tx - ux * 2 * ball_r), int(ty - uy * 2 * ball_r))
                cue_path    = [cue_cam, ghost]
                target_path = [target_cam, pocket_cam]
            else:
                cue_path = [cue_cam, target_cam]
        else:
            # No pocket selected: straight line cue → target only
            cx, cy = cue_cam
            tx, ty = target_cam
            cue_path = [
                (int(cx + (tx - cx) * t / 9), int(cy + (ty - cy) * t / 9))
                for t in range(10)
            ]

        return cue_path, target_path

    @staticmethod
    def _extend_path_backward(path: List[Tuple]) -> List[Tuple]:
        """
        Prepend a backward extension point to *path* so the aiming line
        continues past the cue ball, showing the player where to stand and aim.

        The extension length equals the full path length (cue → ghost),
        capped at 150 camera pixels, giving a symmetric aiming line.
        """
        if len(path) < 2:
            return path
        p0x, p0y = float(path[0][0]),  float(path[0][1])
        p1x, p1y = float(path[-1][0]), float(path[-1][1])
        dx, dy   = p1x - p0x, p1y - p0y
        dist     = (dx ** 2 + dy ** 2) ** 0.5
        if dist < 1:
            return path
        ext      = min(dist, 150.0)
        ux, uy   = dx / dist, dy / dist
        backward = (int(p0x - ux * ext), int(p0y - uy * ext))
        return [backward] + list(path)

    # ─────────────────────────────────────────────────────────────────────────
    # Screen-mode rendering  (draws on the camera frame)
    # ─────────────────────────────────────────────────────────────────────────

    def _render_frame(self, canvas: np.ndarray,
                      balls: List[Dict],
                      pockets_cam: List[Dict],
                      cue_path: List[Tuple],
                      target_path: List[Tuple]) -> np.ndarray:
        """Draw overlays on the camera frame for screen-mode display."""
        # 1. Target-ball → pocket trajectory (orange, under everything)
        if len(target_path) >= 2:
            self._draw_trajectory(
                canvas,
                [(int(p[0]), int(p[1])) for p in target_path],
                color=COLOR_TARGET_PATH,
            )

        # 2. Cue-ball → ghost-ball trajectory (cyan), extended backwards to show
        #    where the player should stand and aim from.
        if len(cue_path) >= 2:
            extended = self._extend_path_backward(cue_path)
            self._draw_trajectory(
                canvas,
                [(int(p[0]), int(p[1])) for p in extended],
                color=COLOR_TRAJECTORY,
            )

        # 3. Ghost-ball outline at contact point (last point of cue_path)
        if len(cue_path) >= 2 and self.selected_ball:
            ghost = (int(cue_path[-1][0]), int(cue_path[-1][1]))
            ball_r = self.selected_ball.get('radius', 16)
            cv2.circle(canvas, ghost, ball_r, COLOR_TRAJECTORY, 2, cv2.LINE_AA)

        # 4. Pocket markers (draw before balls so balls appear on top)
        for pocket in pockets_cam:
            is_sel = (self.selected_pocket is not None and
                      pocket['name'] == self.selected_pocket['name'])
            self._draw_pocket(canvas, pocket['cam_pos'], selected=is_sel)

        # 5. All detected balls
        for ball in balls:
            cx, cy = ball['center']
            self._draw_ball(
                canvas,
                center     = (int(cx), int(cy)),
                radius     = ball.get('radius', 15),
                color_name = ball.get('color', 'gray'),
                is_cue     = ball.get('is_cue', False),
                selected   = (ball is self.selected_ball),
            )

        # 6. Pause indicator
        if self.paused:
            cv2.putText(canvas, 'PAUSED', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2,
                        cv2.LINE_AA)

        return canvas

    # ─────────────────────────────────────────────────────────────────────────
    # Projection-mode rendering  (black canvas in projector coordinates)
    # ─────────────────────────────────────────────────────────────────────────

    def _render_projection(self, balls: List[Dict],
                           pockets_cam: List[Dict],
                           cue_path: List[Tuple],
                           target_path: List[Tuple]) -> np.ndarray:
        """Build a 1920×1080 black canvas with coloured overlays for projection."""
        canvas = np.zeros(
            (config.PROJECTOR_HEIGHT, config.PROJECTOR_WIDTH, 3),
            dtype=np.uint8
        )

        if self.proj_H_inv is None:
            cv2.putText(canvas, 'NO PROJECTOR CALIBRATION',
                        (300, 540), cv2.FONT_HERSHEY_SIMPLEX,
                        2.0, (0, 0, 220), 4, cv2.LINE_AA)
            return canvas

        def _cam_path_to_proj(path_cam):
            """Convert a list of camera-pixel points to projector pixels."""
            result = []
            for pt in path_cam:
                t = self._camera_to_table(pt)
                if t:
                    p = self._table_to_projector(t)
                    if p:
                        result.append(p)
            return result

        # 1. Target-ball → pocket trajectory (orange)
        if len(target_path) >= 2:
            tp_proj = _cam_path_to_proj(target_path)
            if len(tp_proj) >= 2:
                self._draw_trajectory(canvas, tp_proj, thickness=3,
                                      color=COLOR_TARGET_PATH)

        # 2. Cue-ball → ghost-ball trajectory (cyan), extended backwards
        if len(cue_path) >= 2:
            extended_cam = self._extend_path_backward(cue_path)
            cp_proj = _cam_path_to_proj(extended_cam)
            if len(cp_proj) >= 2:
                self._draw_trajectory(canvas, cp_proj, thickness=3,
                                      color=COLOR_TRAJECTORY)

        # 3. Ghost-ball outline at contact point
        if len(cue_path) >= 2 and self.selected_ball:
            ghost_t = self._camera_to_table(cue_path[-1])
            if ghost_t:
                ghost_proj = self._table_to_projector(ghost_t)
                if ghost_proj:
                    cv2.circle(canvas, ghost_proj, 22, COLOR_TRAJECTORY, 2, cv2.LINE_AA)

        # 4. Pocket markers in projector coords
        for pocket in pockets_cam:
            proj_pt = self._table_to_projector(pocket['pos'])
            if proj_pt:
                is_sel = (self.selected_pocket is not None and
                          pocket['name'] == self.selected_pocket['name'])
                self._draw_pocket(canvas, proj_pt, selected=is_sel, radius=18)

        # 5. Balls in projector coords
        for ball in balls:
            table_pt = self._camera_to_table(ball['center'])
            if table_pt is None:
                continue
            proj_pt = self._table_to_projector(table_pt)
            if proj_pt is None:
                continue
            self._draw_ball(
                canvas,
                center     = proj_pt,
                radius     = 22,   # fixed projector-space radius
                color_name = ball.get('color', 'gray'),
                is_cue     = ball.get('is_cue', False),
                selected   = (ball is self.selected_ball),
            )

        return canvas

    # ─────────────────────────────────────────────────────────────────────────
    # Drawing primitives
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_ball(self, canvas: np.ndarray,
                   center: Tuple[int, int],
                   radius: int,
                   color_name: str,
                   is_cue: bool,
                   selected: bool = False):
        """Draw one ball with fill, border, optional selection ring, and label."""
        bgr = BALL_COLORS_BGR.get(color_name.lower(), BALL_COLORS_BGR['gray'])

        # Filled circle
        cv2.circle(canvas, center, radius, bgr, -1)

        # Border
        border = (255, 255, 255) if is_cue else (40, 40, 40)
        cv2.circle(canvas, center, radius, border, 2)

        # Small centre dot for cue ball
        if is_cue:
            cv2.circle(canvas, center, max(3, radius // 4), (200, 200, 200), -1)

        # Selection highlight ring
        if selected:
            cv2.circle(canvas, center, radius + 6, COLOR_SELECTION, 3)

        # Label below ball
        label = 'cue' if is_cue else color_name[:3]
        lx = center[0] - 10
        ly = center[1] + radius + 14
        cv2.putText(canvas, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1,
                    cv2.LINE_AA)

    def _draw_pocket(self, canvas: np.ndarray,
                     center: Tuple[int, int],
                     selected: bool = False,
                     radius: int = 10):
        """Draw a diamond-shaped pocket marker."""
        x, y = center
        r = radius
        pts = np.array([[x, y - r], [x + r, y], [x, y + r], [x - r, y]])
        color = COLOR_POCKET_SEL if selected else COLOR_POCKET
        cv2.fillPoly(canvas, [pts], color)
        cv2.polylines(canvas, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)
        if selected:
            # Outer glow ring
            cv2.circle(canvas, center, radius + 5, COLOR_POCKET_SEL, 2, cv2.LINE_AA)

    def _draw_trajectory(self, canvas: np.ndarray,
                         path_px: List[Tuple[int, int]],
                         thickness: int = 2,
                         color: Tuple[int, int, int] = COLOR_TRAJECTORY):
        """Draw a dashed line with an arrowhead at the target end."""
        if len(path_px) < 2:
            return

        dash_len = 14
        gap_len  = 7

        for i in range(len(path_px) - 1):
            p1 = np.array(path_px[i],   dtype=np.float32)
            p2 = np.array(path_px[i+1], dtype=np.float32)
            seg = p2 - p1
            length = float(np.linalg.norm(seg))
            if length < 1.0:
                continue
            unit   = seg / length
            drawn  = 0.0
            dash   = True
            while drawn < length:
                chunk = dash_len if dash else gap_len
                end   = min(drawn + chunk, length)
                if dash:
                    a = (int(p1[0] + unit[0] * drawn),
                         int(p1[1] + unit[1] * drawn))
                    b = (int(p1[0] + unit[0] * end),
                         int(p1[1] + unit[1] * end))
                    cv2.line(canvas, a, b, color, thickness, cv2.LINE_AA)
                drawn += chunk
                dash   = not dash

        # Arrowhead at the target end
        end_pt   = path_px[-1]
        prev_pt  = path_px[-2]
        dx, dy   = end_pt[0] - prev_pt[0], end_pt[1] - prev_pt[1]
        angle    = np.arctan2(dy, dx)
        arr_len  = 16 + thickness
        for delta in (+0.42, -0.42):
            ax = int(end_pt[0] - arr_len * np.cos(angle + delta))
            ay = int(end_pt[1] - arr_len * np.sin(angle + delta))
            cv2.line(canvas, end_pt, (ax, ay),
                     color, thickness + 1, cv2.LINE_AA)

    # ─────────────────────────────────────────────────────────────────────────
    # Coordinate transforms
    # ─────────────────────────────────────────────────────────────────────────

    def _camera_to_table(self, pt) -> Optional[Tuple[float, float]]:
        """Camera pixels → table cm using cam_H."""
        if self.cam_H is None:
            return None
        arr = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
        res = cv2.perspectiveTransform(arr, self.cam_H)
        return (float(res[0][0][0]), float(res[0][0][1]))

    def _table_to_projector(self, pt) -> Optional[Tuple[int, int]]:
        """Table cm → projector pixels using proj_H_inv."""
        if self.proj_H_inv is None:
            return None
        arr = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
        res = cv2.perspectiveTransform(arr, self.proj_H_inv)
        return (int(res[0][0][0]), int(res[0][0][1]))

    def _table_to_camera(self, pt) -> Optional[Tuple[int, int]]:
        """Table cm → camera pixels using inverse of cam_H."""
        if self.cam_H is None:
            return None
        inv_H = np.linalg.inv(self.cam_H).astype(np.float32)
        arr   = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
        res   = cv2.perspectiveTransform(arr, inv_H)
        return (int(res[0][0][0]), int(res[0][0][1]))

    # ─────────────────────────────────────────────────────────────────────────
    # Mouse click → ball selection
    # ─────────────────────────────────────────────────────────────────────────

    def _on_camera_click(self, label_x: int, label_y: int):
        """
        Translate a click on the QLabel into camera pixel coords.
        Priority:
          1. If click is near a pocket marker → select that pocket.
          2. If click is near a ball → select that ball as target.
          3. Otherwise → clear both selections.
        """
        frame = self.current_frame if self.current_frame is not None else self.frozen_frame
        if frame is None:
            return

        fh, fw = frame.shape[:2]
        lw = self.camera_label.width()
        lh = self.camera_label.height()

        # Account for aspect-ratio letterboxing inside the label
        scale = min(lw / fw, lh / fh)
        off_x = (lw - fw * scale) / 2
        off_y = (lh - fh * scale) / 2
        cam_x = (label_x - off_x) / scale
        cam_y = (label_y - off_y) / scale

        # 1. Check pockets first (they are at table corners — small targets)
        pockets_cam = self._get_pocket_camera_positions(frame)
        pocket_hit = self._find_pocket_at(cam_x, cam_y, pockets_cam)
        if pocket_hit is not None:
            if (self.selected_pocket is not None and
                    pocket_hit['name'] == self.selected_pocket['name']):
                # Already selected → toggle off (deselect everything)
                self._on_reset()
                self.statusBar().showMessage('Selection cleared.')
            else:
                self.selected_pocket = pocket_hit
                self.statusBar().showMessage(
                    f'Pocket selected: {pocket_hit["name"]}. '
                    'Now click a target ball to see the shot.'
                )
                self._update_status_panel()
            return

        # 2. Check balls
        ball_hit = self._find_ball_at(cam_x, cam_y)
        if ball_hit is None:
            self.selected_ball   = None
            self.selected_pocket = None
            self.cue_path        = []
            self.target_path     = []
            self.statusBar().showMessage('Click a ball to select a target, or a ◆ pocket.')
        elif ball_hit.get('is_cue'):
            self.statusBar().showMessage('Cannot select the cue ball as target.')
        elif ball_hit is self.selected_ball:
            # Already selected → toggle off (deselect everything)
            self._on_reset()
            self.statusBar().showMessage('Selection cleared.')
            return
        else:
            self.selected_ball = ball_hit
            color = ball_hit.get('color', '?')
            if self.selected_pocket:
                self.statusBar().showMessage(
                    f'Target: {color} ball → {self.selected_pocket["name"]} pocket.'
                )
            else:
                self.statusBar().showMessage(
                    f'Target: {color} ball. Now click a ◆ pocket to aim.'
                )
        self._update_status_panel()

    def _find_ball_at(self, x: float, y: float) -> Optional[Dict]:
        """Find the first ball within (radius + 6px) of camera-space point (x, y)."""
        for ball in self.current_balls:
            cx, cy = ball['center']
            r = ball.get('radius', 15)
            if (x - cx) ** 2 + (y - cy) ** 2 <= (r + 6) ** 2:
                return ball
        return None

    def _find_pocket_at(self, x: float, y: float,
                        pockets_cam: List[Dict]) -> Optional[Dict]:
        """Find the pocket whose marker is within POCKET_CLICK_RADIUS of (x, y)."""
        for pocket in pockets_cam:
            px, py = pocket['cam_pos']
            if (x - px) ** 2 + (y - py) ** 2 <= POCKET_CLICK_RADIUS ** 2:
                return pocket
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Button callbacks
    # ─────────────────────────────────────────────────────────────────────────

    def _on_capture(self):
        """Freeze / unfreeze the camera feed."""
        self.paused = not self.paused
        if self.paused:
            self.btn_capture.setText('Resume Feed')
            self.btn_capture.setObjectName('btn_mode_active')
            self.statusBar().showMessage('Feed paused — click Resume to continue.')
        else:
            self.btn_capture.setText('Capture Frame')
            self.btn_capture.setObjectName('')
            self.statusBar().showMessage('Feed resumed.')
        self.btn_capture.setStyleSheet('')   # force style refresh

    def _on_reset(self):
        """Clear the selected target ball, pocket, and trajectories."""
        self.selected_ball   = None
        self.selected_pocket = None
        self.cue_path        = []
        self.target_path     = []
        self._update_status_panel()
        self.statusBar().showMessage('Selection cleared.')

    def _on_toggle_mode(self):
        """Switch between Screen Mode and Projection Mode."""
        if self.mode == MODE_SCREEN:
            self.mode = MODE_PROJECTION
            self.btn_mode.setText('Switch to Screen')
            self.btn_mode.setObjectName('btn_mode_active')
            # Create projector window once, passing this method as the
            # escape callback so Esc / close always returns to screen mode.
            if self.proj_window is None:
                self.proj_window = ProjectorWindow(
                    close_callback=self._on_toggle_mode
                )
            self.proj_window.show_on_best_screen()
            screens = QApplication.screens()
            if len(screens) < 2:
                self.statusBar().showMessage(
                    'Projection mode — no second monitor detected, '
                    'showing overlay in a separate window. Press Esc to return.'
                )
            else:
                self.statusBar().showMessage(
                    'Projection mode — overlay on projector. Press Esc in that window to return.'
                )
        else:
            self.mode = MODE_SCREEN
            self.btn_mode.setText('Switch to Projection')
            self.btn_mode.setObjectName('')
            if self.proj_window is not None:
                # Disconnect callback temporarily to avoid re-entrant call
                self.proj_window._close_callback = None
                self.proj_window.hide()
                self.proj_window._close_callback = self._on_toggle_mode
            self.statusBar().showMessage('Screen mode — overlay shown on camera feed.')
        self.btn_mode.setStyleSheet('')  # force style refresh
        self._update_status_panel()

    # ─────────────────────────────────────────────────────────────────────────
    # Sidebar status update
    # ─────────────────────────────────────────────────────────────────────────

    def _update_status_panel(self):
        self.lbl_mode.setText(
            f'Mode: {"Screen" if self.mode == MODE_SCREEN else "Projection"}'
        )
        self.lbl_balls.setText(f'Balls: {len(self.current_balls)}')
        self.lbl_cue.setText(
            'Cue: found' if self.cue_ball else 'Cue: not found'
        )
        target_text = (
            self.selected_ball.get('color', '?') if self.selected_ball else 'none'
        )
        pocket_text = (
            self.selected_pocket['name'] if self.selected_pocket else 'none'
        )
        self.lbl_target.setText(f'Target: {target_text}')
        self.lbl_pocket.setText(f'Pocket: {pocket_text}')
        self.lbl_detect.setText(
            'Detection: mock' if (self.use_mock or not DETECTION_AVAILABLE)
            else 'Detection: real'
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Keyboard shortcuts
    # ─────────────────────────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key_Q, Qt.Key_Escape):
            self.close()
        elif key == Qt.Key_Space:
            self._on_capture()
        elif key == Qt.Key_R:
            self._on_reset()
        elif key == Qt.Key_M:
            self._on_toggle_mode()
        else:
            super().keyPressEvent(event)

    # ─────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._timer.stop()
        if self._cap is not None:
            self._cap.release()
        if self.proj_window is not None:
            self.proj_window.close()
        event.accept()


# ═════════════════════════════════════════════════════════════════════════════
# Module-level draw_overlay  (matches Project_Plan.md interface exactly)
# ═════════════════════════════════════════════════════════════════════════════

def draw_overlay(image: np.ndarray,
                 balls: List[Dict],
                 path: List[Tuple]) -> np.ndarray:
    """
    Pure-OpenCV rendering function — no Qt window, no run loop.
    Returns a copy of *image* with balls and trajectory drawn on it.

    Parameters
    ----------
    image : np.ndarray
        BGR camera frame.
    balls : List[Dict]
        Each dict: {'center': (x,y), 'radius': int, 'color': str, 'is_cue': bool}
        Centers in camera pixel coordinates.
    path : List[Tuple]
        List of (x, y) points in camera pixel coordinates.

    Returns
    -------
    np.ndarray
        Annotated frame (same shape as input).
    """
    canvas = image.copy()

    # Trajectory
    if len(path) >= 2:
        _draw_trajectory_static(canvas, [(int(p[0]), int(p[1])) for p in path])

    # Balls
    for ball in balls:
        cx, cy = ball['center']
        bgr = BALL_COLORS_BGR.get(
            ball.get('color', 'gray').lower(), BALL_COLORS_BGR['gray']
        )
        radius   = ball.get('radius', 15)
        is_cue   = ball.get('is_cue', False)
        center   = (int(cx), int(cy))

        cv2.circle(canvas, center, radius, bgr, -1)
        border = (255, 255, 255) if is_cue else (40, 40, 40)
        cv2.circle(canvas, center, radius, border, 2)
        if is_cue:
            cv2.circle(canvas, center, max(3, radius // 4), (200, 200, 200), -1)
        label = 'cue' if is_cue else ball.get('color', 'gray')[:3]
        cv2.putText(canvas, label,
                    (center[0] - 10, center[1] + radius + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1,
                    cv2.LINE_AA)

    return canvas


def _draw_trajectory_static(canvas: np.ndarray,
                             path_px: List[Tuple[int, int]],
                             thickness: int = 2):
    """Standalone dashed-arrow draw (no BilliardsApp instance needed)."""
    dash_len, gap_len = 14, 7
    for i in range(len(path_px) - 1):
        p1 = np.array(path_px[i],   dtype=np.float32)
        p2 = np.array(path_px[i+1], dtype=np.float32)
        seg = p2 - p1
        length = float(np.linalg.norm(seg))
        if length < 1.0:
            continue
        unit  = seg / length
        drawn = 0.0
        dash  = True
        while drawn < length:
            chunk = dash_len if dash else gap_len
            end   = min(drawn + chunk, length)
            if dash:
                a = (int(p1[0] + unit[0] * drawn), int(p1[1] + unit[1] * drawn))
                b = (int(p1[0] + unit[0] * end),   int(p1[1] + unit[1] * end))
                cv2.line(canvas, a, b, COLOR_TRAJECTORY, thickness, cv2.LINE_AA)
            drawn += chunk
            dash   = not dash

    end_pt  = path_px[-1]
    prev_pt = path_px[-2]
    dx, dy  = end_pt[0] - prev_pt[0], end_pt[1] - prev_pt[1]
    angle   = np.arctan2(dy, dx)
    for delta in (+0.42, -0.42):
        ax = int(end_pt[0] - 16 * np.cos(angle + delta))
        ay = int(end_pt[1] - 16 * np.sin(angle + delta))
        cv2.line(canvas, end_pt, (ax, ay), COLOR_TRAJECTORY,
                 thickness + 1, cv2.LINE_AA)


# ═════════════════════════════════════════════════════════════════════════════
# Utilities
# ═════════════════════════════════════════════════════════════════════════════

def _bgr_to_pixmap(frame: np.ndarray) -> QPixmap:
    """Convert a BGR numpy frame to QPixmap for display in a Qt widget."""
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w  = rgb.shape[:2]
    qimg  = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def _make_mock_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a synthetic green-felt background for mock mode."""
    frame = np.full((height, width, 3), (34, 100, 34), dtype=np.uint8)
    noise = np.random.randint(-10, 10, frame.shape, dtype=np.int16)
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main(use_mock: bool = False):
    app = QApplication(sys.argv)
    app.setApplicationName('Billiards Assistance System')
    window = BilliardsApp(use_mock=use_mock)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    # Default: mock mode so it works without a camera
    _mock = '--camera' not in sys.argv
    main(use_mock=_mock)
