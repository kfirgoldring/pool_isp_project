"""
Billiards Assistance System - GUI Module
=========================================
PyQt5 application with three-page flow:
  Page 0 — SetupPage:       hardware selection (camera index, projector yes/no)
  Page 1 — CalibrationPage: live camera preview, corner clicking, auto-detect
  Page 2 — Main view:       camera feed with ball overlays and trajectory lines

Calibration homographies are stored in memory only (no project-folder files).
A user-invisible cache at ~/.billiards_assistant/ enables "Use Last" on restart.

Two output modes (main view):
  Screen Mode     — top-down warped camera feed shown on the operator's monitor.
  Projection Mode — black-background overlay in projector coordinates, displayed
                    fullscreen on a second monitor / projector.

Integration usage:
    from gui import BilliardsApp, draw_overlay
"""

import sys
import os
import pathlib
import shutil
import tempfile

import numpy as np
import cv2

# ── Make parent directory importable (config.py, Ball_Detection.py …) ────────
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import config

# ── Optional module imports (graceful fallback) ───────────────────────────────
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

try:
    from Scene_Understanding import (
        get_table_corners,
        compute_homography_from_corners,
    )
    SCENE_AVAILABLE = True
except (ImportError, AttributeError):
    SCENE_AVAILABLE = False

# ── PyQt5 imports ─────────────────────────────────────────────────────────────
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFrame, QSizePolicy, QStatusBar,
    QStackedWidget, QCheckBox, QSpinBox, QGroupBox,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont

from typing import Dict, List, Optional, Tuple

# ═════════════════════════════════════════════════════════════════════════════
# Cache paths  (user home — completely invisible to the user)
# ═════════════════════════════════════════════════════════════════════════════

_CACHE_DIR  = pathlib.Path.home() / '.billiards_assistant'
_CAM_CACHE  = str(_CACHE_DIR / 'camera_homography.npy')
_PROJ_CACHE = str(_CACHE_DIR / 'projector_homography.npy')

# ═════════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════════

MODE_SCREEN     = 'screen'
MODE_PROJECTION = 'projection'

# Top-down (bird's-eye) view — 8 px/cm → 122×61 cm → 976×488 px canvas
TABLE_DISPLAY_SCALE  = 8
BALL_RADIUS_TOP_DOWN = 23   # ≈ 2.85 cm × 8 px/cm (standard pool ball)

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

COLOR_TRAJECTORY  = (0, 255, 150)   # bright green-cyan  — cue ball path
COLOR_TARGET_PATH = (0, 165, 255)   # orange             — target ball → pocket
COLOR_SELECTION   = (0, 255, 255)   # cyan selection ring
COLOR_POCKET      = (200, 200, 200) # light grey pocket marker
COLOR_POCKET_SEL  = (0, 220, 255)   # highlighted selected pocket

POCKET_CLICK_RADIUS = 30  # px tolerance for clicking a pocket marker

# Standard billiards table: 6 pockets in table-cm coordinates.
POCKET_POSITIONS_TABLE = [
    {'name': 'top-left',      'pos': (0.0,                      0.0)},
    {'name': 'top-center',    'pos': (config.TABLE_WIDTH_CM / 2, 0.0)},
    {'name': 'top-right',     'pos': (config.TABLE_WIDTH_CM,     0.0)},
    {'name': 'bottom-left',   'pos': (0.0,                      config.TABLE_HEIGHT_CM)},
    {'name': 'bottom-center', 'pos': (config.TABLE_WIDTH_CM / 2, config.TABLE_HEIGHT_CM)},
    {'name': 'bottom-right',  'pos': (config.TABLE_WIDTH_CM,     config.TABLE_HEIGHT_CM)},
]

# Fallback pocket fractions of frame size (used when no calibration)
_POCKET_FRACTIONS = [
    (0.03, 0.04), (0.50, 0.02), (0.97, 0.04),
    (0.03, 0.96), (0.50, 0.98), (0.97, 0.96),
]

# ── Shared dark-theme stylesheet ──────────────────────────────────────────────
_APP_STYLE = """
QMainWindow, QWidget {
    background-color: #1a1a2e;
    color: #e0e0e0;
    font-family: 'Segoe UI', Arial, sans-serif;
}
QLabel { color: #e0e0e0; }
QPushButton {
    background-color: #16213e;
    color: #e0e0e0;
    border: 1px solid #0f3460;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 13px;
    min-height: 32px;
}
QPushButton:hover  { background-color: #0f3460; border-color: #e94560; }
QPushButton:pressed{ background-color: #e94560; }
QPushButton:disabled { background-color: #111; color: #555; border-color: #333; }
QPushButton#btn_mode_active { background-color: #e94560; border-color: #e94560; color: #fff; }
QGroupBox {
    border: 1px solid #0f3460;
    border-radius: 6px;
    margin-top: 10px;
    padding: 8px;
    font-size: 12px;
    color: #aaa;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QCheckBox { color: #e0e0e0; font-size: 13px; }
QCheckBox::indicator { width: 16px; height: 16px; }
QSpinBox {
    background-color: #16213e;
    color: #e0e0e0;
    border: 1px solid #0f3460;
    border-radius: 4px;
    padding: 2px 6px;
}
QFrame#divider { color: #0f3460; }
QStatusBar { background-color: #16213e; color: #aaa; }
"""


# ═════════════════════════════════════════════════════════════════════════════
# ClickableLabel — QLabel that emits a signal on left mouse click
# ═════════════════════════════════════════════════════════════════════════════

class ClickableLabel(QLabel):
    """QLabel that emits (x, y) in label pixel coords when left-clicked."""
    clicked       = pyqtSignal(int, int)
    right_clicked = pyqtSignal(int, int)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(event.x(), event.y())
        elif event.button() == Qt.RightButton:
            self.right_clicked.emit(event.x(), event.y())
        super().mousePressEvent(event)


# ═════════════════════════════════════════════════════════════════════════════
# ProjectorWindow — fullscreen overlay on the projector / second monitor
# ═════════════════════════════════════════════════════════════════════════════

class ProjectorWindow(QWidget):
    """Overlay window for the projector. Goes fullscreen on the second monitor."""

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
        pixmap = _bgr_to_pixmap(frame)
        self._label.setPixmap(
            pixmap.scaled(self.width(), self.height(),
                          Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        )

    def show_on_best_screen(self):
        screens = QApplication.screens()
        if len(screens) >= 2:
            geom = screens[1].geometry()
            self.move(geom.left(), geom.top())
            self.resize(geom.width(), geom.height())
            self.showFullScreen()
        else:
            self.showNormal()
            self.resize(800, 500)
            self.move(100, 100)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Escape, Qt.Key_M, Qt.Key_Q):
            self._return_to_screen()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self._return_to_screen()
        event.accept()

    def _return_to_screen(self):
        if self._close_callback is not None:
            self._close_callback()


# ═════════════════════════════════════════════════════════════════════════════
# SetupPage — hardware selection (page 0)
# ═════════════════════════════════════════════════════════════════════════════

class SetupPage(QWidget):
    """
    Hardware configuration screen shown on startup.
    Emits start_requested(camera_index, has_camera, has_projector).
    """
    start_requested = pyqtSignal(int, bool, bool)

    def __init__(self):
        super().__init__()
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(40, 40, 40, 40)
        outer.setSpacing(20)

        # Title
        title = QLabel('Billiards Assistance System')
        title.setAlignment(Qt.AlignCenter)
        f = QFont()
        f.setPointSize(16)
        f.setBold(True)
        title.setFont(f)
        title.setStyleSheet('color: #e94560; padding: 8px 0;')
        outer.addWidget(title)

        sub = QLabel('Configure your hardware below, then click Start.')
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet('color: #888; font-size: 12px;')
        outer.addWidget(sub)

        # ── Camera group ─────────────────────────────────────────────────────
        cam_group = QGroupBox('Camera')
        cam_layout = QVBoxLayout(cam_group)

        self._chk_camera = QCheckBox('Camera connected')
        self._chk_camera.setChecked(True)
        self._chk_camera.toggled.connect(self._on_camera_toggled)
        cam_layout.addWidget(self._chk_camera)

        idx_row = QHBoxLayout()
        idx_lbl = QLabel('Camera index:')
        idx_lbl.setStyleSheet('font-size: 12px;')
        self._spin_camera = QSpinBox()
        self._spin_camera.setRange(0, 9)
        self._spin_camera.setValue(config.CAMERA_INDEX)
        self._spin_camera.setFixedWidth(60)
        idx_row.addWidget(idx_lbl)
        idx_row.addWidget(self._spin_camera)
        idx_row.addStretch()
        cam_layout.addLayout(idx_row)
        outer.addWidget(cam_group)

        # ── Projector group ───────────────────────────────────────────────────
        proj_group = QGroupBox('Projector')
        proj_layout = QVBoxLayout(proj_group)
        self._chk_projector = QCheckBox('Projector connected')
        self._chk_projector.setChecked(False)
        proj_layout.addWidget(self._chk_projector)
        outer.addWidget(proj_group)

        # ── Mock label ────────────────────────────────────────────────────────
        self._lbl_mock = QLabel('(No hardware selected — will run in mock/demo mode)')
        self._lbl_mock.setAlignment(Qt.AlignCenter)
        self._lbl_mock.setStyleSheet('color: #888; font-size: 11px;')
        self._lbl_mock.setVisible(False)
        outer.addWidget(self._lbl_mock)

        outer.addStretch()

        # ── Start button ──────────────────────────────────────────────────────
        self._btn_start = QPushButton('Start Pipeline')
        self._btn_start.setMinimumHeight(44)
        f2 = QFont()
        f2.setPointSize(13)
        f2.setBold(True)
        self._btn_start.setFont(f2)
        self._btn_start.clicked.connect(self._on_start)
        outer.addWidget(self._btn_start)

        self._on_camera_toggled(True)

    def _on_camera_toggled(self, checked: bool):
        self._spin_camera.setEnabled(checked)
        has_any = checked or self._chk_projector.isChecked()
        self._lbl_mock.setVisible(not checked and not self._chk_projector.isChecked())

    def _on_start(self):
        self.start_requested.emit(
            self._spin_camera.value(),
            self._chk_camera.isChecked(),
            self._chk_projector.isChecked(),
        )


# ═════════════════════════════════════════════════════════════════════════════
# CalibrationPage — in-GUI camera calibration (page 1)
# ═════════════════════════════════════════════════════════════════════════════

class CalibrationPage(QWidget):
    """
    Live camera preview with interactive 4-corner calibration.

    The user can:
      • Click the feed to add / drag corner points (up to 4)
      • Click "Auto-detect" to run green-felt detection automatically
      • Click "Accept" (enabled when 4 corners are placed) to compute H
      • Click "Use Last" (enabled when a cache file exists) to reuse saved H
      • Click "Skip" to proceed without camera calibration

    Emits calibration_done(H) where H is np.ndarray or None.
    """
    calibration_done = pyqtSignal(object)  # np.ndarray | None

    def __init__(self):
        super().__init__()
        self._cap        = None
        self._timer      = None
        self._frame      = None          # latest raw camera frame
        self._corners    : List         = []
        self._drag_idx   : int          = -1
        self._camera_idx : int          = 0
        self._build()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QLabel('Camera Calibration — Click to place 4 table corners')
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet('color: #e94560; font-size: 13px; font-weight: bold;')
        layout.addWidget(header)

        self._instructions = QLabel(
            'Click on the 4 table corners in order:\n'
            '  1 Top-Left   2 Top-Right   3 Bottom-Right   4 Bottom-Left\n'
            'You can drag existing points to adjust them.'
        )
        self._instructions.setAlignment(Qt.AlignCenter)
        self._instructions.setStyleSheet('color: #888; font-size: 11px;')
        layout.addWidget(self._instructions)

        # Camera preview
        self._cam_label = ClickableLabel()
        self._cam_label.setAlignment(Qt.AlignCenter)
        self._cam_label.setStyleSheet(
            'background-color: #0a0a1a; border: 2px solid #0f3460; border-radius: 4px;'
        )
        self._cam_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._cam_label.setText('Starting camera…')
        self._cam_label.clicked.connect(self._on_label_click)
        layout.addWidget(self._cam_label, stretch=1)

        # Corner count status
        self._lbl_corners = QLabel('Corners: 0 / 4')
        self._lbl_corners.setAlignment(Qt.AlignCenter)
        self._lbl_corners.setStyleSheet('font-size: 12px; color: #aaa;')
        layout.addWidget(self._lbl_corners)

        # Button row
        btn_row = QHBoxLayout()
        self._btn_auto   = QPushButton('Auto-detect')
        self._btn_accept = QPushButton('Accept')
        self._btn_last   = QPushButton('Use Last')
        self._btn_skip   = QPushButton('Skip')

        self._btn_auto.setToolTip('Try to detect table corners automatically from the current frame')
        self._btn_accept.setToolTip('Compute homography from the 4 placed corners')
        self._btn_last.setToolTip('Use the calibration saved from the previous session')
        self._btn_skip.setToolTip('Continue without camera calibration (overlays will be approximate)')

        self._btn_accept.setEnabled(False)
        self._btn_last.setEnabled(os.path.exists(_CAM_CACHE))
        self._btn_auto.setEnabled(SCENE_AVAILABLE)

        self._btn_auto.clicked.connect(self._on_auto_detect)
        self._btn_accept.clicked.connect(self._on_accept)
        self._btn_last.clicked.connect(self._on_use_last)
        self._btn_skip.clicked.connect(self._on_skip)

        btn_row.addWidget(self._btn_auto)
        btn_row.addWidget(self._btn_accept)
        btn_row.addWidget(self._btn_last)
        btn_row.addWidget(self._btn_skip)
        layout.addLayout(btn_row)

    # ── Camera lifecycle ──────────────────────────────────────────────────────

    def start_camera(self, camera_index: int):
        """Open the camera and start the preview timer."""
        self._camera_idx = camera_index
        self._corners    = []
        self._drag_idx   = -1
        self._frame      = None
        self._update_corner_label()

        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            self._cam_label.setText(f'Cannot open camera {camera_index}')
            return

        # Let the camera settle
        for _ in range(5):
            self._cap.read()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start(33)

    def _stop_camera(self):
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ── Timer — live preview ──────────────────────────────────────────────────

    def _on_timer(self):
        if self._cap is None or not self._cap.isOpened():
            return
        ok, frame = self._cap.read()
        if not ok:
            return
        self._frame = frame
        self._show_frame(frame)

    def _show_frame(self, frame: np.ndarray):
        """Overlay corner markers on the frame and display in the label."""
        display = frame.copy()
        pts = self._corners

        for i, pt in enumerate(pts):
            color = (255, 255, 0) if i == self._drag_idx else (0, 255, 0)
            cv2.circle(display, pt, 8,  color, -1)
            cv2.circle(display, pt, 15, color,  2)
            cv2.putText(display, str(i + 1), (pt[0] + 18, pt[1] + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if len(pts) >= 2:
            arr = np.array(pts, np.int32)
            cv2.polylines(display, [arr], len(pts) == 4, (255, 0, 0), 2)

        cv2.putText(display, f'Corners: {len(pts)}/4', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        if len(pts) == 4:
            cv2.putText(display, 'Click Accept to confirm', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        pixmap = _bgr_to_pixmap(display)
        self._cam_label.setPixmap(
            pixmap.scaled(self._cam_label.width(), self._cam_label.height(),
                          Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    # ── Corner placement ──────────────────────────────────────────────────────

    def _label_to_frame(self, lx: int, ly: int) -> Optional[Tuple[int, int]]:
        """Convert a click on the QLabel to frame pixel coordinates."""
        if self._frame is None:
            return None
        fh, fw = self._frame.shape[:2]
        lw = self._cam_label.width()
        lh = self._cam_label.height()
        scale = min(lw / fw, lh / fh)
        off_x = (lw - fw * scale) / 2
        off_y = (lh - fh * scale) / 2
        fx = int((lx - off_x) / scale)
        fy = int((ly - off_y) / scale)
        if 0 <= fx < fw and 0 <= fy < fh:
            return (fx, fy)
        return None

    def _nearest_corner(self, fx: int, fy: int, threshold: int = 20) -> int:
        for i, (cx, cy) in enumerate(self._corners):
            if (fx - cx) ** 2 + (fy - cy) ** 2 < threshold ** 2:
                return i
        return -1

    def _on_label_click(self, lx: int, ly: int):
        pt = self._label_to_frame(lx, ly)
        if pt is None:
            return
        idx = self._nearest_corner(*pt)
        if idx >= 0:
            self._corners[idx] = pt   # move existing corner
        elif len(self._corners) < 4:
            self._corners.append(pt)
        self._update_corner_label()
        if self._frame is not None:
            self._show_frame(self._frame)

    def _update_corner_label(self):
        n = len(self._corners)
        self._lbl_corners.setText(f'Corners: {n} / 4')
        self._btn_accept.setEnabled(n == 4)

    # ── Button handlers ───────────────────────────────────────────────────────

    def _on_auto_detect(self):
        if self._frame is None or not SCENE_AVAILABLE:
            return
        corners = get_table_corners(self._frame)
        if corners is not None:
            self._corners = [(int(x), int(y)) for x, y in corners]
            self._update_corner_label()
            self._show_frame(self._frame)
        else:
            self._lbl_corners.setText('Auto-detect failed — click corners manually')

    def _on_accept(self):
        if len(self._corners) != 4 or not SCENE_AVAILABLE:
            return
        corners_arr = np.array(self._corners, dtype=np.float32)
        H = compute_homography_from_corners(corners_arr)
        self._finish(H)

    def _on_use_last(self):
        if not os.path.exists(_CAM_CACHE):
            return
        H = np.load(_CAM_CACHE).astype(np.float32)
        self._finish(H)

    def _on_skip(self):
        self._finish(None)

    def _finish(self, H):
        self._stop_camera()
        self.calibration_done.emit(H)


# ═════════════════════════════════════════════════════════════════════════════
# BilliardsApp — main PyQt5 application window
# ═════════════════════════════════════════════════════════════════════════════

class BilliardsApp(QMainWindow):
    """
    Main application window.

    Three-page flow:
      0 → SetupPage       (hardware selection)
      1 → CalibrationPage (camera calibration)
      2 → Main view       (camera feed + overlays)
    """

    def __init__(self):
        super().__init__()

        # ── Hardware / mode ──────────────────────────────────────────────────
        self.camera_index   : int           = config.CAMERA_INDEX
        self.use_mock       : bool          = False
        self._wants_projector: bool         = False

        # ── Rendering state ─────────────────────────────────────────────────
        self.mode    = MODE_SCREEN
        self.paused  = False

        # ── Calibration homographies (in-memory only) ────────────────────────
        self.cam_H      : Optional[np.ndarray] = None   # camera → table
        self.proj_H_inv : Optional[np.ndarray] = None   # table  → projector

        # ── Ball / path state ────────────────────────────────────────────────
        self.current_balls   : List[Dict]        = []
        self.cue_ball        : Optional[Dict]    = None
        self.selected_ball   : Optional[Dict]    = None
        self.selected_pocket : Optional[Dict]    = None
        self.cue_path        : List[Tuple]       = []
        self.target_path     : List[Tuple]       = []

        # ── Camera frame ─────────────────────────────────────────────────────
        self.current_frame  : Optional[np.ndarray] = None
        self.frozen_frame   : Optional[np.ndarray] = None
        self._cap           = None
        self._timer         = None

        # ── Projector window ─────────────────────────────────────────────────
        self.proj_window : Optional[ProjectorWindow] = None

        # ── Build UI ─────────────────────────────────────────────────────────
        self._build_ui()

    # ─────────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle('Billiards Assistance System')
        self.setMinimumSize(500, 380)
        self.setStyleSheet(_APP_STYLE)

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        # Page 0: hardware setup
        self._setup_page = SetupPage()
        self._setup_page.start_requested.connect(self._on_setup_done)
        self._stack.addWidget(self._setup_page)

        # Page 1: camera calibration
        self._calib_page = CalibrationPage()
        self._calib_page.calibration_done.connect(self._on_calibration_done)
        self._stack.addWidget(self._calib_page)

        # Page 2: main view (camera feed + sidebar)
        self._stack.addWidget(self._build_main_ui())

        self._stack.setCurrentIndex(0)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage('Configure hardware and click Start.')

    def _build_main_ui(self) -> QWidget:
        """Build the camera feed + sidebar layout and return it as a QWidget."""
        widget = QWidget()
        main_layout = QHBoxLayout(widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # ── Left: camera feed ─────────────────────────────────────────────────
        self.camera_label = ClickableLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet(
            'background-color: #0a0a1a; border: 2px solid #0f3460; border-radius: 4px;'
        )
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_label.setText('Initialising camera…')
        self.camera_label.clicked.connect(self._on_camera_click)
        self.camera_label.right_clicked.connect(lambda lx, ly: self._on_reset())
        main_layout.addWidget(self.camera_label, stretch=4)

        # ── Right: sidebar ────────────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(210)
        sb = QVBoxLayout(sidebar)
        sb.setContentsMargins(4, 4, 4, 4)
        sb.setSpacing(6)

        title = QLabel('Billiards\nAssistance')
        title.setAlignment(Qt.AlignCenter)
        tf = QFont()
        tf.setPointSize(13)
        tf.setBold(True)
        title.setFont(tf)
        title.setStyleSheet('color: #e94560; padding: 4px 0;')
        sb.addWidget(title)
        sb.addWidget(self._make_divider())

        # Status labels
        hdr = QLabel('STATUS')
        hdr.setStyleSheet('color: #888; font-size: 10px; letter-spacing: 1px;')
        sb.addWidget(hdr)

        self.lbl_mode   = self._make_status_label('Mode: Screen')
        self.lbl_calib  = self._make_status_label('Calibration: none')
        self.lbl_balls  = self._make_status_label('Balls: —')
        self.lbl_cue    = self._make_status_label('Cue: —')
        self.lbl_target = self._make_status_label('Target: none')
        self.lbl_pocket = self._make_status_label('Pocket: none')
        self.lbl_detect = self._make_status_label(
            'Detection: mock' if not DETECTION_AVAILABLE else 'Detection: real'
        )
        for lbl in (self.lbl_mode, self.lbl_calib, self.lbl_balls,
                    self.lbl_cue, self.lbl_target, self.lbl_pocket, self.lbl_detect):
            sb.addWidget(lbl)

        sb.addWidget(self._make_divider())

        # Controls
        ctrl_hdr = QLabel('CONTROLS')
        ctrl_hdr.setStyleSheet('color: #888; font-size: 10px; letter-spacing: 1px;')
        sb.addWidget(ctrl_hdr)

        self.btn_capture = QPushButton('Capture Frame')
        self.btn_reset   = QPushButton('Reset Selection')
        self.btn_mode    = QPushButton('Switch to Projection')
        self.btn_recalib = QPushButton('Re-calibrate')

        self.btn_capture.setToolTip('Freeze / unfreeze camera feed (Space)')
        self.btn_reset.setToolTip('Clear selection (R)')
        self.btn_mode.setToolTip('Toggle Screen / Projection (M)')
        self.btn_recalib.setToolTip('Go back to calibration page')

        self.btn_capture.clicked.connect(self._on_capture)
        self.btn_reset.clicked.connect(self._on_reset)
        self.btn_mode.clicked.connect(self._on_toggle_mode)
        self.btn_recalib.clicked.connect(self._on_recalibrate)

        for btn in (self.btn_capture, self.btn_reset, self.btn_mode, self.btn_recalib):
            sb.addWidget(btn)

        sb.addStretch()

        hint = QLabel(
            '① Click a ball to select target.\n'
            '② Click a ◆ pocket to aim.\n'
            'Right-click to clear.'
        )
        hint.setStyleSheet('color: #666; font-size: 11px;')
        hint.setAlignment(Qt.AlignCenter)
        hint.setWordWrap(True)
        sb.addWidget(hint)

        main_layout.addWidget(sidebar, stretch=0)
        return widget

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
    # Pipeline flow — setup → calibration → main
    # ─────────────────────────────────────────────────────────────────────────

    def _on_setup_done(self, camera_index: int, has_camera: bool, has_projector: bool):
        """Called when user clicks Start on the setup page."""
        self.camera_index    = camera_index
        self.use_mock        = not has_camera
        self._wants_projector = has_projector

        if has_camera and SCENE_AVAILABLE:
            # Show calibration page and open its camera preview
            self._stack.setCurrentIndex(1)
            self.setMinimumSize(800, 520)
            self.statusBar().showMessage(
                'Place 4 table corners or click Auto-detect, then Accept.'
            )
            self._calib_page.start_camera(camera_index)
        else:
            # No camera or no Scene_Understanding — skip calibration
            self._on_calibration_done(None)

    def _on_calibration_done(self, H):
        """Called by CalibrationPage when the user accepts / skips calibration."""
        if H is not None:
            self.cam_H = H.astype(np.float32)
            # Cache for future "Use Last"
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.save(_CAM_CACHE, self.cam_H)
            print(f'[GUI] Camera homography saved to cache: {_CAM_CACHE}')
        else:
            self.cam_H = None
            print('[GUI] No camera calibration — using raw camera coordinates.')

        # Projector calibration (blocks while running)
        if self._wants_projector and self.cam_H is not None:
            self.statusBar().showMessage('Running projector calibration…')
            QApplication.processEvents()
            self._calibrate_projector_in_memory()

        # Switch to main view
        self._stack.setCurrentIndex(2)
        self.setMinimumSize(900, 560)
        self._start_camera()
        self._start_timer()

        calib_status = 'OK' if self.cam_H is not None else 'none'
        self.lbl_calib.setText(f'Calibration: {calib_status}')
        self.statusBar().showMessage('Ready — click a ball to select a target.')

    def _on_recalibrate(self):
        """Return to calibration page from the main view."""
        if self._timer is not None:
            self._timer.stop()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.cam_H     = None
        self.proj_H_inv = None
        self._stack.setCurrentIndex(1)
        self.setMinimumSize(800, 520)
        self.statusBar().showMessage('Place 4 corners and click Accept.')
        self._calib_page.start_camera(self.camera_index)

    # ─────────────────────────────────────────────────────────────────────────
    # Projector calibration (in-memory, temp file hidden from user)
    # ─────────────────────────────────────────────────────────────────────────

    def _calibrate_projector_in_memory(self):
        """
        Run ProjectorCalibrator without the user managing any files.
        Passes self.cam_H via a temp file (automatically cleaned up).
        """
        try:
            import billiards_calibration_merged as bcm
            tmp_dir  = tempfile.mkdtemp()
            tmp_path = os.path.join(tmp_dir, 'camera_homography.npy')
            np.save(tmp_path, self.cam_H)
            original = bcm.CAMERA_CALIB_FILE
            bcm.CAMERA_CALIB_FILE = tmp_path
            try:
                proj_H = bcm.ProjectorCalibrator().calibrate(self.camera_index)
            finally:
                bcm.CAMERA_CALIB_FILE = original
                shutil.rmtree(tmp_dir, ignore_errors=True)

            if proj_H is not None:
                self.proj_H_inv = np.linalg.inv(proj_H).astype(np.float32)
                _CACHE_DIR.mkdir(parents=True, exist_ok=True)
                np.save(_PROJ_CACHE, proj_H)
                print(f'[GUI] Projector homography cached: {_PROJ_CACHE}')
        except Exception as exc:
            print(f'[GUI] Projector calibration error: {exc}')

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
        # ── Grab raw camera frame ─────────────────────────────────────────────
        if not self.paused:
            frame = self._grab_frame()
            if frame is not None:
                self.current_frame = frame
                self.frozen_frame  = frame.copy()
        else:
            frame = self.frozen_frame

        if frame is None:
            return

        # ── Detection on warped frame (better accuracy when calibrated) ───────
        detect_frame = (self._warp_to_top_down(frame)
                        if self.cam_H is not None else frame)

        balls = self._detect_or_mock(detect_frame)
        self.current_balls = balls
        self.cue_ball = next((b for b in balls if b.get('is_cue')), None)

        if self.selected_ball is not None:
            self.selected_ball = self._rematch_selection(self.selected_ball, balls)

        # ── Pocket positions in the current display coordinate space ──────────
        pockets = self._get_pocket_positions(frame)

        # ── Compute paths ─────────────────────────────────────────────────────
        self.cue_path, self.target_path = self._get_paths(pockets)

        # ── Render and display ────────────────────────────────────────────────
        annotated = self._render_frame(
            frame.copy(), balls, pockets, self.cue_path, self.target_path
        )
        self.camera_label.setPixmap(
            _bgr_to_pixmap(annotated).scaled(
                self.camera_label.width(), self.camera_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

        # ── Update projector window ───────────────────────────────────────────
        if self.proj_window is not None and self.proj_window.isVisible():
            proj_canvas = self._render_projection(
                balls, pockets, self.cue_path, self.target_path
            )
            self.proj_window.update_frame(proj_canvas)

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
        """Run real detection if available; fall back to mock balls."""
        if not self.use_mock and DETECTION_AVAILABLE:
            try:
                result = _detect_balls_impl(frame)
                if result:
                    return result
            except Exception as exc:
                print(f'[GUI] Detection error: {exc}')

        # Mock: positions scaled to current frame size
        h, w = frame.shape[:2]
        return [
            {'center': (int(w * 0.50), int(h * 0.50)), 'radius': 18, 'color': 'white',  'is_cue': True},
            {'center': (int(w * 0.35), int(h * 0.38)), 'radius': 16, 'color': 'red',    'is_cue': False},
            {'center': (int(w * 0.65), int(h * 0.33)), 'radius': 16, 'color': 'blue',   'is_cue': False},
            {'center': (int(w * 0.28), int(h * 0.62)), 'radius': 16, 'color': 'yellow', 'is_cue': False},
            {'center': (int(w * 0.72), int(h * 0.60)), 'radius': 16, 'color': 'green',  'is_cue': False},
            {'center': (int(w * 0.55), int(h * 0.72)), 'radius': 16, 'color': 'purple', 'is_cue': False},
        ]

    def _rematch_selection(self, prev: Dict, balls: List[Dict]) -> Optional[Dict]:
        """Re-find the previously selected ball by proximity."""
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
        return best if best_d < 2500 else None

    # ─────────────────────────────────────────────────────────────────────────
    # Pocket positions — unified display-space coordinates
    # ─────────────────────────────────────────────────────────────────────────

    def _get_pocket_positions(self, raw_frame: np.ndarray) -> List[Dict]:
        """
        Return 6 pocket dicts, each with 'cam_pos' in the current display space.

        • Top-down mode (cam_H set): positions are in warped display pixels
          (table_cm × TABLE_DISPLAY_SCALE).  No camera transform needed.
        • Raw mode: positions are in camera pixels via inverse cam_H or fractions.
        """
        if self.cam_H is not None:
            return [
                {**p, 'cam_pos': (int(p['pos'][0] * TABLE_DISPLAY_SCALE),
                                  int(p['pos'][1] * TABLE_DISPLAY_SCALE))}
                for p in POCKET_POSITIONS_TABLE
            ]
        # Raw-frame fallback
        h, w = raw_frame.shape[:2]
        result = []
        for i, p in enumerate(POCKET_POSITIONS_TABLE):
            cam_pos = self._table_to_camera(p['pos'])
            if cam_pos is None:
                fx, fy = _POCKET_FRACTIONS[i]
                cam_pos = (int(w * fx), int(h * fy))
            result.append({**p, 'cam_pos': cam_pos})
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Path calculation
    # ─────────────────────────────────────────────────────────────────────────

    def _get_paths(self, pockets: List[Dict]) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Return (cue_path, target_path) both in the current display coordinate space.

        Top-down mode: coordinates are display pixels (table_cm × scale).
        Raw mode:      coordinates are camera pixels.
        """
        if self.cue_ball is None or self.selected_ball is None:
            return [], []

        top_down   = self.cam_H is not None
        cue_disp   = self.cue_ball['center']
        target_disp= self.selected_ball['center']
        pocket_disp= self.selected_pocket['cam_pos'] if self.selected_pocket else None

        # ── Convert to table cm ───────────────────────────────────────────────
        if top_down:
            S = TABLE_DISPLAY_SCALE
            cue_t    = (cue_disp[0]    / S, cue_disp[1]    / S)
            target_t = (target_disp[0] / S, target_disp[1] / S)
        else:
            cue_t    = self._camera_to_table(cue_disp)
            target_t = self._camera_to_table(target_disp)

        # ── Try real physics module (table cm in, table cm out) ────────────────
        if PHYSICS_AVAILABLE:
            try:
                if cue_t and target_t:
                    cue_d    = {**self.cue_ball,     'center': cue_t}
                    target_d = {**self.selected_ball, 'center': target_t}
                    pocket_d = None
                    if self.selected_pocket:
                        pocket_d = {**self.selected_pocket,
                                    'center': self.selected_pocket['pos']}
                    try:
                        result = _calculate_path_impl(cue_d, target_d, pocket_d)
                    except TypeError:
                        result = _calculate_path_impl(cue_d, target_d)

                    def _to_disp(pt_cm):
                        if top_down:
                            return (int(pt_cm[0] * TABLE_DISPLAY_SCALE),
                                    int(pt_cm[1] * TABLE_DISPLAY_SCALE))
                        p = self._table_to_camera(pt_cm)
                        return p

                    if isinstance(result, tuple) and len(result) == 2:
                        cp_t, tp_t = result
                        cp = [_to_disp(p) for p in cp_t]
                        tp = [_to_disp(p) for p in tp_t]
                        return [p for p in cp if p], [p for p in tp if p]
                    else:
                        cp = [_to_disp(p) for p in result]
                        return ([p for p in cp if p],
                                [target_disp, pocket_disp] if pocket_disp else [])
            except Exception as exc:
                print(f'[GUI] Physics error: {exc}')

        # ── Geometric fallback (ghost-ball) ────────────────────────────────────
        ball_r = BALL_RADIUS_TOP_DOWN if top_down else self.selected_ball.get('radius', 16)
        cue_path: List[Tuple]    = []
        target_path: List[Tuple] = []

        if pocket_disp:
            tx, ty = target_disp
            px, py = pocket_disp
            dx, dy = px - tx, py - ty
            dist = (dx ** 2 + dy ** 2) ** 0.5
            if dist > 0:
                ux, uy = dx / dist, dy / dist
                ghost = (int(tx - ux * 2 * ball_r), int(ty - uy * 2 * ball_r))
                cue_path    = [cue_disp, ghost]
                target_path = [target_disp, pocket_disp]
            else:
                cue_path = [cue_disp, target_disp]
        else:
            cx, cy = cue_disp
            tx, ty = target_disp
            cue_path = [
                (int(cx + (tx - cx) * t / 9), int(cy + (ty - cy) * t / 9))
                for t in range(10)
            ]

        return cue_path, target_path

    @staticmethod
    def _extend_path_backward(path: List[Tuple]) -> List[Tuple]:
        """Prepend a backward extension so the aiming line continues past the cue."""
        if len(path) < 2:
            return path
        p0x, p0y = float(path[0][0]),  float(path[0][1])
        p1x, p1y = float(path[-1][0]), float(path[-1][1])
        dx, dy   = p1x - p0x, p1y - p0y
        dist     = (dx ** 2 + dy ** 2) ** 0.5
        if dist < 1:
            return path
        ext    = min(dist, 150.0)
        ux, uy = dx / dist, dy / dist
        return [(int(p0x - ux * ext), int(p0y - uy * ext))] + list(path)

    # ─────────────────────────────────────────────────────────────────────────
    # Top-down warp helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _warp_to_top_down(self, frame: np.ndarray) -> np.ndarray:
        """
        Warp the raw camera frame to a rectified bird's-eye view.

        cam_H maps camera pixels → table cm.  We compose it with a scale
        matrix so the output is in display pixels (table_cm × TABLE_DISPLAY_SCALE).
        """
        w = int(config.TABLE_WIDTH_CM  * TABLE_DISPLAY_SCALE)
        h = int(config.TABLE_HEIGHT_CM * TABLE_DISPLAY_SCALE)
        S = np.array([[TABLE_DISPLAY_SCALE, 0, 0],
                      [0, TABLE_DISPLAY_SCALE, 0],
                      [0, 0,                  1]], dtype=np.float32)
        return cv2.warpPerspective(frame, S @ self.cam_H, (w, h))

    # ─────────────────────────────────────────────────────────────────────────
    # Screen-mode rendering
    # ─────────────────────────────────────────────────────────────────────────

    def _render_frame(self, canvas: np.ndarray,
                      balls: List[Dict],
                      pockets: List[Dict],
                      cue_path: List[Tuple],
                      target_path: List[Tuple]) -> np.ndarray:
        """
        Draw overlays on the frame for screen-mode display.

        When cam_H is set, the canvas is warped to a top-down view first and
        all coordinates are already in display pixel space (balls detected on
        the warped frame, pockets computed in warped space).
        Without cam_H the raw camera frame is used as-is.
        """
        top_down = self.cam_H is not None

        if top_down:
            canvas = self._warp_to_top_down(canvas)
            # Coordinates from detection and pocket computation are already in
            # display pixel space — no extra transform needed.
            ball_r_draw = BALL_RADIUS_TOP_DOWN
        else:
            ball_r_draw = 15

        # 1. Target-ball → pocket trajectory (orange)
        if len(target_path) >= 2:
            self._draw_trajectory(
                canvas, [(int(p[0]), int(p[1])) for p in target_path],
                color=COLOR_TARGET_PATH)

        # 2. Cue-ball → ghost-ball trajectory (cyan), extended backwards
        if len(cue_path) >= 2:
            extended = self._extend_path_backward(cue_path)
            self._draw_trajectory(
                canvas, [(int(p[0]), int(p[1])) for p in extended],
                color=COLOR_TRAJECTORY)

        # 3. Ghost-ball outline at contact point
        if len(cue_path) >= 2 and self.selected_ball:
            ghost = (int(cue_path[-1][0]), int(cue_path[-1][1]))
            cv2.circle(canvas, ghost, ball_r_draw, COLOR_TRAJECTORY, 2, cv2.LINE_AA)

        # 4. Pocket markers
        for pocket in pockets:
            is_sel = (self.selected_pocket is not None and
                      pocket['name'] == self.selected_pocket['name'])
            self._draw_pocket(canvas, pocket['cam_pos'], selected=is_sel)

        # 5. Detected balls
        for ball in balls:
            cx, cy = ball['center']
            self._draw_ball(
                canvas,
                center     = (int(cx), int(cy)),
                radius     = ball_r_draw,
                color_name = ball.get('color', 'gray'),
                is_cue     = ball.get('is_cue', False),
                selected   = (ball is self.selected_ball),
            )

        # 6. Pause indicator
        if self.paused:
            cv2.putText(canvas, 'PAUSED', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2, cv2.LINE_AA)

        return canvas

    # ─────────────────────────────────────────────────────────────────────────
    # Projection-mode rendering  (black canvas in projector coordinates)
    # ─────────────────────────────────────────────────────────────────────────

    def _render_projection(self, balls: List[Dict],
                           pockets: List[Dict],
                           cue_path: List[Tuple],
                           target_path: List[Tuple]) -> np.ndarray:
        """Build a 1920×1080 black canvas with overlays for projection."""
        canvas = np.zeros(
            (config.PROJECTOR_HEIGHT, config.PROJECTOR_WIDTH, 3), dtype=np.uint8
        )
        if self.proj_H_inv is None:
            cv2.putText(canvas, 'NO PROJECTOR CALIBRATION',
                        (300, 540), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 220), 4)
            return canvas

        top_down = self.cam_H is not None

        def _disp_to_proj(pt):
            """Convert display-space point to projector pixels."""
            if top_down:
                # display pixels → table cm → projector pixels
                t = (pt[0] / TABLE_DISPLAY_SCALE, pt[1] / TABLE_DISPLAY_SCALE)
            else:
                t = self._camera_to_table(pt)
            if t is None:
                return None
            return self._table_to_projector(t)

        if len(target_path) >= 2:
            tp_proj = [_disp_to_proj(p) for p in target_path]
            tp_proj = [p for p in tp_proj if p]
            if len(tp_proj) >= 2:
                self._draw_trajectory(canvas, tp_proj, thickness=3,
                                      color=COLOR_TARGET_PATH)

        if len(cue_path) >= 2:
            ext = self._extend_path_backward(cue_path)
            cp_proj = [_disp_to_proj(p) for p in ext]
            cp_proj = [p for p in cp_proj if p]
            if len(cp_proj) >= 2:
                self._draw_trajectory(canvas, cp_proj, thickness=3,
                                      color=COLOR_TRAJECTORY)

        if len(cue_path) >= 2 and self.selected_ball:
            ghost_proj = _disp_to_proj(cue_path[-1])
            if ghost_proj:
                cv2.circle(canvas, ghost_proj, 22, COLOR_TRAJECTORY, 2, cv2.LINE_AA)

        for pocket in pockets:
            proj_pt = self._table_to_projector(pocket['pos'])
            if proj_pt:
                is_sel = (self.selected_pocket is not None and
                          pocket['name'] == self.selected_pocket['name'])
                self._draw_pocket(canvas, proj_pt, selected=is_sel, radius=18)

        for ball in balls:
            if top_down:
                t = (ball['center'][0] / TABLE_DISPLAY_SCALE,
                     ball['center'][1] / TABLE_DISPLAY_SCALE)
            else:
                t = self._camera_to_table(ball['center'])
            if t is None:
                continue
            proj_pt = self._table_to_projector(t)
            if proj_pt is None:
                continue
            self._draw_ball(canvas, proj_pt, 22,
                            ball.get('color', 'gray'),
                            ball.get('is_cue', False),
                            ball is self.selected_ball)

        return canvas

    # ─────────────────────────────────────────────────────────────────────────
    # Drawing primitives
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_ball(self, canvas, center, radius, color_name, is_cue, selected=False):
        bgr    = BALL_COLORS_BGR.get(color_name.lower(), BALL_COLORS_BGR['gray'])
        border = (255, 255, 255) if is_cue else (40, 40, 40)
        cv2.circle(canvas, center, radius, bgr,    -1)
        cv2.circle(canvas, center, radius, border,  2)
        if is_cue:
            cv2.circle(canvas, center, max(3, radius // 4), (200, 200, 200), -1)
        if selected:
            cv2.circle(canvas, center, radius + 6, COLOR_SELECTION, 3)
        label = 'cue' if is_cue else color_name[:3]
        cv2.putText(canvas, label, (center[0] - 10, center[1] + radius + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

    def _draw_pocket(self, canvas, center, selected=False, radius=10):
        x, y = center
        r    = radius
        pts  = np.array([[x, y - r], [x + r, y], [x, y + r], [x - r, y]])
        color = COLOR_POCKET_SEL if selected else COLOR_POCKET
        cv2.fillPoly(canvas, [pts], color)
        cv2.polylines(canvas, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)
        if selected:
            cv2.circle(canvas, center, radius + 5, COLOR_POCKET_SEL, 2, cv2.LINE_AA)

    def _draw_trajectory(self, canvas, path_px, thickness=2, color=COLOR_TRAJECTORY):
        if len(path_px) < 2:
            return
        dash_len, gap_len = 14, 7
        for i in range(len(path_px) - 1):
            p1 = np.array(path_px[i],   dtype=np.float32)
            p2 = np.array(path_px[i+1], dtype=np.float32)
            seg    = p2 - p1
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
                    cv2.line(canvas, a, b, color, thickness, cv2.LINE_AA)
                drawn += chunk
                dash   = not dash
        end_pt  = path_px[-1]
        prev_pt = path_px[-2]
        dx, dy  = end_pt[0] - prev_pt[0], end_pt[1] - prev_pt[1]
        angle   = np.arctan2(dy, dx)
        arr_len = 16 + thickness
        for delta in (+0.42, -0.42):
            ax = int(end_pt[0] - arr_len * np.cos(angle + delta))
            ay = int(end_pt[1] - arr_len * np.sin(angle + delta))
            cv2.line(canvas, end_pt, (ax, ay), color, thickness + 1, cv2.LINE_AA)

    # ─────────────────────────────────────────────────────────────────────────
    # Coordinate transforms (camera ↔ table ↔ projector)
    # ─────────────────────────────────────────────────────────────────────────

    def _camera_to_table(self, pt) -> Optional[Tuple[float, float]]:
        if self.cam_H is None:
            return None
        arr = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
        res = cv2.perspectiveTransform(arr, self.cam_H)
        return (float(res[0][0][0]), float(res[0][0][1]))

    def _table_to_projector(self, pt) -> Optional[Tuple[int, int]]:
        if self.proj_H_inv is None:
            return None
        arr = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
        res = cv2.perspectiveTransform(arr, self.proj_H_inv)
        return (int(res[0][0][0]), int(res[0][0][1]))

    def _table_to_camera(self, pt) -> Optional[Tuple[int, int]]:
        if self.cam_H is None:
            return None
        inv_H = np.linalg.inv(self.cam_H).astype(np.float32)
        arr   = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
        res   = cv2.perspectiveTransform(arr, inv_H)
        return (int(res[0][0][0]), int(res[0][0][1]))

    # ─────────────────────────────────────────────────────────────────────────
    # Mouse click → ball / pocket selection
    # ─────────────────────────────────────────────────────────────────────────

    def _on_camera_click(self, label_x: int, label_y: int):
        """
        Translate a label click to display-space coords, then find the nearest
        ball or pocket.  In top-down mode, display coords are already the
        working coordinate space (display pixels).  In raw mode, they map to
        camera pixels.
        """
        frame = self.current_frame if self.current_frame is not None else self.frozen_frame
        if frame is None:
            return

        lw = self.camera_label.width()
        lh = self.camera_label.height()
        top_down = self.cam_H is not None

        # Determine rendered frame size for letterbox computation
        if top_down:
            fw = int(config.TABLE_WIDTH_CM  * TABLE_DISPLAY_SCALE)
            fh = int(config.TABLE_HEIGHT_CM * TABLE_DISPLAY_SCALE)
        else:
            fh, fw = frame.shape[:2]

        scale = min(lw / fw, lh / fh)
        off_x = (lw - fw * scale) / 2
        off_y = (lh - fh * scale) / 2
        find_x = (label_x - off_x) / scale
        find_y = (label_y - off_y) / scale
        # In both modes, find_x/find_y are now in the display coordinate space
        # which matches ball['center'] and pocket['cam_pos'].

        pockets = self._get_pocket_positions(frame)

        # 1. Check pockets
        pocket_hit = self._find_pocket_at(find_x, find_y, pockets)
        if pocket_hit is not None:
            if (self.selected_pocket is not None and
                    pocket_hit['name'] == self.selected_pocket['name']):
                self._on_reset()
                self.statusBar().showMessage('Selection cleared.')
            else:
                self.selected_pocket = pocket_hit
                self.statusBar().showMessage(
                    f'Pocket selected: {pocket_hit["name"]}. Now click a target ball.')
                self._update_status_panel()
            return

        # 2. Check balls
        ball_hit = self._find_ball_at(find_x, find_y)
        if ball_hit is None:
            self.selected_ball   = None
            self.selected_pocket = None
            self.cue_path        = []
            self.target_path     = []
            self.statusBar().showMessage('Click a ball to select a target, or a ◆ pocket.')
        elif ball_hit.get('is_cue'):
            self.statusBar().showMessage('Cannot select the cue ball as target.')
        elif ball_hit is self.selected_ball:
            self._on_reset()
            self.statusBar().showMessage('Selection cleared.')
            return
        else:
            self.selected_ball = ball_hit
            color = ball_hit.get('color', '?')
            if self.selected_pocket:
                self.statusBar().showMessage(
                    f'Target: {color} ball → {self.selected_pocket["name"]} pocket.')
            else:
                self.statusBar().showMessage(
                    f'Target: {color} ball. Now click a ◆ pocket to aim.')
        self._update_status_panel()

    def _find_ball_at(self, x: float, y: float) -> Optional[Dict]:
        for ball in self.current_balls:
            cx, cy = ball['center']
            r = ball.get('radius', BALL_RADIUS_TOP_DOWN if self.cam_H else 15)
            if (x - cx) ** 2 + (y - cy) ** 2 <= (r + 8) ** 2:
                return ball
        return None

    def _find_pocket_at(self, x: float, y: float,
                        pockets: List[Dict]) -> Optional[Dict]:
        for pocket in pockets:
            px, py = pocket['cam_pos']
            if (x - px) ** 2 + (y - py) ** 2 <= POCKET_CLICK_RADIUS ** 2:
                return pocket
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Button callbacks
    # ─────────────────────────────────────────────────────────────────────────

    def _on_capture(self):
        self.paused = not self.paused
        if self.paused:
            self.btn_capture.setText('Resume Feed')
            self.btn_capture.setObjectName('btn_mode_active')
            self.statusBar().showMessage('Feed paused — click Resume to continue.')
        else:
            self.btn_capture.setText('Capture Frame')
            self.btn_capture.setObjectName('')
            self.statusBar().showMessage('Feed resumed.')
        self.btn_capture.setStyleSheet('')

    def _on_reset(self):
        self.selected_ball   = None
        self.selected_pocket = None
        self.cue_path        = []
        self.target_path     = []
        self._update_status_panel()
        self.statusBar().showMessage('Selection cleared.')

    def _on_toggle_mode(self):
        if self.mode == MODE_SCREEN:
            self.mode = MODE_PROJECTION
            self.btn_mode.setText('Switch to Screen')
            self.btn_mode.setObjectName('btn_mode_active')
            if self.proj_window is None:
                self.proj_window = ProjectorWindow(close_callback=self._on_toggle_mode)
            self.proj_window.show_on_best_screen()
            screens = QApplication.screens()
            msg = ('Projection mode — overlay on projector. Press Esc to return.'
                   if len(screens) >= 2 else
                   'Projection mode — no second monitor; showing in separate window.')
            self.statusBar().showMessage(msg)
        else:
            self.mode = MODE_SCREEN
            self.btn_mode.setText('Switch to Projection')
            self.btn_mode.setObjectName('')
            if self.proj_window is not None:
                self.proj_window._close_callback = None
                self.proj_window.hide()
                self.proj_window._close_callback = self._on_toggle_mode
            self.statusBar().showMessage('Screen mode.')
        self.btn_mode.setStyleSheet('')
        self._update_status_panel()

    # ─────────────────────────────────────────────────────────────────────────
    # Sidebar status update
    # ─────────────────────────────────────────────────────────────────────────

    def _update_status_panel(self):
        self.lbl_mode.setText(
            f'Mode: {"Screen" if self.mode == MODE_SCREEN else "Projection"}'
        )
        self.lbl_calib.setText(
            f'Calibration: {"OK" if self.cam_H is not None else "none"}'
        )
        self.lbl_balls.setText(f'Balls: {len(self.current_balls)}')
        self.lbl_cue.setText('Cue: found' if self.cue_ball else 'Cue: not found')
        self.lbl_target.setText(
            f'Target: {self.selected_ball.get("color","?") if self.selected_ball else "none"}'
        )
        self.lbl_pocket.setText(
            f'Pocket: {self.selected_pocket["name"] if self.selected_pocket else "none"}'
        )
        self.lbl_detect.setText(
            'Detection: mock' if (self.use_mock or not DETECTION_AVAILABLE)
            else 'Detection: real'
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Keyboard shortcuts
    # ─────────────────────────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        key = event.key()
        if self._stack.currentIndex() == 2:   # only in main view
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
        else:
            super().keyPressEvent(event)

    # ─────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        # Stop calibration page camera/timer if still running
        self._calib_page._stop_camera()
        if self._timer is not None:
            self._timer.stop()
        if self._cap is not None:
            self._cap.release()
        if self.proj_window is not None:
            self.proj_window.close()
        event.accept()


# ═════════════════════════════════════════════════════════════════════════════
# Module-level draw_overlay  (Project_Plan.md interface)
# ═════════════════════════════════════════════════════════════════════════════

def draw_overlay(image: np.ndarray,
                 balls: List[Dict],
                 path:  List[Tuple]) -> np.ndarray:
    """
    Pure-OpenCV rendering — no Qt window, no run loop.
    Returns a copy of *image* with balls and trajectory drawn on it.
    """
    canvas = image.copy()
    if len(path) >= 2:
        _draw_trajectory_static(canvas, [(int(p[0]), int(p[1])) for p in path])
    for ball in balls:
        cx, cy = ball['center']
        bgr    = BALL_COLORS_BGR.get(ball.get('color','gray').lower(),
                                     BALL_COLORS_BGR['gray'])
        radius = ball.get('radius', 15)
        is_cue = ball.get('is_cue', False)
        center = (int(cx), int(cy))
        cv2.circle(canvas, center, radius, bgr, -1)
        cv2.circle(canvas, center, radius,
                   (255, 255, 255) if is_cue else (40, 40, 40), 2)
        if is_cue:
            cv2.circle(canvas, center, max(3, radius // 4), (200, 200, 200), -1)
        cv2.putText(canvas, 'cue' if is_cue else ball.get('color','gray')[:3],
                    (center[0] - 10, center[1] + radius + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


# ═════════════════════════════════════════════════════════════════════════════
# Utilities
# ═════════════════════════════════════════════════════════════════════════════

def _draw_trajectory_static(canvas: np.ndarray,
                             path_px: List[Tuple[int, int]],
                             thickness: int = 2):
    dash_len, gap_len = 14, 7
    for i in range(len(path_px) - 1):
        p1 = np.array(path_px[i],   dtype=np.float32)
        p2 = np.array(path_px[i+1], dtype=np.float32)
        seg    = p2 - p1
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


def _bgr_to_pixmap(frame: np.ndarray) -> QPixmap:
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def _make_mock_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Synthetic green-felt background for mock mode."""
    frame = np.full((height, width, 3), (34, 100, 34), dtype=np.uint8)
    noise = np.random.randint(-10, 10, frame.shape, dtype=np.int16)
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# ═════════════════════════════════════════════════════════════════════════════
# Standalone entry point (for testing without main.py)
# ═════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setApplicationName('Billiards Assistance System')
    window = BilliardsApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
