"""
Billiards Assistance System — GUI / Rendering Module
=====================================================
PyQt5 application with three-page flow:
  Page 0 — SetupPage:       hardware selection (camera index, projector yes/no)
  Page 1 — CalibrationPage: live camera preview, corner clicking, auto-detect
  Page 2 — Main view:       camera feed with ball overlays and trajectory lines

This file is the rendering / GUI layer only.  No game logic or path math.

Public API used by main.py:
  BilliardsApp(tick_callback)   — main window; tick_callback called every 33 ms
  app.grab_frame()              — latest raw BGR camera frame
  app.consume_pending_clicks()  — (x_cm, y_cm) clicks since last call
  app.render(frame, balls, ...) — draw overlays and push to screen/projector
  app.cam_H                     — camera→table homography (set by CalibrationPage)
  app.table_corners             — 4 camera-pixel corner points (set by CalibrationPage)
  app.selected_pocket_cm        — manually selected pocket in table cm (or None)

Coordinate system rule:
  All coordinates ENTERING render() are in table centimetres.
  Conversion to display pixels (table_cm × TABLE_DISPLAY_SCALE) happens here only.
"""

import sys
import os
import pathlib
import shutil
import tempfile
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import cv2

import config

# ── Optional module imports (graceful fallback) ───────────────────────────────
try:
    from Scene_Understanding import (
        get_table_corners,
        compute_homography_from_corners,
    )
    SCENE_AVAILABLE = True
except (ImportError, AttributeError):
    SCENE_AVAILABLE = False

from Ball_Detection import BallDetectionConfig

# ── PyQt5 imports ─────────────────────────────────────────────────────────────
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFrame, QSizePolicy, QStatusBar,
    QStackedWidget, QCheckBox, QSpinBox, QGroupBox,
    QDoubleSpinBox, QComboBox, QScrollArea,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont

# ═════════════════════════════════════════════════════════════════════════════
# Cache paths  (user home — invisible to the user)
# ═════════════════════════════════════════════════════════════════════════════

_CACHE_DIR  = pathlib.Path.home() / '.billiards_assistant'
_CAM_CACHE  = str(_CACHE_DIR / 'camera_homography.npy')
_PROJ_CACHE = str(_CACHE_DIR / 'projector_homography.npy')
_REF_CACHE  = str(_CACHE_DIR / 'ref.jpeg')

# ═════════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════════

MODE_SCREEN     = 'screen'
MODE_PROJECTION = 'projection'

# Top-down (bird's-eye) view — 8 px/cm → 122×61 cm → 976×488 px canvas
TABLE_DISPLAY_SCALE  = 8
BALL_RADIUS_TOP_DOWN = 23   # ≈ 2.875 cm × 8 px/cm (standard pool ball)

# Billiards ball color name → OpenCV BGR
BALL_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    'white':    (255, 255, 255),
    'yellow':   (0,   230, 255),
    'blue':     (210, 100,   0),
    'red':      (0,     0, 210),
    'bordeaux': (0,    30, 120),
    'purple':   (170,   0, 120),
    'orange':   (0,   130, 255),
    'green':    (0,   160,  50),
    'maroon':   (0,    30, 120),
    'black':    (40,   40,  40),
    'gray':     (128, 128, 128),
    'unknown':  (100, 100, 100),
}

COLOR_TRAJECTORY  = (0, 255, 150)   # bright green-cyan  — cue ball path
COLOR_TARGET_PATH = (0, 165, 255)   # orange             — target ball → pocket
COLOR_SELECTION   = (0, 255, 255)   # cyan selection ring
COLOR_POCKET      = (200, 200, 200) # light grey pocket marker
COLOR_POCKET_SEL  = (0, 220, 255)   # highlighted selected pocket

POCKET_CLICK_RADIUS = 30  # px tolerance for clicking a pocket marker

# Standard billiards table: 6 pockets in table-cm coordinates.
# Derived from table dimensions (not hardcoded arbitrary values).
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

        proj_group = QGroupBox('Projector')
        proj_layout = QVBoxLayout(proj_group)
        self._chk_projector = QCheckBox('Projector connected')
        self._chk_projector.setChecked(False)
        proj_layout.addWidget(self._chk_projector)
        outer.addWidget(proj_group)

        self._lbl_mock = QLabel('(No hardware selected — will run in mock/demo mode)')
        self._lbl_mock.setAlignment(Qt.AlignCenter)
        self._lbl_mock.setStyleSheet('color: #888; font-size: 11px;')
        self._lbl_mock.setVisible(False)
        outer.addWidget(self._lbl_mock)

        outer.addStretch()

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
    Emits calibration_done(H) where H is np.ndarray or None.
    After accepting, self.accepted_corners holds the 4 camera-pixel corner points.
    """
    calibration_done = pyqtSignal(object)  # np.ndarray | None

    def __init__(self):
        super().__init__()
        self._cap        = None
        self._timer      = None
        self._frame      = None
        self._corners    : List         = []
        self._drag_idx   : int          = -1
        self.accepted_corners: Optional[np.ndarray] = None
        self._build()

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

        self._cam_label = ClickableLabel()
        self._cam_label.setAlignment(Qt.AlignCenter)
        self._cam_label.setStyleSheet(
            'background-color: #0a0a1a; border: 2px solid #0f3460; border-radius: 4px;'
        )
        self._cam_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._cam_label.setText('Starting camera…')
        self._cam_label.clicked.connect(self._on_label_click)
        layout.addWidget(self._cam_label, stretch=1)

        self._lbl_corners = QLabel('Corners: 0 / 4')
        self._lbl_corners.setAlignment(Qt.AlignCenter)
        self._lbl_corners.setStyleSheet('font-size: 12px; color: #aaa;')
        layout.addWidget(self._lbl_corners)

        btn_row = QHBoxLayout()
        self._btn_auto   = QPushButton('Auto-detect')
        self._btn_accept = QPushButton('Accept')
        self._btn_last   = QPushButton('Use Last')
        self._btn_skip   = QPushButton('Skip')

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

    def start_camera(self, cap):
        """Begin live preview using a shared cv2.VideoCapture."""
        self._corners  = []
        self._drag_idx = -1
        self._frame    = None
        self._update_corner_label()

        self._cap = cap
        if self._cap is None or not self._cap.isOpened():
            self._cam_label.setText('Camera not available')
            return

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start(33)

    def _stop_camera(self):
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _on_timer(self):
        if self._cap is None or not self._cap.isOpened():
            return
        ok, frame = self._cap.read()
        if not ok:
            return
        self._frame = frame
        self._show_frame(frame)

    def _show_frame(self, frame: np.ndarray):
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

    def _label_to_frame(self, lx: int, ly: int) -> Optional[Tuple[int, int]]:
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
            self._corners[idx] = pt
        elif len(self._corners) < 4:
            self._corners.append(pt)
        self._update_corner_label()
        if self._frame is not None:
            self._show_frame(self._frame)

    def _update_corner_label(self):
        n = len(self._corners)
        self._lbl_corners.setText(f'Corners: {n} / 4')
        self._btn_accept.setEnabled(n == 4)

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
        if len(self._corners) != 4:
            return
        corners_arr = np.array(self._corners, dtype=np.float32)
        self.accepted_corners = corners_arr.copy()
        if SCENE_AVAILABLE:
            H = compute_homography_from_corners(corners_arr)
        else:
            H = self._compute_homography_fallback(corners_arr)
        self._finish(H)

    @staticmethod
    def _compute_homography_fallback(corners: np.ndarray) -> np.ndarray:
        """Compute camera->table_cm homography without Scene_Understanding."""
        src = corners.reshape(4, 2).astype(np.float32)
        dst = np.array([
            [0.0,                    0.0],
            [config.TABLE_WIDTH_CM,  0.0],
            [config.TABLE_WIDTH_CM,  config.TABLE_HEIGHT_CM],
            [0.0,                    config.TABLE_HEIGHT_CM],
        ], dtype=np.float32)
        H, _ = cv2.findHomography(src, dst)
        return H.astype(np.float32)

    def _on_use_last(self):
        if not os.path.exists(_CAM_CACHE):
            return
        H = np.load(_CAM_CACHE).astype(np.float32)
        # No corners available from cache — accepted_corners stays None
        self.accepted_corners = None
        self._finish(H)

    def _on_skip(self):
        self.accepted_corners = None
        self._finish(None)

    def _finish(self, H):
        self._stop_camera()
        self.calibration_done.emit(H)


# ═════════════════════════════════════════════════════════════════════════════
# ReferencePage — capture an empty-table reference image (page 2)
# ═════════════════════════════════════════════════════════════════════════════

class ReferencePage(QWidget):
    """
    Shows a live camera preview and asks the user to clear the table,
    then capture a reference frame used for background subtraction.
    Emits reference_done(ref_path: str | None).
    """
    reference_done = pyqtSignal(object)  # str path or None

    def __init__(self):
        super().__init__()
        self._cap   = None
        self._timer = None
        self._frame = None
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QLabel('Reference Capture — Remove all objects from the table')
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet('color: #e94560; font-size: 13px; font-weight: bold;')
        layout.addWidget(header)

        self._instructions = QLabel(
            'Clear the table completely (no balls, no cue).\n'
            'This frame will be used for background subtraction during ball detection.\n'
            'When the table is empty, click "Capture Reference".'
        )
        self._instructions.setAlignment(Qt.AlignCenter)
        self._instructions.setStyleSheet('color: #888; font-size: 11px;')
        layout.addWidget(self._instructions)

        self._cam_label = QLabel()
        self._cam_label.setAlignment(Qt.AlignCenter)
        self._cam_label.setStyleSheet(
            'background-color: #0a0a1a; border: 2px solid #0f3460; border-radius: 4px;'
        )
        self._cam_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._cam_label.setText('Starting camera…')
        layout.addWidget(self._cam_label, stretch=1)

        self._lbl_status = QLabel('')
        self._lbl_status.setAlignment(Qt.AlignCenter)
        self._lbl_status.setStyleSheet('font-size: 12px; color: #aaa;')
        layout.addWidget(self._lbl_status)

        btn_row = QHBoxLayout()
        self._btn_capture = QPushButton('Capture Reference')
        self._btn_last    = QPushButton('Use Last')
        self._btn_skip    = QPushButton('Skip')

        self._btn_last.setEnabled(os.path.exists(_REF_CACHE))

        self._btn_capture.clicked.connect(self._on_capture)
        self._btn_last.clicked.connect(self._on_use_last)
        self._btn_skip.clicked.connect(self._on_skip)

        btn_row.addWidget(self._btn_capture)
        btn_row.addWidget(self._btn_last)
        btn_row.addWidget(self._btn_skip)
        layout.addLayout(btn_row)

    def start_camera(self, cap):
        """Begin live preview using a shared cv2.VideoCapture."""
        self._frame = None
        self._cap = cap
        if self._cap is None or not self._cap.isOpened():
            self._cam_label.setText('Camera not available')
            return

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start(33)

    def _stop_camera(self):
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _on_timer(self):
        if self._cap is None or not self._cap.isOpened():
            return
        ok, frame = self._cap.read()
        if not ok:
            return
        self._frame = frame
        pixmap = _bgr_to_pixmap(frame)
        self._cam_label.setPixmap(
            pixmap.scaled(self._cam_label.width(), self._cam_label.height(),
                          Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _on_capture(self):
        if self._frame is None:
            self._lbl_status.setText('No frame available — wait for camera.')
            return
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(_REF_CACHE, self._frame)
        self._lbl_status.setText(f'Reference saved.')
        print(f'[GUI] Reference image saved to: {_REF_CACHE}')
        self._stop_camera()
        self.reference_done.emit(_REF_CACHE)

    def _on_use_last(self):
        if not os.path.exists(_REF_CACHE):
            return
        self._stop_camera()
        self.reference_done.emit(_REF_CACHE)

    def _on_skip(self):
        self._stop_camera()
        self.reference_done.emit(None)


# ═════════════════════════════════════════════════════════════════════════════
# BilliardsApp — main PyQt5 application window
# ═════════════════════════════════════════════════════════════════════════════

class BilliardsApp(QMainWindow):
    """
    Main application window.

    Four-page flow:
      0 → SetupPage       (hardware selection)
      1 → CalibrationPage (camera calibration)
      2 → ReferencePage   (capture empty-table reference image)
      3 → Main view       (camera feed + overlays)

    Integration with main.py:
      Pass tick_callback to __init__. It is called every 33 ms by the QTimer.
      main.py calls grab_frame(), then does detection + game logic, then calls render().
    """

    def __init__(self, tick_callback: Optional[Callable] = None):
        super().__init__()

        # ── Injected orchestration callback ──────────────────────────────────
        self._tick_callback = tick_callback

        # ── Hardware / mode ──────────────────────────────────────────────────
        self.camera_index    : int  = config.CAMERA_INDEX
        self.use_mock        : bool = False
        self._wants_projector: bool = False

        # ── Rendering state ──────────────────────────────────────────────────
        self.mode   = MODE_SCREEN
        self.paused = False

        # ── Calibration homographies (in-memory only) ────────────────────────
        self.cam_H      : Optional[np.ndarray] = None   # camera px → table cm
        self.proj_H_inv : Optional[np.ndarray] = None   # table cm → projector px

        # ── Table corners (camera px; stored during calibration) ─────────────
        self._table_corners: Optional[np.ndarray] = None   # shape (4, 2) float32

        # ── Reference image path (set by ReferencePage) ──────────────────────
        self.ref_path: Optional[str] = None

        # ── Detection config (mutated live by sidebar settings panel) ─────
        self.detection_config = BallDetectionConfig()

        # ── Pending user clicks (table cm coordinates, drained by main.py) ───
        self._pending_clicks: List[Tuple[float, float]] = []

        # ── Manual pocket selection (UI state, rendering hint for main.py) ───
        self._selected_pocket_name: Optional[str]                  = None
        self._selected_pocket_cm  : Optional[Tuple[float, float]]  = None

        # ── Camera frame ─────────────────────────────────────────────────────
        self.current_frame : Optional[np.ndarray] = None
        self.frozen_frame  : Optional[np.ndarray] = None
        self._cap          = None
        self._timer        = None

        # ── Projector window ─────────────────────────────────────────────────
        self.proj_window: Optional[ProjectorWindow] = None

        # ── Game-state cache for status panel (set by render()) ──────────────
        self._last_stroke_count : int       = 0
        self._last_remaining    : List[str] = []
        self._last_game_state   : str       = 'WAITING_FOR_SHOT'
        self._last_selected_color: Optional[str] = None
        self._last_tracker_state : str       = 'TRACKING'

        # ── Build UI ─────────────────────────────────────────────────────────
        self._build_ui()

    # ─────────────────────────────────────────────────────────────────────────
    # Public API for main.py
    # ─────────────────────────────────────────────────────────────────────────

    def grab_frame(self) -> Optional[np.ndarray]:
        """
        Return the latest raw camera frame (BGR ndarray).
        Handles mock mode and paused state.
        Does NOT apply any warp — returns camera-pixel-space frame.
        """
        if not self.paused:
            frame = self._grab_frame()
            if frame is not None:
                self.current_frame = frame
                self.frozen_frame  = frame.copy()
        else:
            frame = self.frozen_frame
        return frame

    def consume_pending_clicks(self) -> List[Tuple[float, float]]:
        """
        Return and clear the list of user clicks since the last call.
        Each item is (x_cm, y_cm) in table centimetre coordinates.
        """
        result = list(self._pending_clicks)
        self._pending_clicks.clear()
        return result

    def render(
        self,
        frame:          np.ndarray,
        balls:          List[Dict],
        cue_path:       List[Tuple[float, float]],
        target_path:    List[Tuple[float, float]],
        game_state:     str,
        stroke_count:   int,
        remaining:      List[str],
        selected_color: Optional[str] = None,
        tracker_state:  str = 'TRACKING',
    ) -> None:
        """
        Draw overlays on the frame and push to the camera label.
        Also updates the projector window if active.

        Parameters
        ----------
        frame        : raw BGR camera frame (not warped).
        balls        : list of ball dicts with 'center_cm', 'color', 'is_cue'.
                       'center' (camera px) is used as fallback when cam_H is None.
        cue_path     : [(x_cm, y_cm), ...] in table centimetres.
        target_path  : [(x_cm, y_cm), ...] in table centimetres.
        game_state   : current state string (e.g. 'WAITING_FOR_SHOT').
        stroke_count : number of strokes so far.
        remaining    : list of color names still to pocket.
        selected_color : color of the currently selected target ball, or None.
        """
        top_down = self.cam_H is not None
        S = TABLE_DISPLAY_SCALE

        # Cache game state for status panel
        self._last_stroke_count  = stroke_count
        self._last_remaining     = remaining
        self._last_game_state    = game_state
        self._last_selected_color = selected_color
        self._last_tracker_state = tracker_state

        # ── Produce display-space (pixel) canvas ──────────────────────────────
        if top_down:
            canvas = self._warp_to_top_down(frame)
        else:
            canvas = frame.copy()

        # ── Convert cm paths → display pixels ────────────────────────────────
        def cm_to_disp(pt_cm: Tuple[float, float]) -> Tuple[int, int]:
            if top_down:
                return (int(pt_cm[0] * S), int(pt_cm[1] * S))
            p = self._table_to_camera(pt_cm)
            return p if p is not None else None

        cue_path_px    = [cm_to_disp(p) for p in cue_path]
        cue_path_px    = [p for p in cue_path_px if p is not None]
        target_path_px = [cm_to_disp(p) for p in target_path]
        target_path_px = [p for p in target_path_px if p is not None]

        # ── Convert ball centers → display pixels ─────────────────────────────
        # Adds '_disp' key to a copy of each ball dict for rendering.
        balls_disp = []
        for ball in balls:
            cm = ball.get('center_cm')
            if top_down and cm is not None:
                disp = (int(cm[0] * S), int(cm[1] * S))
            else:
                disp = ball.get('center')  # camera px fallback
            balls_disp.append({**ball, '_disp': disp})

        # ── Pocket positions in display space ─────────────────────────────────
        pockets = self._get_pocket_positions(frame)

        # ── Draw screen-mode overlays ─────────────────────────────────────────
        ball_r = BALL_RADIUS_TOP_DOWN if top_down else 15

        # 1. Target path (orange)
        if len(target_path_px) >= 2:
            self._draw_trajectory(canvas, target_path_px, color=COLOR_TARGET_PATH)

        # 2. Cue path (cyan), extended backwards for aiming
        if len(cue_path_px) >= 2:
            extended = self._extend_path_backward(cue_path_px)
            self._draw_trajectory(canvas, extended, color=COLOR_TRAJECTORY)

        # 3. Ghost-ball outline at contact point
        if len(cue_path_px) >= 2 and selected_color is not None:
            ghost = cue_path_px[-1]
            cv2.circle(canvas, ghost, ball_r, COLOR_TRAJECTORY, 2, cv2.LINE_AA)

        # 4. Pocket markers
        for pocket in pockets:
            is_sel = (pocket['name'] == self._selected_pocket_name)
            self._draw_pocket(canvas, pocket['cam_pos'], selected=is_sel)

        # 5. Detected balls
        for ball in balls_disp:
            disp = ball.get('_disp')
            if disp is None:
                continue
            self._draw_ball(
                canvas,
                center     = disp,
                radius     = ball_r,
                color_name = ball.get('color', 'gray'),
                is_cue     = ball.get('is_cue', False),
                selected   = (ball.get('color') == selected_color and not ball.get('is_cue')),
            )

        # 6. Game-state overlays
        if game_state == 'GAME_OVER':
            self._draw_game_over(canvas, stroke_count)
        else:
            self._draw_score_panel(canvas, stroke_count, remaining)

        # 7. Pause indicator
        if self.paused:
            cv2.putText(canvas, 'PAUSED', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2, cv2.LINE_AA)

        # ── Push to camera label ──────────────────────────────────────────────
        self.camera_label.setPixmap(
            _bgr_to_pixmap(canvas).scaled(
                self.camera_label.width(), self.camera_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

        # ── Update projector window ───────────────────────────────────────────
        if self.proj_window is not None and self.proj_window.isVisible():
            proj_canvas = self._render_projection_from_cm(
                balls, cue_path, target_path, selected_color
            )
            self.proj_window.update_frame(proj_canvas)

        # ── Update status panel ───────────────────────────────────────────────
        self._update_status_panel()

    @property
    def table_corners(self) -> Optional[np.ndarray]:
        """4 camera-pixel corner points, shape (4, 2) float32. Set during calibration."""
        return self._table_corners

    @property
    def selected_pocket_cm(self) -> Optional[Tuple[float, float]]:
        """Manually selected pocket in table cm, or None if auto-selection."""
        return self._selected_pocket_cm

    def update_status_bar(self, message: str) -> None:
        """Set the status bar text."""
        self.statusBar().showMessage(message)

    # ─────────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle('Billiards Assistance System')
        self.setMinimumSize(500, 380)
        self.setStyleSheet(_APP_STYLE)

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._setup_page = SetupPage()
        self._setup_page.start_requested.connect(self._on_setup_done)
        self._stack.addWidget(self._setup_page)

        self._calib_page = CalibrationPage()
        self._calib_page.calibration_done.connect(self._on_calibration_done)
        self._stack.addWidget(self._calib_page)

        self._ref_page = ReferencePage()
        self._ref_page.reference_done.connect(self._on_reference_done)
        self._stack.addWidget(self._ref_page)

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
        sidebar.setFixedWidth(240)
        sb = QVBoxLayout(sidebar)
        sb.setContentsMargins(4, 4, 4, 4)
        sb.setSpacing(6)

        title = QLabel('Billiards\nGolf Game')
        title.setAlignment(Qt.AlignCenter)
        tf = QFont()
        tf.setPointSize(13)
        tf.setBold(True)
        title.setFont(tf)
        title.setStyleSheet('color: #e94560; padding: 4px 0;')
        sb.addWidget(title)
        sb.addWidget(self._make_divider())

        hdr = QLabel('GAME STATE')
        hdr.setStyleSheet('color: #888; font-size: 10px; letter-spacing: 1px;')
        sb.addWidget(hdr)

        self.lbl_strokes   = self._make_status_label('Strokes: 0')
        self.lbl_remaining = self._make_status_label('Remaining: —')
        self.lbl_pocket    = self._make_status_label('Pocket: auto')
        for lbl in (self.lbl_strokes, self.lbl_remaining, self.lbl_pocket):
            sb.addWidget(lbl)

        sb.addWidget(self._make_divider())

        hdr2 = QLabel('STATUS')
        hdr2.setStyleSheet('color: #888; font-size: 10px; letter-spacing: 1px;')
        sb.addWidget(hdr2)

        self.lbl_mode    = self._make_status_label('Mode: Screen')
        self.lbl_calib   = self._make_status_label('Calibration: none')
        self.lbl_detect  = self._make_status_label('Detection: waiting')
        self.lbl_tracker = self._make_status_label('Tracker: Tracking')
        for lbl in (self.lbl_mode, self.lbl_calib, self.lbl_detect, self.lbl_tracker):
            sb.addWidget(lbl)

        sb.addWidget(self._make_divider())

        ctrl_hdr = QLabel('CONTROLS')
        ctrl_hdr.setStyleSheet('color: #888; font-size: 10px; letter-spacing: 1px;')
        sb.addWidget(ctrl_hdr)

        self.btn_capture = QPushButton('Capture Frame')
        self.btn_reset   = QPushButton('Reset Selection')
        self.btn_mode    = QPushButton('Switch to Projection')
        self.btn_recalib = QPushButton('Re-calibrate')

        self.btn_capture.setToolTip('Freeze / unfreeze camera feed (Space)')
        self.btn_reset.setToolTip('Clear pocket selection (R)')
        self.btn_mode.setToolTip('Toggle Screen / Projection (M)')
        self.btn_recalib.setToolTip('Go back to calibration page')

        self.btn_capture.clicked.connect(self._on_capture)
        self.btn_reset.clicked.connect(self._on_reset)
        self.btn_mode.clicked.connect(self._on_toggle_mode)
        self.btn_recalib.clicked.connect(self._on_recalibrate)

        for btn in (self.btn_capture, self.btn_reset, self.btn_mode, self.btn_recalib):
            sb.addWidget(btn)

        sb.addWidget(self._make_divider())

        self._det_toggle_btn = QPushButton('Detection Settings ▸')
        self._det_toggle_btn.setStyleSheet('text-align: left; padding: 3px 6px; font-size: 10px;')
        self._det_settings_widget = self._build_detection_settings()
        self._det_settings_widget.setVisible(False)
        self._det_toggle_btn.clicked.connect(self._toggle_detection_settings)
        sb.addWidget(self._det_toggle_btn)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._det_settings_widget)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet('background: transparent;')
        sb.addWidget(scroll, stretch=1)

        hint = QLabel(
            'Click a pocket to aim manually.\n'
            'Right-click to clear pocket.'
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
    # Detection settings panel (collapsible sidebar section)
    # ─────────────────────────────────────────────────────────────────────────

    def _toggle_detection_settings(self):
        vis = not self._det_settings_widget.isVisible()
        self._det_settings_widget.setVisible(vis)
        self._det_toggle_btn.setText('Detection Settings ▾' if vis else 'Detection Settings ▸')

    def _build_detection_settings(self) -> QWidget:
        cfg = self.detection_config
        widget = QWidget()
        lay = QVBoxLayout(widget)
        lay.setContentsMargins(0, 4, 0, 0)
        lay.setSpacing(4)

        lbl_style = 'font-size: 10px; color: #aaa;'
        spin_style = 'font-size: 10px;'

        def _add_label(text):
            lbl = QLabel(text)
            lbl.setStyleSheet(lbl_style)
            lay.addWidget(lbl)

        def _add_double(attr, lo, hi, step, decimals=2):
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setSingleStep(step)
            sb.setDecimals(decimals)
            sb.setValue(getattr(cfg, attr))
            sb.setStyleSheet(spin_style)
            sb.valueChanged.connect(lambda v, a=attr: setattr(cfg, a, v))
            lay.addWidget(sb)
            return sb

        def _add_int(attr, lo, hi, step=1):
            sb = QSpinBox()
            sb.setRange(lo, hi)
            sb.setSingleStep(step)
            sb.setValue(getattr(cfg, attr))
            sb.setStyleSheet(spin_style)
            sb.valueChanged.connect(lambda v, a=attr: setattr(cfg, a, v))
            lay.addWidget(sb)
            return sb

        # -- Detection mode
        _add_label('Detection Mode')
        mode_combo = QComboBox()
        mode_combo.setStyleSheet(spin_style)
        mode_combo.addItems(['Both (Hough + Contour)', 'Hough Only', 'Contour Only'])
        _mode_map = {'both': 0, 'hough': 1, 'contour': 2}
        _mode_rev = {0: 'both', 1: 'hough', 2: 'contour'}
        mode_combo.setCurrentIndex(_mode_map.get(cfg.detection_mode, 0))
        mode_combo.currentIndexChanged.connect(
            lambda idx: setattr(cfg, 'detection_mode', _mode_rev.get(idx, 'both'))
        )
        lay.addWidget(mode_combo)

        # -- Hough parameters
        _add_label('Hough dp')
        _add_double('hough_dp', 1.0, 3.0, 0.1)
        _add_label('Hough param1')
        _add_double('hough_param1', 10.0, 300.0, 5.0, 1)
        _add_label('Hough param2')
        _add_double('hough_param2', 5.0, 50.0, 1.0, 1)

        # -- Contour parameters
        _add_label('Min Circularity')
        _add_double('min_circularity', 0.0, 1.0, 0.05)
        _add_label('Min Area (px)')
        _add_int('min_area_px', 10, 500, 10)

        # -- Common parameters
        _add_label('Non-green Ratio')
        _add_double('non_green_ratio', 0.0, 1.0, 0.05)
        _add_label('Edge Margin')
        _add_double('edge_margin', 0.0, 0.20, 0.01)
        _add_label('Max Balls')
        _add_int('max_balls', 1, 15)

        return widget

    # ─────────────────────────────────────────────────────────────────────────
    # Pipeline flow — setup → calibration → main
    # ─────────────────────────────────────────────────────────────────────────

    def _on_setup_done(self, camera_index: int, has_camera: bool, has_projector: bool):
        self.camera_index     = camera_index
        self.use_mock         = not has_camera
        self._wants_projector = has_projector

        if has_camera:
            self._cap = cv2.VideoCapture(camera_index)
            if not self._cap.isOpened():
                print(f'[GUI] Cannot open camera {camera_index}. Falling back to mock.')
                self._cap = None
                self.use_mock = True
                self._on_calibration_done(None)
                return
            # Let auto-exposure / white-balance settle
            for _ in range(5):
                self._cap.read()

            self._stack.setCurrentIndex(1)
            self.setMinimumSize(800, 520)
            self.statusBar().showMessage(
                'Place 4 table corners or click Auto-detect, then Accept.'
            )
            self._calib_page.start_camera(self._cap)
        else:
            self._on_calibration_done(None)

    def _on_calibration_done(self, H):
        if H is not None:
            self.cam_H = H.astype(np.float32)
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.save(_CAM_CACHE, self.cam_H)
            print(f'[GUI] Camera homography saved to cache: {_CAM_CACHE}')
        else:
            self.cam_H = None
            print('[GUI] No camera calibration — using raw camera coordinates.')

        self._table_corners = self._calib_page.accepted_corners

        if self._wants_projector and self.cam_H is not None:
            self.statusBar().showMessage('Running projector calibration…')
            QApplication.processEvents()
            self._calibrate_projector_in_memory()

        if self.use_mock:
            self._on_reference_done(None)
            return

        # Proceed to reference capture page (same camera session)
        self._stack.setCurrentIndex(2)
        self.statusBar().showMessage(
            'Clear the table and capture a reference image for ball detection.'
        )
        self._ref_page.start_camera(self._cap)

    def _on_reference_done(self, ref_path):
        if ref_path is not None:
            self.ref_path = str(ref_path)
            print(f'[GUI] Using reference image: {self.ref_path}')
        else:
            self.ref_path = None
            print('[GUI] No reference image — ball detection may be degraded.')

        self._stack.setCurrentIndex(3)
        self.setMinimumSize(900, 560)
        self._start_camera()
        self._start_timer()

        calib_status = 'OK' if self.cam_H is not None else 'none'
        self.lbl_calib.setText(f'Calibration: {calib_status}')
        self.statusBar().showMessage('Ready — click a ball to select a target.')

    def _on_recalibrate(self):
        if self._timer is not None:
            self._timer.stop()
        self.cam_H      = None
        self.proj_H_inv = None
        self.ref_path   = None
        self._table_corners = None
        self._stack.setCurrentIndex(1)
        self.setMinimumSize(800, 520)
        self.statusBar().showMessage('Place 4 corners and click Accept.')
        self._calib_page.start_camera(self._cap)

    # ─────────────────────────────────────────────────────────────────────────
    # Projector calibration (in-memory, temp file hidden from user)
    # ─────────────────────────────────────────────────────────────────────────

    def _calibrate_projector_in_memory(self):
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
        if self._cap is None or not self._cap.isOpened():
            print('[GUI] WARNING: Camera not available. Falling back to mock.')
            self.use_mock = True

    def _start_timer(self):
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start(33)   # ~30 fps

    # ─────────────────────────────────────────────────────────────────────────
    # Main timer callback
    # ─────────────────────────────────────────────────────────────────────────

    def _on_timer(self):
        """Called every 33 ms by the QTimer.
        When tick_callback is injected by main.py, delegate entirely to it.
        Otherwise, show a minimal camera feed (no game logic) for standalone testing."""
        if self._tick_callback is not None:
            self._tick_callback()
            return

        # ── Standalone fallback (no game logic / trajectory) ─────────────────
        frame = self.grab_frame()
        if frame is None:
            return

        top_down = self.cam_H is not None
        canvas = self._warp_to_top_down(frame) if top_down else frame.copy()

        # Draw pocket markers on raw feed for visual reference
        pockets = self._get_pocket_positions(frame)
        for pocket in pockets:
            self._draw_pocket(canvas, pocket['cam_pos'])

        if self.paused:
            cv2.putText(canvas, 'PAUSED', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2, cv2.LINE_AA)

        self.camera_label.setPixmap(
            _bgr_to_pixmap(canvas).scaled(
                self.camera_label.width(), self.camera_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Frame acquisition (internal helper)
    # ─────────────────────────────────────────────────────────────────────────

    def _grab_frame(self) -> Optional[np.ndarray]:
        if self.use_mock:
            return _make_mock_frame()
        if self._cap is None or not self._cap.isOpened():
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    # ─────────────────────────────────────────────────────────────────────────
    # Pocket positions — unified display-space coordinates
    # ─────────────────────────────────────────────────────────────────────────

    def _get_pocket_positions(self, raw_frame: np.ndarray) -> List[Dict]:
        """Return 6 pocket dicts each with 'cam_pos' in the current display space."""
        if self.cam_H is not None:
            # Top-down mode: positions are table_cm × TABLE_DISPLAY_SCALE
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
    # Game-state overlays (score panel and game-over screen)
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_score_panel(
        self,
        canvas:       np.ndarray,
        stroke_count: int,
        remaining:    List[str],
    ) -> None:
        """Draw a semi-transparent score panel in the top-right corner."""
        h, w = canvas.shape[:2]
        panel_w, panel_h = 200, 56
        x0 = w - panel_w - 8
        y0 = 8
        # Semi-transparent background
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h),
                      (20, 20, 50), -1)
        cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)
        cv2.rectangle(canvas, (x0, y0), (x0 + panel_w, y0 + panel_h),
                      (80, 80, 160), 1)
        cv2.putText(canvas, f'Strokes: {stroke_count}',
                    (x0 + 8, y0 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, f'Remaining: {len(remaining)}',
                    (x0 + 8, y0 + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    def _draw_game_over(
        self,
        canvas:       np.ndarray,
        stroke_count: int,
    ) -> None:
        """Draw a game-over overlay on the canvas."""
        h, w = canvas.shape[:2]
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 30), -1)
        cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

        # Title
        text1 = 'GAME OVER'
        (tw1, th1), _ = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)
        cv2.putText(canvas, text1,
                    ((w - tw1) // 2, h // 2 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 200, 255), 4, cv2.LINE_AA)

        # Score
        text2 = f'Completed in {stroke_count} stroke{"s" if stroke_count != 1 else ""}!'
        (tw2, _), _ = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.putText(canvas, text2,
                    ((w - tw2) // 2, h // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2, cv2.LINE_AA)

    # ─────────────────────────────────────────────────────────────────────────
    # Projection-mode rendering  (black canvas in projector coordinates)
    # ─────────────────────────────────────────────────────────────────────────

    def _render_projection_from_cm(
        self,
        balls:         List[Dict],
        cue_path:      List[Tuple[float, float]],
        target_path:   List[Tuple[float, float]],
        selected_color: Optional[str],
    ) -> np.ndarray:
        """Build a 1920×1080 black canvas with overlays in projector coordinates.
        All input paths are in table cm; balls have 'center_cm'."""
        canvas = np.zeros(
            (config.PROJECTOR_HEIGHT, config.PROJECTOR_WIDTH, 3), dtype=np.uint8
        )
        if self.proj_H_inv is None:
            cv2.putText(canvas, 'NO PROJECTOR CALIBRATION',
                        (300, 540), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 220), 4)
            return canvas

        def to_proj(pt_cm: Tuple[float, float]) -> Optional[Tuple[int, int]]:
            return self._table_to_projector(pt_cm)

        # Target path
        if len(target_path) >= 2:
            tp_proj = [to_proj(p) for p in target_path]
            tp_proj = [p for p in tp_proj if p]
            if len(tp_proj) >= 2:
                self._draw_trajectory(canvas, tp_proj, thickness=3,
                                      color=COLOR_TARGET_PATH)

        # Cue path (extended backwards)
        if len(cue_path) >= 2:
            cp_proj = [to_proj(p) for p in cue_path]
            cp_proj = [p for p in cp_proj if p]
            if len(cp_proj) >= 2:
                ext = self._extend_path_backward(cp_proj)
                self._draw_trajectory(canvas, ext, thickness=3,
                                      color=COLOR_TRAJECTORY)

        # Ghost-ball outline
        if len(cue_path) >= 2 and selected_color is not None:
            ghost_proj = to_proj(cue_path[-1])
            if ghost_proj:
                cv2.circle(canvas, ghost_proj, 22, COLOR_TRAJECTORY, 2, cv2.LINE_AA)

        # Pockets
        for pocket in POCKET_POSITIONS_TABLE:
            proj_pt = to_proj(pocket['pos'])
            if proj_pt:
                is_sel = (pocket['name'] == self._selected_pocket_name)
                self._draw_pocket(canvas, proj_pt, selected=is_sel, radius=18)

        # Balls
        for ball in balls:
            cm = ball.get('center_cm')
            if cm is None:
                continue
            proj_pt = to_proj(cm)
            if proj_pt is None:
                continue
            self._draw_ball(canvas, proj_pt, 22,
                            ball.get('color', 'gray'),
                            ball.get('is_cue', False),
                            ball.get('color') == selected_color and not ball.get('is_cue'))

        return canvas

    # ─────────────────────────────────────────────────────────────────────────
    # Drawing primitives
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_ball(self, canvas, center, radius, color_name, is_cue, selected=False):
        bgr    = BALL_COLORS_BGR.get(str(color_name).lower(), BALL_COLORS_BGR['gray'])
        border = (255, 255, 255) if is_cue else (40, 40, 40)
        cv2.circle(canvas, center, radius, bgr,    -1)
        cv2.circle(canvas, center, radius, border,  2)
        if is_cue:
            cv2.circle(canvas, center, max(3, radius // 4), (200, 200, 200), -1)
        if selected:
            cv2.circle(canvas, center, radius + 6, COLOR_SELECTION, 3)
        label = 'cue' if is_cue else str(color_name)[:3]
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
    # Top-down warp helper
    # ─────────────────────────────────────────────────────────────────────────

    def _warp_to_top_down(self, frame: np.ndarray) -> np.ndarray:
        """Warp raw camera frame to bird's-eye display-pixel view."""
        w = int(config.TABLE_WIDTH_CM  * TABLE_DISPLAY_SCALE)
        h = int(config.TABLE_HEIGHT_CM * TABLE_DISPLAY_SCALE)
        S = np.array([[TABLE_DISPLAY_SCALE, 0, 0],
                      [0, TABLE_DISPLAY_SCALE, 0],
                      [0, 0,                  1]], dtype=np.float32)
        return cv2.warpPerspective(frame, S @ self.cam_H, (w, h))

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
    # Mouse click → ball click queuing / pocket selection
    # ─────────────────────────────────────────────────────────────────────────

    def _on_camera_click(self, label_x: int, label_y: int):
        """
        Translate a label click to table coordinates.
        • Pocket click  → store pocket selection in app (rendering hint).
        • Ball-area click → enqueue (x_cm, y_cm) for main.py to handle.
        """
        frame = self.current_frame if self.current_frame is not None else self.frozen_frame
        if frame is None:
            return

        lw = self.camera_label.width()
        lh = self.camera_label.height()
        top_down = self.cam_H is not None

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

        pockets = self._get_pocket_positions(frame)

        # ── Check pocket hit ──────────────────────────────────────────────────
        pocket_hit = self._find_pocket_at(find_x, find_y, pockets)
        if pocket_hit is not None:
            if pocket_hit['name'] == self._selected_pocket_name:
                # Toggle off
                self._selected_pocket_name = None
                self._selected_pocket_cm   = None
                self.statusBar().showMessage('Pocket deselected — using auto-selection.')
            else:
                self._selected_pocket_name = pocket_hit['name']
                self._selected_pocket_cm   = pocket_hit['pos']  # table cm tuple
                self.statusBar().showMessage(
                    f'Pocket override: {pocket_hit["name"]}. Click same pocket to deselect.'
                )
            return

        # ── Translate click to table cm and enqueue for main.py ──────────────
        if top_down:
            x_cm = find_x / TABLE_DISPLAY_SCALE
            y_cm = find_y / TABLE_DISPLAY_SCALE
        else:
            result = self._camera_to_table((find_x, find_y))
            if result is None:
                return
            x_cm, y_cm = result

        self._pending_clicks.append((x_cm, y_cm))

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
        """Clear the manual pocket selection. Ball selection is cleared by main.py."""
        self._pending_clicks.clear()
        self._selected_pocket_name = None
        self._selected_pocket_cm   = None
        self.statusBar().showMessage('Pocket selection cleared — using auto-selection.')

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
        self.lbl_strokes.setText(f'Strokes: {self._last_stroke_count}')
        self.lbl_remaining.setText(f'Remaining: {len(self._last_remaining)}')
        self.lbl_pocket.setText(
            f'Pocket: {self._selected_pocket_name or "auto"}'
        )
        self.lbl_detect.setText(
            f'Game: {self._last_game_state.replace("_", " ").title()}'
        )
        self.lbl_tracker.setText(
            f'Tracker: {self._last_tracker_state.title()}'
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Keyboard shortcuts
    # ─────────────────────────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        key = event.key()
        if self._stack.currentIndex() == 3:
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
        # Stop all page timers
        self._calib_page._stop_camera()
        self._ref_page._stop_camera()
        if self._timer is not None:
            self._timer.stop()
        # Single camera release point
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self.proj_window is not None:
            self.proj_window.close()
        event.accept()


# ═════════════════════════════════════════════════════════════════════════════
# Module-level draw_overlay  (Project_Plan.md interface — unchanged)
# ═════════════════════════════════════════════════════════════════════════════

def draw_overlay(image: np.ndarray,
                 balls: List[Dict],
                 path:  List[Tuple]) -> np.ndarray:
    """
    Pure-OpenCV rendering — no Qt window, no run loop.
    Returns a copy of *image* with balls and trajectory drawn on it.
    Expects balls[i]['center'] in image-pixel coordinates.
    """
    canvas = image.copy()
    if len(path) >= 2:
        _draw_trajectory_static(canvas, [(int(p[0]), int(p[1])) for p in path])
    for ball in balls:
        cx, cy = ball['center']
        bgr    = BALL_COLORS_BGR.get(str(ball.get('color', 'gray')).lower(),
                                     BALL_COLORS_BGR['gray'])
        radius = ball.get('radius', 15)
        is_cue = ball.get('is_cue', False)
        center = (int(cx), int(cy))
        cv2.circle(canvas, center, radius, bgr, -1)
        cv2.circle(canvas, center, radius,
                   (255, 255, 255) if is_cue else (40, 40, 40), 2)
        if is_cue:
            cv2.circle(canvas, center, max(3, radius // 4), (200, 200, 200), -1)
        cv2.putText(canvas, 'cue' if is_cue else str(ball.get('color', 'gray'))[:3],
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
    if len(path_px) >= 2:
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
    window = BilliardsApp()   # no tick_callback → standalone fallback mode
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
