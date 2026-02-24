"""
Billiards Assistance System — GUI / Rendering Module
=====================================================
PyQt5 application with three-page flow:
  Page 0 — SetupPage:       hardware selection (camera index, camera yes/no)
  Page 1 — CalibrationPage: live camera preview, corner clicking, auto-detect
  Page 2 — Main view:       camera feed with ball overlays and trajectory lines

This file is the rendering / GUI layer only.  No game logic or path math.

Public API used by main.py:
  BilliardsApp(tick_callback)   — main window; tick_callback called every 33 ms
  app.grab_frame()              — latest raw BGR camera frame
  app.consume_pending_clicks()  — (x_cm, y_cm) clicks since last call
  app.render(frame, balls, ...) — draw overlays and push to screen
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
import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import cv2

import config

# ── Optional module imports (graceful fallback) ───────────────────────────────
try:
    from scene_understanding import (
        get_table_corners,
        compute_homography_from_corners,
    )
    SCENE_AVAILABLE = True
except (ImportError, AttributeError):
    SCENE_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont as _PILImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ── PyQt5 imports ─────────────────────────────────────────────────────────────
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QSizePolicy, QStatusBar,
    QStackedWidget, QCheckBox, QSpinBox, QGroupBox, QLineEdit,
    QProgressBar, QScrollArea,
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon

# ═════════════════════════════════════════════════════════════════════════════
# Cache paths  (user home — invisible to the user)
# ═════════════════════════════════════════════════════════════════════════════

_CACHE_DIR  = pathlib.Path.home() / '.billiards_assistant'
_CAM_CACHE  = str(_CACHE_DIR / 'camera_homography.npy')
_REF_CACHE  = str(_CACHE_DIR / 'ref.jpeg')

# ═════════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════════

# Top-down (bird's-eye) view — 8 px/cm → 122×61 cm → 976×488 px canvas
TABLE_DISPLAY_SCALE  = 8
BALL_RADIUS_TOP_DOWN = 16   # slightly smaller than actual ball size for cleaner overlay

# Golf Mode — fixed starting positions per ball color (table cm, origin = top-left corner)
# Symmetric layout: cluster anchored at x=27; white mirrors at x=122-27=95; spacing=10 cm
GOLF_MODE_STARTING_POSITIONS: Dict[str, Tuple[float, float]] = {
    'blue':     (27.0, 20.5),   # 10 cm above centre
    'yellow':   (27.0, 30.5),   # centre height
    'purple':   (37.0, 30.5),   # 10 cm right of yellow
    'bordeaux': (27.0, 40.5),   # 10 cm below centre
    'white':    (95.0, 30.5),   # mirror x, centre height
}
BALL_IN_POSITION_THRESHOLD_CM: float = 3.0   # cm radius to count as "in position"
PLACEMENT_RING_OFFSET: int = 8               # ring radius = ball_r + this

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

COLOR_PATH       = ( 26,  95, 217)   # Orange  #d95f1a BGR — primary trajectory
COLOR_PATH_2     = ( 16, 138, 192)   # Mustard #c08a10 BGR — secondary trajectory
COLOR_SELECTION  = ( 26,  95, 217)   # Orange ring on selected ball
COLOR_POCKET     = ( 96, 128,  90)   # Sage    #5a8060 BGR
COLOR_POCKET_SEL = ( 38,  58,  28)   # Forest  #1c3a26 BGR

# Ball color name → hex string (for Qt rich-text labels)
_BALL_HEX: Dict[str, str] = {
    'white':    '#f0f0f0',
    'yellow':   '#ffe100',
    'blue':     '#0064d2',
    'red':      '#d20000',
    'bordeaux': '#7a001a',
    'orange':   '#ff6a00',
    'green':    '#00a032',
    'purple':   '#7a0078',
    'gray':     '#888888',
    'unknown':  '#888888',
}

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

# ── OptiCue brand stylesheet ───────────────────────────────────────────────────
_APP_STYLE = """
QMainWindow, QWidget {
    background-color: #f4efe4;
    color: #2a2520;
    font-family: 'DM Sans', 'Segoe UI', sans-serif;
}

QLabel {
    color: #2a2520;
    font-family: 'Lora', Georgia, serif;
    font-size: 13px;
}

QPushButton {
    background-color: #1c3a26;
    color:            #f4efe4;
    border:           none;
    border-radius:    2px;
    padding:          9px 14px;
    font-family:      'DM Sans', 'Segoe UI', sans-serif;
    font-size:        13px;
    font-weight:      500;
    min-height:       34px;
    text-align:       left;
}
QPushButton:hover   { background-color: #2a5c38; }
QPushButton:pressed { background-color: #d95f1a; }
QPushButton:disabled {
    background-color: #ece6d8;
    color:            #b0a48c;
    border:           1px solid #ddd4bc;
}
QPushButton#btn_accent {
    background-color: #d95f1a;
    color:            #f4efe4;
    border:           none;
}
QPushButton#btn_accent:hover   { background-color: #c04a10; }
QPushButton#btn_accent:pressed { background-color: #a03a08; }
QPushButton#btn_ghost {
    background-color: transparent;
    color:            #2a2520;
    border:           1px solid #c4b898;
}
QPushButton#btn_ghost:hover   { border-color: #1c3a26; color: #1c3a26; }
QPushButton#btn_ghost:pressed { background-color: #ece6d8; }
QPushButton#btn_active {
    background-color: #d95f1a;
    color:            #f4efe4;
    border:           none;
}
QPushButton#btn_active:hover { background-color: #c04a10; }

QGroupBox {
    background-color: #f4efe4;
    border: 1px solid #c4b898;
    border-radius: 2px;
    margin-top: 12px;
    padding: 10px 8px 8px 8px;
    font-size: 9px;
    letter-spacing: 2px;
    color: #8a7d68;
    font-family: 'DM Sans', 'Segoe UI', sans-serif;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    background-color: transparent;
    text-transform: uppercase;
}

QCheckBox {
    color: #2a2520;
    font-size: 13px;
    font-family: 'DM Sans', 'Segoe UI', sans-serif;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #c4b898;
    border-radius: 2px;
    background-color: #f9f6f0;
}
QCheckBox::indicator:checked {
    background-color: #1c3a26;
    border-color: #1c3a26;
}

QSpinBox {
    background-color: #f9f6f0;
    color: #2a2520;
    border: 1px solid #c4b898;
    border-radius: 2px;
    padding: 4px 8px;
    font-family: 'DM Sans', 'Segoe UI', sans-serif;
}
QSpinBox:focus { border-color: #1c3a26; }

QFrame#divider {
    color: #c4b898;
    background-color: #c4b898;
    max-height: 1px;
}

QStatusBar {
    background-color: #ece6d8;
    color: #8a7d68;
    font-size: 10px;
    font-family: 'DM Sans', 'Segoe UI', sans-serif;
    border-top: 1px solid #c4b898;
}

QScrollArea { background-color: transparent; border: none; }
QScrollBar:vertical {
    background: #ece6d8;
    width: 6px;
    border-radius: 3px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #c4b898;
    border-radius: 3px;
    min-height: 20px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""

# Button inline-stylesheet constants.
# These are needed because sidebar container widgets use unscoped
# setStyleSheet('background-color: ...'), which Qt treats as
# QWidget { background-color: ... } — cascading to all descendants and
# overriding the global _APP_STYLE QPushButton rules.  Explicit per-widget
# stylesheets always win, so every button state is set directly.
_BTN_PRIMARY_SS = (
    'QPushButton { background-color: #1c3a26; color: #f4efe4; border: none; }'
    'QPushButton:hover   { background-color: #2a5c38; }'
    'QPushButton:pressed { background-color: #d95f1a; }'
)
_BTN_ACCENT_SS = (
    'QPushButton { background-color: #d95f1a; color: #f4efe4; border: none; }'
    'QPushButton:hover   { background-color: #c04a10; }'
    'QPushButton:pressed { background-color: #a03a08; }'
)
_BTN_GHOST_SS = (
    'QPushButton          { background-color: transparent; color: #2a2520; border: 1px solid #c4b898; }'
    'QPushButton:hover    { border-color: #1c3a26; color: #1c3a26; }'
    'QPushButton:pressed  { background-color: #ece6d8; }'
    'QPushButton:disabled { background-color: #ece6d8; color: #b0a48c; border: 1px solid #ddd4bc; }'
)
_BTN_DISABLED_SS = (
    'QPushButton { background-color: #ece6d8; color: #b0a48c; border: 1px solid #ddd4bc; }'
)


# ═════════════════════════════════════════════════════════════════════════════
# ClickableLabel — QLabel that emits a signal on left mouse click
# ═════════════════════════════════════════════════════════════════════════════

class ClickableLabel(QLabel):
    """QLabel that emits (x, y) in label pixel coords on click, drag, and release."""
    clicked       = pyqtSignal(int, int)
    right_clicked = pyqtSignal(int, int)
    dragged       = pyqtSignal(int, int)
    released      = pyqtSignal(int, int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dragging = False
        self.setMouseTracking(False)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self.clicked.emit(event.x(), event.y())
        elif event.button() == Qt.RightButton:
            self.right_clicked.emit(event.x(), event.y())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging:
            self.dragged.emit(event.x(), event.y())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self.released.emit(event.x(), event.y())
        super().mouseReleaseEvent(event)


# ═════════════════════════════════════════════════════════════════════════════
# SetupPage — hardware selection (page 0)
# ═════════════════════════════════════════════════════════════════════════════

class SetupPage(QWidget):
    """
    Hardware configuration screen shown on startup.
    Emits start_requested(camera_index, has_camera).
    """
    start_requested = pyqtSignal(int, bool)

    def __init__(self):
        super().__init__()
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(40, 40, 40, 40)
        outer.setSpacing(20)

        # Logo area — silently skipped if file missing
        import os as _os
        _logo_path = _os.path.join(_os.path.dirname(__file__), 'assets', 'opticue_logo.png')
        if _os.path.exists(_logo_path):
            from PyQt5.QtGui import QPixmap as _QPixmap
            logo_lbl = QLabel()
            logo_lbl.setAlignment(Qt.AlignCenter)
            logo_lbl.setMaximumHeight(130)
            pix = _QPixmap(_logo_path)
            logo_lbl.setPixmap(pix.scaledToHeight(110, Qt.SmoothTransformation))
            outer.addWidget(logo_lbl)

        title = QLabel()
        title.setTextFormat(Qt.RichText)
        title.setText(
            '<span style="font-family:\'Playfair Display\',Georgia,serif;'
            ' font-weight:900; font-size:32px; color:#1c3a26;">Opti</span>'
            '<span style="font-family:\'Playfair Display\',Georgia,serif;'
            ' font-weight:700; font-style:italic; font-size:32px; color:#d95f1a;">Cue</span>'
        )
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet('padding: 8px 0 2px 0;')
        outer.addWidget(title)

        sub = QLabel('Configure your hardware to begin.')
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet("""
            font-family: 'Lora', 'Georgia', serif;
            font-size: 13px;
            font-style: italic;
            color: #8a7d68;
            padding-bottom: 8px;
        """)
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

        outer.addStretch()

        self._btn_start = QPushButton('Start  →')
        self._btn_start.setMinimumHeight(46)
        self._btn_start.setStyleSheet("""
            background-color: #1c3a26;
            color: #f4efe4;
            border: none;
            border-radius: 2px;
            font-family: 'DM Sans', 'Segoe UI', sans-serif;
            font-size: 15px;
            font-weight: 500;
            padding: 12px;
            text-align: center;
        """)
        self._btn_start.clicked.connect(self._on_start)
        outer.addWidget(self._btn_start)

        self._on_camera_toggled(True)

    def _on_camera_toggled(self, checked: bool):
        self._spin_camera.setEnabled(checked)

    def _on_start(self):
        self.start_requested.emit(
            self._spin_camera.value(),
            self._chk_camera.isChecked(),
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

    _CORNER_NAMES = ['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left']

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Left: camera area (stack: live feed | loading overlay) ────────────
        self._cam_label = ClickableLabel()
        self._cam_label.setAlignment(Qt.AlignCenter)
        self._cam_label.setStyleSheet(
            'background-color: #1a1a1a; border: 1px solid #c4b898;'
        )
        self._cam_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._cam_label.setText('Starting camera…')
        self._cam_label.clicked.connect(self._on_label_click)
        self._cam_label.dragged.connect(self._on_label_drag)
        self._cam_label.released.connect(self._on_label_release)

        # Loading overlay (shown while camera is opening)
        loading_w = QWidget()
        loading_w.setStyleSheet('background-color: #1a1a1a;')
        lw = QVBoxLayout(loading_w)
        lw.setAlignment(Qt.AlignCenter)
        lw.setSpacing(16)

        load_title = QLabel('Opening Camera')
        load_title.setAlignment(Qt.AlignCenter)
        load_title.setStyleSheet(
            "font-family: 'Playfair Display', Georgia, serif;"
            " font-size: 24px; font-weight: bold; color: #f4efe4;"
        )
        lw.addWidget(load_title)

        load_sub = QLabel('Please wait\u2026')
        load_sub.setAlignment(Qt.AlignCenter)
        load_sub.setStyleSheet(
            "font-family: 'Lora', Georgia, serif;"
            " font-style: italic; font-size: 14px; color: #8a7d68;"
        )
        lw.addWidget(load_sub)

        self._loading_bar = QProgressBar()
        self._loading_bar.setRange(0, 0)       # indeterminate / marquee
        self._loading_bar.setFixedHeight(10)
        self._loading_bar.setFixedWidth(280)
        self._loading_bar.setTextVisible(False)
        self._loading_bar.setStyleSheet("""
            QProgressBar {
                background-color: #2e2e2e;
                border: none;
                border-radius: 5px;
            }
            QProgressBar::chunk {
                background-color: #1c3a26;
                border-radius: 5px;
            }
        """)
        lw.addWidget(self._loading_bar, alignment=Qt.AlignCenter)

        # Stack: page 0 = live cam, page 1 = loading overlay
        self._cam_stack = QStackedWidget()
        self._cam_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._cam_stack.addWidget(self._cam_label)   # index 0
        self._cam_stack.addWidget(loading_w)          # index 1
        root.addWidget(self._cam_stack, stretch=1)

        # ── Right: sidebar ────────────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(260)
        sidebar.setStyleSheet('background-color: #ece6d8;')
        outer_sb_layout = QVBoxLayout(sidebar)
        outer_sb_layout.setContentsMargins(0, 0, 0, 0)
        outer_sb_layout.setSpacing(0)

        calib_scroll = QScrollArea()
        calib_scroll.setFrameShape(QFrame.NoFrame)
        calib_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        calib_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        calib_scroll.setWidgetResizable(True)
        calib_scroll.setStyleSheet('background-color: #ece6d8;')

        calib_content = QWidget()
        calib_content.setStyleSheet('background-color: #ece6d8;')
        sb = QVBoxLayout(calib_content)
        sb.setContentsMargins(16, 16, 16, 16)
        sb.setSpacing(10)

        title_lbl = QLabel('Calibration')
        title_lbl.setStyleSheet("""
            font-family: 'Playfair Display', Georgia, serif;
            font-size: 22px;
            font-weight: bold;
            color: #1c3a26;
        """)
        sb.addWidget(title_lbl)

        sub_lbl = QLabel('Click all 4 table corners')
        sub_lbl.setStyleSheet("""
            font-family: 'Lora', Georgia, serif;
            font-style: italic;
            font-size: 12px;
            color: #8a7d68;
        """)
        sb.addWidget(sub_lbl)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet('background-color: #c4b898; max-height: 1px; border: none;')
        sb.addWidget(divider)

        # ── 2×2 corner cross ──────────────────────────────────────────────────
        # Corners are placed spatially: TL=0 / TR=1 (top row), BL=3 / BR=2 (bottom row)
        # A single vsep widget spans all 3 rows so the vertical line is continuous.
        # The horizontal bar is split into left/right halves meeting the vsep at centre.
        def _make_hsep_half():
            f = QFrame()
            f.setFrameShape(QFrame.HLine)
            f.setFixedHeight(1)
            f.setStyleSheet('background-color: #d4b896; border: none;')
            return f

        cross = QWidget()
        cross.setStyleSheet('background-color: transparent;')
        cross_grid = QGridLayout(cross)
        cross_grid.setContentsMargins(0, 0, 0, 0)
        cross_grid.setSpacing(0)

        self._corner_rows = []
        for _ in range(4):
            lbl = QLabel()
            lbl.setTextFormat(Qt.RichText)
            lbl.setMinimumHeight(28)
            lbl.setStyleSheet('padding: 3px 6px; background: transparent; border: none;')
            self._corner_rows.append(lbl)

        # Single vertical separator spanning all 3 rows (always at grid centre)
        vsep = QFrame()
        vsep.setFrameShape(QFrame.VLine)
        vsep.setFixedWidth(1)
        vsep.setStyleSheet('background-color: #d4b896; border: none;')

        cross_grid.addWidget(self._corner_rows[0], 0, 0)      # TL
        cross_grid.addWidget(vsep,                 0, 1, 3, 1) # vertical line spans rows 0-2
        cross_grid.addWidget(self._corner_rows[1], 0, 2)      # TR
        cross_grid.addWidget(_make_hsep_half(),    1, 0)      # H-sep left half
        # cell (1, 1) is occupied by vsep row-span — no widget needed
        cross_grid.addWidget(_make_hsep_half(),    1, 2)      # H-sep right half
        cross_grid.addWidget(self._corner_rows[3], 2, 0)      # BL
        cross_grid.addWidget(self._corner_rows[2], 2, 2)      # BR

        cross_grid.setColumnStretch(0, 1)
        cross_grid.setColumnStretch(2, 1)

        sb.addWidget(cross)

        # Status badge (count label below the cross)
        self._lbl_corner_status = QLabel('0 of 4 corners placed')
        self._lbl_corner_status.setWordWrap(True)
        self._lbl_corner_status.setStyleSheet("""
            background-color: #fdf6e3;
            border: 1px solid #d4b896;
            border-radius: 2px;
            padding: 6px 10px;
            font-family: 'Lora', Georgia, serif;
            font-style: italic;
            font-size: 12px;
            color: #8a7d68;
        """)
        sb.addWidget(self._lbl_corner_status)

        # Keep a hidden legacy label so _update_corner_label can still be called
        self._lbl_corners = QLabel()
        self._lbl_corners.setVisible(False)

        sb.addStretch()

        self._btn_auto   = QPushButton('Auto-detect')
        self._btn_clear  = QPushButton('Clear Corners')
        self._btn_accept = QPushButton('Confirm Calibration')

        self._btn_auto.setStyleSheet(_BTN_PRIMARY_SS)
        self._btn_clear.setEnabled(False)
        self._btn_clear.setStyleSheet(_BTN_DISABLED_SS)
        self._btn_accept.setEnabled(False)
        self._btn_accept.setStyleSheet(_BTN_DISABLED_SS)

        self._btn_auto.clicked.connect(self._on_auto_detect)
        self._btn_clear.clicked.connect(self._on_clear)
        self._btn_accept.clicked.connect(self._on_accept)

        for btn in (self._btn_auto, self._btn_clear, self._btn_accept):
            sb.addWidget(btn)

        calib_scroll.setWidget(calib_content)
        outer_sb_layout.addWidget(calib_scroll)

        root.addWidget(sidebar)

    def start_camera(self, cap):
        """Begin live preview using a shared cv2.VideoCapture."""
        self._corners  = []
        self._drag_idx = -1
        self._frame    = None
        self.accepted_corners = None
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
            color = (16, 74, 192) if i == self._drag_idx else (26, 95, 217)
            cv2.circle(display, pt, 8,  color, -1)
            cv2.circle(display, pt, 15, color,  2)

        if len(pts) >= 2:
            arr = np.array(pts, np.int32)
            cv2.polylines(display, [arr], len(pts) == 4, (26, 95, 217), 2)

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

    def _nearest_corner(self, fx: int, fy: int, threshold: int = 30) -> int:
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
            self._drag_idx = idx
            self._corners[idx] = pt
        elif len(self._corners) < 4:
            self._corners.append(pt)
        self._update_corner_label()
        if self._frame is not None:
            self._show_frame(self._frame)

    def _on_label_drag(self, lx: int, ly: int):
        if self._drag_idx < 0 or self._drag_idx >= len(self._corners):
            return
        pt = self._label_to_frame(lx, ly)
        if pt is None:
            return
        self._corners[self._drag_idx] = pt
        if self._frame is not None:
            self._show_frame(self._frame)

    def _on_label_release(self, lx: int, ly: int):
        if self._drag_idx >= 0:
            pt = self._label_to_frame(lx, ly)
            if pt is not None and self._drag_idx < len(self._corners):
                self._corners[self._drag_idx] = pt
            self._drag_idx = -1
            self._update_corner_label()
            if self._frame is not None:
                self._show_frame(self._frame)

    def _update_corner_label(self):
        n = len(self._corners)
        for i, row_lbl in enumerate(self._corner_rows):
            if i < n:
                row_lbl.setText(
                    '<span style="color:#d95f1a; font-size:15px; font-weight:bold;">✓</span>'
                    f'&nbsp;&nbsp;<span style="color:#2a2520; font-size:13px;">'
                    f'Corner {i + 1} set</span>'
                )
            else:
                name = self._CORNER_NAMES[i]
                row_lbl.setText(
                    f'<span style="color:#c4b898; font-size:13px;">{i + 1}</span>'
                    f'&nbsp;&nbsp;<span style="color:#c4b898; font-size:13px;">'
                    f'{name}…</span>'
                )
        self._lbl_corner_status.setText(f'{n} of 4 corners placed')

        # Clear Corners: disabled until ≥1 corner placed, then Accent
        if n > 0:
            self._btn_clear.setEnabled(True)
            self._btn_clear.setStyleSheet(_BTN_ACCENT_SS)
        else:
            self._btn_clear.setEnabled(False)
            self._btn_clear.setStyleSheet(_BTN_DISABLED_SS)

        # Confirm Calibration: disabled until exactly 4 corners → Primary
        if n == 4:
            self._btn_accept.setEnabled(True)
            self._btn_accept.setStyleSheet(_BTN_PRIMARY_SS)
        else:
            self._btn_accept.setEnabled(False)
            self._btn_accept.setStyleSheet(_BTN_DISABLED_SS)

    def _on_clear(self):
        self._corners = []
        self._drag_idx = -1
        self._update_corner_label()
        if self._frame is not None:
            self._show_frame(self._frame)

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
        """Compute camera->table_cm homography without scene_understanding."""
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

    def start_loading_animation(self):
        """Show the progress-bar loading overlay while the camera opens."""
        self._cam_stack.setCurrentIndex(1)   # loading overlay

    def stop_loading_animation(self):
        """Return to the camera feed view."""
        self._cam_stack.setCurrentIndex(0)   # live cam label
        self._cam_label.setText('')

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
# PlayerRegistrationPage — arcade-style player name entry (page 1)
# ═════════════════════════════════════════════════════════════════════════════

class PlayerRegistrationPage(QWidget):
    """
    Shown before each game round.  One player enters their name and clicks
    Continue.  Emits registration_done(name: str).
    """
    registration_done = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(60, 40, 60, 40)
        outer.setSpacing(16)

        # Logo area — silently skipped if file missing
        import os as _os
        _logo_path = _os.path.join(_os.path.dirname(__file__), 'assets', 'opticue_logo.png')
        if _os.path.exists(_logo_path):
            from PyQt5.QtGui import QPixmap as _QPixmap
            logo_lbl = QLabel()
            logo_lbl.setAlignment(Qt.AlignCenter)
            logo_lbl.setMaximumHeight(130)
            pix = _QPixmap(_logo_path)
            logo_lbl.setPixmap(pix.scaledToHeight(110, Qt.SmoothTransformation))
            outer.addWidget(logo_lbl)

        title = QLabel('Player Registration')
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            font-family: 'Playfair Display', Georgia, serif;
            font-size: 28px;
            font-weight: bold;
            color: #1c3a26;
            padding: 8px 0 2px 0;
        """)
        outer.addWidget(title)

        sub = QLabel('Enter your name to play.')
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet("""
            font-family: 'Lora', 'Georgia', serif;
            font-size: 13px;
            font-style: italic;
            color: #8a7d68;
            padding-bottom: 12px;
        """)
        outer.addWidget(sub)

        self._edit = QLineEdit()
        self._edit.setPlaceholderText('Your name…')
        self._edit.setMinimumHeight(40)
        self._edit.setStyleSheet("""
            background-color: #f9f6f0;
            color: #2a2520;
            border: 1px solid #c4b898;
            border-radius: 2px;
            padding: 8px 12px;
            font-size: 14px;
            font-family: 'DM Sans', 'Segoe UI', sans-serif;
        """)
        self._edit.returnPressed.connect(self._on_continue)
        outer.addWidget(self._edit)

        self._lbl_error = QLabel('')
        self._lbl_error.setAlignment(Qt.AlignCenter)
        self._lbl_error.setStyleSheet('color: #d95f1a; font-size: 12px;')
        self._lbl_error.setVisible(False)
        outer.addWidget(self._lbl_error)

        outer.addStretch()

        self._btn_continue = QPushButton('Continue  →')
        self._btn_continue.setMinimumHeight(46)
        self._btn_continue.setEnabled(False)
        self._btn_continue.clicked.connect(self._on_continue)
        self._edit.textChanged.connect(
            lambda txt: self._btn_continue.setEnabled(bool(txt.strip()))
        )
        outer.addWidget(self._btn_continue)

    def showEvent(self, event):
        """Clear the name field whenever the page is shown."""
        super().showEvent(event)
        self._edit.clear()
        self._lbl_error.setVisible(False)
        self._edit.setFocus()
        self._btn_continue.setEnabled(False)

    def _on_continue(self):
        name = self._edit.text().strip()
        if not name:
            self._lbl_error.setText('Please enter your name.')
            self._lbl_error.setVisible(True)
            return
        self._lbl_error.setVisible(False)
        self.registration_done.emit(name)


# ═════════════════════════════════════════════════════════════════════════════
# LeaderboardPage — session high-score board (page 4)
# ═════════════════════════════════════════════════════════════════════════════

class LeaderboardPage(QWidget):
    """
    Shown after each game round.
    Displays all scores recorded in the current session, ranked by strokes.
    Emits play_again, recalibrate_requested, or quit_requested.
    """
    play_again           = pyqtSignal()
    recalibrate_requested = pyqtSignal()
    quit_requested       = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(60, 40, 60, 40)
        outer.setSpacing(20)

        title = QLabel('High Scores')
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            font-family: 'Playfair Display', Georgia, serif;
            font-size: 32px;
            font-weight: bold;
            color: #1c3a26;
            padding: 8px 0 2px 0;
        """)
        outer.addWidget(title)

        sub = QLabel('Fewer strokes wins.')
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet("""
            font-family: 'Lora', 'Georgia', serif;
            font-size: 13px;
            font-style: italic;
            color: #8a7d68;
            padding-bottom: 12px;
        """)
        outer.addWidget(sub)

        # Dynamic score rows are placed inside this container
        self._scores_container = QWidget()
        self._scores_layout = QVBoxLayout(self._scores_container)
        self._scores_layout.setSpacing(6)
        self._scores_layout.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._scores_container)

        outer.addStretch()

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        btn_play = QPushButton('Play Again')
        btn_play.setMinimumHeight(42)
        btn_play.clicked.connect(self.play_again.emit)

        btn_recalib = QPushButton('Re-calibrate')
        btn_recalib.setObjectName('btn_ghost')
        btn_recalib.setMinimumHeight(42)
        btn_recalib.clicked.connect(self.recalibrate_requested.emit)

        btn_quit = QPushButton('Quit')
        btn_quit.setObjectName('btn_accent')
        btn_quit.setMinimumHeight(42)
        btn_quit.clicked.connect(self.quit_requested.emit)

        btn_row.addWidget(btn_play)
        btn_row.addWidget(btn_recalib)
        btn_row.addWidget(btn_quit)
        outer.addLayout(btn_row)

    def update_scores(self, rankings):
        """
        Populate the score table.
        rankings: List[Tuple[str, int]] sorted ascending by strokes.
        """
        # Clear existing rows
        while self._scores_layout.count():
            item = self._scores_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not rankings:
            no_scores = QLabel('No scores yet.')
            no_scores.setAlignment(Qt.AlignCenter)
            no_scores.setStyleSheet('color: #8a7d68; font-size: 13px;')
            self._scores_layout.addWidget(no_scores)
            return

        for rank, (name, strokes) in enumerate(rankings, 1):
            row = QWidget()
            row.setStyleSheet("""
                background-color: %s;
                border: 1px solid #c4b898;
                border-radius: 2px;
            """ % ('#f9f6f0' if rank == 1 else '#f4efe4'))
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(12, 8, 12, 8)

            rank_lbl = QLabel(f'#{rank}')
            rank_lbl.setStyleSheet("""
                font-family: 'Playfair Display', Georgia, serif;
                font-size: 16px;
                font-weight: bold;
                color: %s;
                min-width: 40px;
            """ % ('#d95f1a' if rank == 1 else '#1c3a26'))

            name_lbl = QLabel(name)
            name_lbl.setStyleSheet(
                "font-family: 'DM Sans', 'Segoe UI', sans-serif;"
                " font-size: 14px; color: #2a2520;"
            )

            score_lbl = QLabel(f'{strokes} strokes')
            score_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            score_lbl.setStyleSheet(
                "font-family: 'DM Sans', 'Segoe UI', sans-serif;"
                " font-size: 14px; color: #8a7d68;"
            )

            row_layout.addWidget(rank_lbl)
            row_layout.addWidget(name_lbl, stretch=1)
            row_layout.addWidget(score_lbl)
            self._scores_layout.addWidget(row)


# ═════════════════════════════════════════════════════════════════════════════
# CameraOpenThread — opens cv2.VideoCapture without blocking the UI thread
# ═════════════════════════════════════════════════════════════════════════════

class CameraOpenThread(QThread):
    """Opens cv2.VideoCapture in a background thread to avoid freezing the UI."""
    opened = pyqtSignal(object)   # emits cv2.VideoCapture or None

    def __init__(self, camera_index: int):
        super().__init__()
        self._index = camera_index

    def run(self):
        cap = cv2.VideoCapture(self._index)
        if cap.isOpened():
            for _ in range(5):    # warm-up reads so auto-exposure settles
                cap.read()
            self.opened.emit(cap)
        else:
            cap.release()
            self.opened.emit(None)


# ═════════════════════════════════════════════════════════════════════════════
# BilliardsApp — main PyQt5 application window
# ═════════════════════════════════════════════════════════════════════════════

class BilliardsApp(QMainWindow):
    """
    Main application window.

    Five-page flow:
      0 → SetupPage              (hardware selection)
      1 → PlayerRegistrationPage (arcade-style name entry)
      2 → CalibrationPage        (camera calibration; auto-captures reference frame)
      3 → Main view              (camera feed + overlays)
      4 → LeaderboardPage        (session high-score board)

    Integration with main.py:
      Pass tick_callback to __init__. It is called every 33 ms by the QTimer.
      main.py calls grab_frame(), then does detection + game logic, then calls render().
    """

    def __init__(self, tick_callback: Optional[Callable] = None):
        super().__init__()

        # ── Injected orchestration callback ──────────────────────────────────
        self._tick_callback = tick_callback

        # ── Hardware ─────────────────────────────────────────────────────────
        self.camera_index    : int  = config.CAMERA_INDEX

        # ── Rendering state ──────────────────────────────────────────────────
        self.paused    = False
        self._raw_view = False

        # ── Calibration homographies (in-memory only) ────────────────────────
        self.cam_H      : Optional[np.ndarray] = None   # camera px → table cm

        # ── Table corners (camera px; stored during calibration) ─────────────
        self._table_corners: Optional[np.ndarray] = None   # shape (4, 2) float32

        # ── Reference image path (auto-captured from calibration frame) ───────
        self.ref_path: Optional[str] = None

        # ── Run directory (created at calibration, used for all captures) ─────
        self._run_dir: Optional[str] = None

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

        # ── Game-state cache for status panel (set by render()) ──────────────
        self._last_stroke_count : int       = 0
        self._last_remaining    : List[str] = []
        self._last_game_state   : str       = 'WAITING_FOR_BALLS'
        self._last_selected_color: Optional[str] = None
        self._last_cue_present  : bool      = True   # False when cue ball leaves table
        self._render_tick       : int       = 0      # drives pulsing animation

        # ── Player / leaderboard state ───────────────────────────────────────
        self._current_player_name: str  = ''
        self.player_scores       : list = []   # [(name, strokes), …] session accumulates

        # ── Build UI ─────────────────────────────────────────────────────────
        self._build_ui()

    # ─────────────────────────────────────────────────────────────────────────
    # Public API for main.py
    # ─────────────────────────────────────────────────────────────────────────

    def grab_frame(self) -> Optional[np.ndarray]:
        """
        Return the latest raw camera frame (BGR ndarray).
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
    ) -> None:
        """
        Draw overlays on the frame and push to the camera label.

        Parameters
        ----------
        frame        : raw BGR camera frame (not warped).
        balls        : list of ball dicts with 'center_cm', 'color', 'is_cue'.
                       'center' (camera px) is used as fallback when cam_H is None.
        cue_path     : [(x_cm, y_cm), ...] in table centimetres.
        target_path  : [(x_cm, y_cm), ...] in table centimetres.
        game_state   : current state string (e.g. 'TRACKING', 'DISTURBED').
        stroke_count : number of strokes so far.
        remaining    : list of color names still to pocket.
        selected_color : color of the currently selected target ball, or None.
        """
        top_down = self.cam_H is not None
        S = TABLE_DISPLAY_SCALE
        self._render_tick += 1

        # Cache game state for status panel
        self._last_stroke_count  = stroke_count
        self._last_remaining     = remaining
        self._last_game_state    = game_state
        self._last_selected_color = selected_color
        # Track cue ball and cache raw ball list (used by _on_start_game)
        self._last_balls = balls
        if balls:
            self._last_cue_present = any(b.get('is_cue') for b in balls)

        # ── Produce display-space (pixel) canvas ──────────────────────────────
        if top_down and not self._raw_view:
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

        # ── Draw screen-mode overlays ─────────────────────────────────────────
        ball_r = BALL_RADIUS_TOP_DOWN if top_down else 10

        if game_state != 'GAME_OVER':
            # 1. Target path (orange)
            if len(target_path_px) >= 2:
                self._draw_trajectory(canvas, target_path_px, color=COLOR_PATH_2)

            # 2. Cue path (cyan), extended backwards for aiming
            if len(cue_path_px) >= 2:
                extended = self._extend_path_backward(cue_path_px)
                self._draw_trajectory(canvas, extended, color=COLOR_PATH)

            # 3. Ghost-ball outline at contact point
            if len(cue_path_px) >= 2 and selected_color is not None:
                ghost = cue_path_px[-1]
                cv2.circle(canvas, ghost, ball_r, COLOR_PATH, 2, cv2.LINE_AA)

            # 3b. Placement target rings (top-down mode only)
            if top_down:
                import math
                from game_tracker import ST_WAITING_FOR_BALLS as _ST_WAIT
                game_obj     = getattr(self, 'game', None)
                cue_pocketed = getattr(game_obj, 'cue_ball_pocketed', False)
                ring_r       = ball_r + PLACEMENT_RING_OFFSET

                if game_state == _ST_WAIT:
                    # Pre-game: show all 5 placement rings
                    for color, target_cm in GOLF_MODE_STARTING_POSITIONS.items():
                        ctr = cm_to_disp(target_cm)
                        if ctr is not None:
                            self._draw_placement_ring(
                                canvas, ctr, ring_r,
                                BALL_COLORS_BGR.get(color, BALL_COLORS_BGR['gray']),
                            )
                elif cue_pocketed:
                    # Scratch recovery: show a pulsing double-ring so the player
                    # can clearly distinguish it from the static pre-game rings.
                    ctr = cm_to_disp(GOLF_MODE_STARTING_POSITIONS['white'])
                    if ctr is not None:
                        # Outer ring — fixed, bright white
                        cv2.circle(canvas, ctr, ring_r + 4,
                                   BALL_COLORS_BGR['white'], 3, cv2.LINE_AA)
                        # Inner pulsing ring — oscillates ±4 px, orange tint
                        pulse = int(4 * math.sin(self._render_tick * 0.25))
                        cv2.circle(canvas, ctr, max(2, ring_r - 4 + pulse),
                                   (0, 120, 255), 2, cv2.LINE_AA)
                        # Small cross-hair at centre
                        cx, cy = ctr
                        cv2.line(canvas, (cx - 6, cy), (cx + 6, cy),
                                 BALL_COLORS_BGR['white'], 1, cv2.LINE_AA)
                        cv2.line(canvas, (cx, cy - 6), (cx, cy + 6),
                                 BALL_COLORS_BGR['white'], 1, cv2.LINE_AA)

                    # 3c. FOUL banner — large centred overlay when cue is pocketed
                    fh, fw = canvas.shape[:2]
                    bar_y0 = int(fh * 0.12)
                    bar_h  = int(fh * 0.22)
                    overlay = canvas.copy()
                    cv2.rectangle(overlay, (0, bar_y0), (fw, bar_y0 + bar_h),
                                  (20, 20, 40), -1)
                    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
                    canvas = _draw_text_pil(
                        canvas,
                        lines=[
                            ('FOUL!',
                             'PlayfairDisplay-Black.ttf', 72,
                             (217, 95, 26)),
                            ('Cue ball pocketed (+3 penalty)',
                             'DMSans-SemiBold.ttf', 28,
                             (244, 239, 228)),
                            ('Place it back on the table',
                             'Lora-Italic.ttf', 22,
                             (196, 184, 152)),
                        ],
                        start_y=bar_y0 + int(bar_h * 0.10),
                        center_x=fw // 2,
                        line_gap=12,
                    )

                # 3d. Post-scratch cooldown countdown banner
                game_obj = getattr(self, 'game', None)
                cooldown = getattr(game_obj, '_scratch_return_cooldown', 0)
                if cooldown > 0:
                    import math as _math
                    seconds_left = _math.ceil(cooldown / 30)
                    msg = f'Game resumes in {seconds_left}s...'
                    fh, fw = canvas.shape[:2]
                    canvas = _draw_text_pil(
                        canvas,
                        lines=[(msg, 'DMSans-Medium.ttf', 30, (217, 95, 26))],
                        start_y=int(fh * 0.04),
                        center_x=fw // 2,
                    )

            # 4. Detected balls
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

        # ── Push to camera label ──────────────────────────────────────────────
        self.camera_label.setPixmap(
            _bgr_to_pixmap(canvas).scaled(
                self.camera_label.width(), self.camera_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

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
        self.setWindowTitle('OptiCue')
        self.setMinimumSize(500, 380)
        self.setStyleSheet(_APP_STYLE)

        # Window icon (silently skipped if missing)
        import os as _os
        from PyQt5.QtGui import QIcon as _QIcon
        _icon_path = _os.path.join(_os.path.dirname(__file__), 'assets', 'opticue_icon.png')
        if _os.path.exists(_icon_path):
            self.setWindowIcon(_QIcon(_icon_path))

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._setup_page = SetupPage()                                   # index 0
        self._setup_page.start_requested.connect(self._on_setup_done)
        self._stack.addWidget(self._setup_page)

        self._calib_page = CalibrationPage()                             # index 1
        self._calib_page.calibration_done.connect(self._on_calibration_done)
        self._stack.addWidget(self._calib_page)

        self._reg_page = PlayerRegistrationPage()                        # index 2
        self._reg_page.registration_done.connect(self._on_registration_done)
        self._stack.addWidget(self._reg_page)

        # ReferencePage is NOT added to the stack — reference is auto-captured
        # during calibration from self._calib_page._frame.

        self._stack.addWidget(self._build_main_ui())                     # index 3

        self._leaderboard_page = LeaderboardPage()                       # index 4
        self._leaderboard_page.play_again.connect(self._on_play_again)
        self._leaderboard_page.recalibrate_requested.connect(self._on_leaderboard_recalibrate)
        self._leaderboard_page.quit_requested.connect(self.close)
        self._stack.addWidget(self._leaderboard_page)

        self._stack.setCurrentIndex(0)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage('Configure hardware and click Start.')
        self.statusBar().setStyleSheet("""
            background-color: #ece6d8;
            color: #8a7d68;
            font-family: 'DM Sans', 'Segoe UI', sans-serif;
            font-size: 10px;
            border-top: 1px solid #c4b898;
            padding: 2px 8px;
        """)

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
            'background-color: #1a1a1a; border: 1px solid #c4b898; border-radius: 2px;'
        )
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_label.setText('Initialising camera…')
        self.camera_label.clicked.connect(self._on_camera_click)
        self.camera_label.right_clicked.connect(lambda lx, ly: self._on_reset())
        main_layout.addWidget(self.camera_label, stretch=4)

        # ── Right: sidebar ────────────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(260)
        sidebar.setStyleSheet('background-color: #ece6d8;')
        outer_sb_layout = QVBoxLayout(sidebar)
        outer_sb_layout.setContentsMargins(0, 0, 0, 0)
        outer_sb_layout.setSpacing(0)

        scroll_area = QScrollArea()
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet('background-color: #ece6d8;')

        scroll_content = QWidget()
        scroll_content.setStyleSheet('background-color: #ece6d8;')
        sb = QVBoxLayout(scroll_content)
        sb.setContentsMargins(10, 8, 10, 8)
        sb.setSpacing(4)

        # Sidebar title
        title = QLabel()
        title.setTextFormat(Qt.RichText)
        title.setText(
            '<span style="font-family:\'Playfair Display\',Georgia,serif;'
            ' font-weight:900; font-size:18px; color:#1c3a26;">Opti</span>'
            '<span style="font-family:\'Playfair Display\',Georgia,serif;'
            ' font-weight:700; font-style:italic; font-size:18px; color:#d95f1a;">Cue</span>'
        )
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet('padding: 10px 0 2px 0;')
        sb.addWidget(title)

        subtitle = QLabel('Golf Mode')
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("""
            font-family: 'Lora', Georgia, serif;
            font-style: italic;
            font-size: 11px;
            color: #8a7d68;
            padding-bottom: 6px;
        """)
        sb.addWidget(subtitle)
        sb.addWidget(self._make_divider())

        _HDR_STYLE = (
            "font-family: 'DM Sans', 'Segoe UI', sans-serif;"
            " font-size: 11px; letter-spacing: 2px; color: #4a3f32;"
            " font-weight: bold; padding-top: 6px;"
        )
        _SUB_STYLE = (
            "font-family: 'DM Sans', 'Segoe UI', sans-serif;"
            " font-size: 10px; letter-spacing: 1px; color: #6a5f52; padding-top: 2px;"
        )

        hdr = QLabel('GAME STATE')
        hdr.setStyleSheet(_HDR_STYLE)
        sb.addWidget(hdr)

        # Strokes + Remaining — side-by-side to save vertical space
        nums_row = QWidget()
        nr = QHBoxLayout(nums_row)
        nr.setContentsMargins(0, 0, 0, 0)
        nr.setSpacing(8)
        for attr, label_text in (
            ('lbl_stroke_num', 'STROKES'),
            ('lbl_remaining_num', 'REMAINING'),
        ):
            col = QWidget()
            cl = QVBoxLayout(col)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(2)
            hdr_lbl = QLabel(label_text)
            hdr_lbl.setStyleSheet(_SUB_STYLE)
            num_lbl = QLabel('0')
            num_lbl.setStyleSheet(
                "font-family: 'Playfair Display', Georgia, serif;"
                " font-size: 28px; font-weight: bold; color: #1c3a26;"
            )
            cl.addWidget(hdr_lbl)
            cl.addWidget(num_lbl)
            setattr(self, attr, num_lbl)
            nr.addWidget(col)
        sb.addWidget(nums_row)

        # Target — colored dot + name
        lbl_t_hdr = QLabel('TARGET')
        lbl_t_hdr.setStyleSheet(_SUB_STYLE)
        sb.addWidget(lbl_t_hdr)
        self.lbl_target = QLabel('—')
        self.lbl_target.setTextFormat(Qt.RichText)
        self.lbl_target.setStyleSheet('font-size: 14px; color: #c4b898;')
        sb.addWidget(self.lbl_target)

        # Ball rack — colored dots for remaining balls
        lbl_rack_hdr = QLabel('BALL RACK')
        lbl_rack_hdr.setStyleSheet(_SUB_STYLE)
        sb.addWidget(lbl_rack_hdr)
        self.lbl_ball_rack = QLabel('—')
        self.lbl_ball_rack.setTextFormat(Qt.RichText)
        self.lbl_ball_rack.setStyleSheet('font-size: 14px; color: #c4b898;')
        sb.addWidget(self.lbl_ball_rack)

        # Game message (error / status)
        self.lbl_game_msg = QLabel('')
        self.lbl_game_msg.setStyleSheet(
            "color: #d95f1a; font-size: 11px; font-weight: bold;"
            " font-family: 'DM Sans', 'Segoe UI', sans-serif;"
        )
        self.lbl_game_msg.setWordWrap(True)
        sb.addWidget(self.lbl_game_msg)

        sb.addWidget(self._make_divider())

        hdr2 = QLabel('STATUS')
        hdr2.setStyleSheet(_HDR_STYLE)
        sb.addWidget(hdr2)

        self.lbl_mode_calib  = self._make_status_label('Screen  \u00b7  Calib: none')
        self.lbl_game_status = self._make_status_label('Game: Setup')
        for lbl in (self.lbl_mode_calib, self.lbl_game_status):
            sb.addWidget(lbl)

        sb.addWidget(self._make_divider())

        ctrl_hdr = QLabel('CONTROLS')
        ctrl_hdr.setStyleSheet(_HDR_STYLE)
        sb.addWidget(ctrl_hdr)

        self.btn_start_game  = QPushButton('Start Game')
        self.btn_start_over  = QPushButton('Start Over')
        self.btn_capture     = QPushButton('Capture')
        self.btn_recalib     = QPushButton('Re-calibrate')
        self.btn_homography  = QPushButton('Disable Homography')

        self.btn_start_game.setEnabled(False)
        self.btn_start_game.setStyleSheet(_BTN_DISABLED_SS)
        self.btn_start_over.setEnabled(False)
        self.btn_start_over.setStyleSheet(_BTN_DISABLED_SS)
        self.btn_capture.setStyleSheet(_BTN_GHOST_SS)
        self.btn_recalib.setStyleSheet(_BTN_GHOST_SS)
        self.btn_homography.setStyleSheet(_BTN_GHOST_SS)

        self.btn_start_game.setToolTip('Validate 5 balls and start a new round')
        self.btn_start_over.setToolTip('Reset strokes and remaining balls to start a new round')
        self.btn_capture.setToolTip('Save current frame to disk (Space)')
        self.btn_recalib.setToolTip('Go back to calibration page')
        self.btn_homography.setToolTip('Toggle raw / top-down view')

        self.btn_start_game.clicked.connect(self._on_start_game)
        self.btn_start_over.clicked.connect(self._on_start_over)
        self.btn_capture.clicked.connect(self._on_capture)
        self.btn_recalib.clicked.connect(self._on_recalibrate)
        self.btn_homography.clicked.connect(self._on_toggle_homography)

        for btn in (self.btn_start_game, self.btn_start_over, self.btn_capture,
                    self.btn_recalib, self.btn_homography):
            sb.addWidget(btn)

        sb.addStretch()

        hint = QLabel(
            'Click a ◆ pocket to aim manually.\n'
            'Right-click to clear pocket.'
        )
        hint.setStyleSheet("""
            font-family: 'Lora', Georgia, serif;
            font-style: italic;
            font-size: 10px;
            color: #b0a48c;
            padding: 4px 0;
        """)
        hint.setAlignment(Qt.AlignCenter)
        hint.setWordWrap(True)
        sb.addWidget(hint)

        scroll_area.setWidget(scroll_content)
        outer_sb_layout.addWidget(scroll_area)

        main_layout.addWidget(sidebar, stretch=0)
        return widget

    def _make_status_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            "font-family: 'DM Sans', 'Segoe UI', sans-serif;"
            " font-size: 13px; color: #2a2520; padding: 2px 0;"
        )
        return lbl

    def _make_divider(self) -> QFrame:
        line = QFrame()
        line.setObjectName('divider')
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Plain)
        line.setStyleSheet('background-color: #c4b898; max-height: 1px; border: none;')
        return line

    # ─────────────────────────────────────────────────────────────────────────
    # Pipeline flow — setup → calibration → main
    # ─────────────────────────────────────────────────────────────────────────

    def _on_setup_done(self, camera_index: int, has_camera: bool):
        self.camera_index = camera_index

        if not has_camera:
            self._on_calibration_done(None)
            return

        self._stack.setCurrentIndex(1)   # → CalibrationPage
        self.setMinimumSize(800, 620)
        self.statusBar().showMessage('Opening camera\u2026')
        self._calib_page.start_loading_animation()

        self._cam_thread = CameraOpenThread(camera_index)
        self._cam_thread.opened.connect(self._on_camera_opened)
        self._cam_thread.start()

    def _on_camera_opened(self, cap):
        """Slot called from CameraOpenThread when camera open attempt completes."""
        self._calib_page.stop_loading_animation()
        if cap is None:
            print(f'[GUI] Cannot open camera {self.camera_index}.')
            self._cap = None
            self._on_calibration_done(None)
            return
        self._cap = cap
        self.statusBar().showMessage('Place 4 corners or Auto-detect, then Accept.')
        self._calib_page.start_camera(self._cap)

    def _on_registration_done(self, name: str):
        """Called when the player submits their name on the registration page."""
        self._current_player_name = name
        # Camera is already open and calibration already done — go straight to main.
        self._stack.setCurrentIndex(3)   # → Main view
        self.setMinimumSize(900, 680)
        self._start_camera()
        self._start_timer()
        calib_str = 'OK' if self.cam_H is not None else 'none'
        self.lbl_mode_calib.setText(f'Calib: {calib_str}')
        self.statusBar().showMessage('Ready — click a ball to select a target.')

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

        # Auto-capture reference frame from the calibration camera
        ref_frame = getattr(self._calib_page, '_frame', None)
        if ref_frame is not None:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(_REF_CACHE), ref_frame)
            try:
                from ball_detection import clear_ref_cache
                clear_ref_cache(str(_REF_CACHE))
            except ImportError:
                pass
            self.ref_path = str(_REF_CACHE)

            ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            self._run_dir = os.path.join(os.path.dirname(__file__), 'runs', ts)
            os.makedirs(self._run_dir, exist_ok=True)
            ref_run_path = os.path.join(self._run_dir, 'ref.jpeg')
            cv2.imwrite(ref_run_path, ref_frame)
            print(f'[GUI] Reference image saved to run: {ref_run_path}')
        else:
            self.ref_path = None
            self._run_dir = None
            print('[GUI] No reference frame available — ball detection may be degraded.')

        # Always route to PlayerRegistrationPage after calibration
        self._stack.setCurrentIndex(2)   # → PlayerRegistrationPage
        self.setMinimumSize(500, 380)
        self.statusBar().showMessage('Enter your name to begin.')

    def _on_recalibrate(self):
        if self._timer is not None:
            self._timer.stop()
        self.cam_H          = None
        self.ref_path       = None
        self._table_corners = None

        if self._cap is None or not self._cap.isOpened():
            print('[GUI] Camera unavailable during recalibration.')
            self._calib_page.accepted_corners = None
            self._on_calibration_done(None)
            return

        self._stack.setCurrentIndex(1)   # → CalibrationPage
        self.setMinimumSize(800, 620)
        self.statusBar().showMessage('Place 4 corners and click Accept.')
        self._calib_page.start_camera(self._cap)

    def _show_leaderboard(self):
        """Record current player's score and show the leaderboard page."""
        name = self._current_player_name
        if name:
            self.player_scores.append((name, self._last_stroke_count))
            self._current_player_name = ''

        ranked = sorted(self.player_scores, key=lambda x: x[1])
        self._leaderboard_page.update_scores(ranked)
        self._stack.setCurrentIndex(4)   # → LeaderboardPage
        self.statusBar().showMessage('Game complete! See the high scores.')

    def _on_play_again(self):
        """Return to player registration for the next player (same calibration)."""
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        # Reset game so stale GAME_OVER state doesn't immediately re-trigger leaderboard
        game = getattr(self, 'game', None)
        if game is not None:
            game.reset()
        self._last_game_state = 'WAITING_FOR_BALLS'
        self._stack.setCurrentIndex(2)   # → PlayerRegistrationPage
        self.statusBar().showMessage('Enter your name to begin.')

    def _on_leaderboard_recalibrate(self):
        """Re-calibrate from the leaderboard; after calibration always go to registration."""
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self.cam_H          = None
        self.ref_path       = None
        self._table_corners = None

        if self._cap is None or not self._cap.isOpened():
            self._on_calibration_done(None)
            return

        self._stack.setCurrentIndex(1)   # → CalibrationPage
        self.setMinimumSize(800, 620)
        self.statusBar().showMessage(
            'Re-calibrating — place 4 corners and click Accept.'
        )
        self._calib_page.start_camera(self._cap)

    # ─────────────────────────────────────────────────────────────────────────
    # Camera setup
    # ─────────────────────────────────────────────────────────────────────────

    def _start_camera(self):
        if self._cap is None or not self._cap.isOpened():
            print('[GUI] WARNING: Camera not available.')

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
    # Game-state overlays (game-over screen)
    # ─────────────────────────────────────────────────────────────────────────

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

        score_text = (
            f'Completed in {stroke_count} stroke{"s" if stroke_count != 1 else ""}!'
        )
        canvas[:] = _draw_text_pil(
            canvas,
            lines=[
                ('GAME OVER',  'PlayfairDisplay-Black.ttf', 88, (217, 95, 26)),
                (score_text,   'DMSans-Regular.ttf',        26, (244, 239, 228)),
            ],
            start_y=h // 2 - 80,
            center_x=w // 2,
            line_gap=24,
        )

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

    def _draw_placement_ring(self, canvas, center, radius, color_bgr, thickness=3):
        """Draw a hollow target ring indicating where a ball should be placed."""
        cv2.circle(canvas, center, radius, color_bgr, thickness, cv2.LINE_AA)

    def _draw_pocket(self, canvas, center, selected=False, radius=10):
        x, y = center
        r    = radius
        pts  = np.array([[x, y - r], [x + r, y], [x, y + r], [x - r, y]])
        color = COLOR_POCKET_SEL if selected else COLOR_POCKET
        cv2.fillPoly(canvas, [pts], color)
        cv2.polylines(canvas, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)
        if selected:
            cv2.circle(canvas, center, radius + 5, COLOR_POCKET_SEL, 2, cv2.LINE_AA)

    def _draw_trajectory(self, canvas, path_px, thickness=2, color=COLOR_PATH):
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
    # Coordinate transforms (camera ↔ table)
    # ─────────────────────────────────────────────────────────────────────────

    def _camera_to_table(self, pt) -> Optional[Tuple[float, float]]:
        if self.cam_H is None:
            return None
        arr = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
        res = cv2.perspectiveTransform(arr, self.cam_H)
        return (float(res[0][0][0]), float(res[0][0][1]))

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

    def _on_start_game(self):
        """Start a new game, or toggle pause/resume if a game is already running."""
        game = getattr(self, 'game', None)
        if game is None:
            self.lbl_game_msg.setText('Game not initialized yet.')
            return

        from game_tracker import ST_WAITING_FOR_BALLS, ST_GAME_OVER
        # If a game is active, toggle pause/resume
        if game.state not in (ST_WAITING_FOR_BALLS, ST_GAME_OVER):
            self.paused = not self.paused
            self._update_status_panel()
            return

        # Validate and start a new game
        balls = getattr(self, '_last_balls', None)
        if not balls:
            self.lbl_game_msg.setText('No balls detected. Place 5 balls on the table.')
            return

        n = len(balls)
        if n != 5:
            self.lbl_game_msg.setText(
                f'Found {n} ball(s) — need exactly 5.\n'
                'Place 1 white + 4 colored balls on the table.'
            )
            return

        colored = [b['color'] for b in balls if not b.get('is_cue')]
        if len(colored) != 4:
            self.lbl_game_msg.setText(
                f'Need 1 white + 4 colored balls (found {len(colored)} colored).'
            )
            return

        self.paused = False
        game.start_game(colored, balls)
        self.lbl_game_msg.setText('Game started!')
        self.lbl_stroke_num.setText('0')
        self.lbl_remaining_num.setText(str(len(colored)))
        self.statusBar().showMessage('Game started — make your first shot!')
        self._update_status_panel()

    def _on_start_over(self):
        """Reset game back to SETUP state so the user can press Start Game again."""
        game = getattr(self, 'game', None)
        if game is None:
            return
        game.reset()
        self.lbl_game_msg.setText('Game reset. Place 5 balls and press Start Game.')
        self.lbl_stroke_num.setText('0')
        self.lbl_remaining_num.setText('0')
        self.statusBar().showMessage('Game reset — place 5 balls on the table.')
        self._update_status_panel()

    def _on_capture(self):
        """Save the current raw camera frame to the current run directory."""
        frame = self.current_frame
        if frame is None:
            self.statusBar().showMessage('No frame available to save.')
            return
        if self._run_dir is None:
            ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            self._run_dir = os.path.join(os.path.dirname(__file__), 'runs', ts)
        os.makedirs(self._run_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(self._run_dir, f'capture_{ts}.jpeg')
        cv2.imwrite(path, frame)
        self.statusBar().showMessage(f'Frame saved: {path}')

    def _on_reset(self):
        """Clear the manual pocket selection. Ball selection is cleared by main.py."""
        self._pending_clicks.clear()
        self._selected_pocket_name = None
        self._selected_pocket_cm   = None
        self.statusBar().showMessage('Pocket selection cleared — using auto-selection.')

    def _on_toggle_homography(self):
        """Toggle between top-down (homography) and raw camera view."""
        self._raw_view = not self._raw_view
        self.btn_homography.setText(
            'Enable Homography' if self._raw_view else 'Disable Homography'
        )

    def _all_balls_in_position(self, balls: List[Dict]) -> bool:
        """True when every Golf Mode target has the correct ball within threshold."""
        for color, target_cm in GOLF_MODE_STARTING_POSITIONS.items():
            match = next(
                (b for b in balls
                 if b.get('color') == color and b.get('center_cm') is not None),
                None,
            )
            if match is None:
                return False
            dx = match['center_cm'][0] - target_cm[0]
            dy = match['center_cm'][1] - target_cm[1]
            if (dx * dx + dy * dy) ** 0.5 > BALL_IN_POSITION_THRESHOLD_CM:
                return False
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Sidebar status update
    # ─────────────────────────────────────────────────────────────────────────

    def _update_status_panel(self):
        calib_str = 'Calib: OK' if self.cam_H is not None else 'Calib: none'
        self.lbl_mode_calib.setText(calib_str)

        game_str = self._last_game_state.replace('_', ' ').title()
        self.lbl_game_status.setText(f'Game: {game_str}')

        # Build rack source first so REMAINING and BALL RACK stay in sync.
        detected_colors = [
            b['color']
            for b in getattr(self, '_last_balls', [])
            if not b.get('is_cue') and b.get('color')
        ]

        # Large number displays
        self.lbl_stroke_num.setText(str(self._last_stroke_count))
        self.lbl_remaining_num.setText(str(len(detected_colors)))

        # Target with colored dot
        if self._last_selected_color:
            hex_c = _BALL_HEX.get(self._last_selected_color, '#888888')
            self.lbl_target.setText(
                f'<span style="color:{hex_c}; font-size:16px;">●</span>'
                f' {self._last_selected_color.title()}'
            )
            self.lbl_target.setStyleSheet('font-size: 14px; color: #2a2520;')
        else:
            self.lbl_target.setText('—')
            self.lbl_target.setStyleSheet('font-size: 14px; color: #c4b898;')

        # Ball rack — cue ball (white) first, then remaining colored balls
        if self._last_cue_present:
            cue_dot = '<span style="color:#f0f0f0; font-size:18px;" title="Cue ball">\u25cf</span>'
        else:
            cue_dot = '<span style="color:#555555; font-size:18px;" title="Cue ball \u2014 not on table">\u25cb</span>'
        colored_dots = ' '.join(
            f'<span style="color:{_BALL_HEX.get(c, "#888888")}; font-size:18px;">\u25cf</span>'
            for c in detected_colors
        )
        rack_html = cue_dot + ('&nbsp;' + colored_dots if colored_dots else '')
        self.lbl_ball_rack.setText(rack_html)

        # ── Button state machine ──────────────────────────────────────────────
        from game_tracker import ST_WAITING_FOR_BALLS, ST_GAME_OVER
        game  = getattr(self, 'game', None)
        balls = getattr(self, '_last_balls', [])
        game_active = (game is not None and
                       game.state not in (ST_WAITING_FOR_BALLS, ST_GAME_OVER))

        # btn_start_game
        if game_active:
            if self.paused:
                self.btn_start_game.setText('Resume Game')
                self.btn_start_game.setStyleSheet(_BTN_PRIMARY_SS)
            else:
                self.btn_start_game.setText('Pause Game')
                self.btn_start_game.setStyleSheet(_BTN_ACCENT_SS)
            self.btn_start_game.setEnabled(True)
        else:
            self.btn_start_game.setText('Start Game')
            can_start = self._all_balls_in_position(balls)
            self.btn_start_game.setEnabled(can_start)
            self.btn_start_game.setStyleSheet(_BTN_PRIMARY_SS if can_start else _BTN_DISABLED_SS)

        # btn_start_over: ghost when game active, disabled otherwise
        if game_active:
            self.btn_start_over.setEnabled(True)
            self.btn_start_over.setStyleSheet(_BTN_GHOST_SS)
        else:
            self.btn_start_over.setEnabled(False)
            self.btn_start_over.setStyleSheet(_BTN_DISABLED_SS)

        if self._last_game_state == 'GAME_OVER':
            self.lbl_game_msg.setText(
                f'Game Over! All balls pocketed in {self._last_stroke_count} strokes.'
            )
            # Trigger leaderboard (only once, while we're still on the main page)
            if self._stack.currentIndex() == 3:
                self._show_leaderboard()
        elif self._last_game_state == 'WAITING_FOR_BALLS':
            self.lbl_game_msg.setText('')

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
        if self._timer is not None:
            self._timer.stop()
        # Single camera release point
        if self._cap is not None:
            self._cap.release()
            self._cap = None
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
                cv2.line(canvas, a, b, COLOR_PATH, thickness, cv2.LINE_AA)
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
            cv2.line(canvas, end_pt, (ax, ay), COLOR_PATH,
                     thickness + 1, cv2.LINE_AA)


def _bgr_to_pixmap(frame: np.ndarray) -> QPixmap:
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ── PIL font helpers ───────────────────────────────────────────────────────────

_PIL_FONTS: dict = {}   # (filename, size) → PIL ImageFont


def _get_pil_font(filename: str, size: int):
    """Return a cached PIL ImageFont; falls back to PIL default on failure."""
    if not PIL_AVAILABLE:
        return None
    key = (filename, size)
    if key not in _PIL_FONTS:
        try:
            _PIL_FONTS[key] = _PILImageFont.truetype(
                str(config.FONTS_DIR / filename), size
            )
        except Exception:
            _PIL_FONTS[key] = _PILImageFont.load_default()
    return _PIL_FONTS[key]


def _draw_text_pil(
    canvas: np.ndarray,
    lines: list,
    start_y: int,
    center_x: int = None,
    left_x: int = None,
    line_gap: int = 10,
) -> np.ndarray:
    """
    Draw branded text lines onto a BGR numpy frame using PIL/Pillow.

    lines  — list of (text, font_filename, font_size_px, rgb_color)
    center_x — if given, each line is horizontally centred at this x
    left_x   — if given, each line starts at this x (overrides center_x)
    Returns the modified canvas (BGR numpy array).
    """
    if not PIL_AVAILABLE:
        return canvas
    img  = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    y = start_y
    for text, font_file, size, rgb in lines:
        font = _get_pil_font(font_file, size)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw   = bbox[2] - bbox[0]
        th   = bbox[3] - bbox[1]
        x    = left_x if left_x is not None else (center_x - tw // 2)
        draw.text((x, y), text, font=font, fill=rgb)
        y += th + line_gap
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


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
