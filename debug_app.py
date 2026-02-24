"""debug_app.py — Standalone debug window for testing the billiards game state
machine with a fully playable physics simulation. No camera or calibration needed.

Run with:
    python debug_app.py

Controls
--------
WAITING / GAME_OVER state
  1–5         Select ball colour  (1=white  2=yellow  3=blue  4=bordeaux  5=purple)
  Left-click  Place ball of selected colour, or grab an existing ball to drag
  Drag        Move a grabbed ball to a new position
  Right-click Remove the ball under the cursor
  Space       Start the game  (requires exactly 5 balls: 1 white + 4 coloured)
  R           Reset

TRACKING state (game in progress)
  Left-click  Shoot cue ball toward the click point
  Space       Shoot along the suggested trajectory
  F           Instant force-shot (no physics — quick state-machine test)
  R           Reset

TRACKING state — after a SCRATCH (cue ball pocketed)
  Left-click  Re-place the white ball at the click point
"""

import math
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget,
)

from config import BALL_RADIUS_CM
from game_tracker import (
    CUE_BALL_PENALTY,
    ST_DISTURBED,
    ST_GAME_OVER,
    ST_TRACKING,
    ST_WAITING_FOR_BALLS,
    TABLE_HEIGHT_CM,
    TABLE_WIDTH_CM,
    POCKET_POSITIONS_CM,
    GameTracker,
)
from trajectory import suggest_best_shot

# ── Display constants ──────────────────────────────────────────────────────────
SCALE: int = 8                               # display pixels per cm
CANVAS_W = int(TABLE_WIDTH_CM  * SCALE)      # 976
CANVAS_H = int(TABLE_HEIGHT_CM * SCALE)      # 488

BALL_R_PX    = 16    # ball radius in display pixels
POCKET_R_PX  = 14    # pocket radius in display pixels
DRAG_RADIUS_CM = 3.0  # grab radius when clicking near an existing ball

# ── Ball colours in BGR (OpenCV) ───────────────────────────────────────────────
BALL_BGR: Dict[str, Tuple[int, int, int]] = {
    "white":    (240, 240, 240),
    "yellow":   (  0, 200, 255),
    "blue":     (190,  60,  10),
    "bordeaux": ( 25,  20, 155),
    "purple":   (170,  40, 170),
}
COLOR_ORDER = ["white", "yellow", "blue", "bordeaux", "purple"]
COLOR_KEYS  = {
    Qt.Key_1: "white",
    Qt.Key_2: "yellow",
    Qt.Key_3: "blue",
    Qt.Key_4: "bordeaux",
    Qt.Key_5: "purple",
}

# Rendering colours (BGR)
COLOR_CUE    = (255, 220,  30)   # cue path
COLOR_TARGET = ( 30, 140, 255)   # target path
TABLE_GREEN  = ( 34, 100,  34)
COLOR_HUD_BG = (  0,   0,   0)
COLOR_HUD_FG = (255, 255, 255)

# ── Physics constants ──────────────────────────────────────────────────────────
SIM_FRICTION         = 0.96    # velocity multiplier per tick (~3.5 s stop time)
SIM_RESTITUTION      = 0.90    # ball-ball coefficient of restitution
SIM_WALL_RESTITUTION = 0.80    # cushion coefficient of restitution
SIM_MIN_SPEED        = 0.05    # cm/tick — "stopped" threshold
SIM_MAX_TICKS        = 600     # safety cutoff (~20 s at 30 fps)
SIM_POCKET_RADIUS    = 3.5     # cm — pocket capture radius
SHOT_POWER           = 7.0     # cm/tick — default cue ball launch speed
SIM_SUB_STEPS        = 4       # sub-steps per tick — prevents tunneling at high speed


# ── Clickable label ────────────────────────────────────────────────────────────
class _ClickableLabel(QLabel):
    """QLabel that emits press / drag / release / right-click signals."""

    pressed       = pyqtSignal(int, int)
    dragged       = pyqtSignal(int, int)
    released      = pyqtSignal(int, int)
    right_clicked = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._dragging = False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self.pressed.emit(event.x(), event.y())
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


# ── Physics simulation ─────────────────────────────────────────────────────────
class _PhysicsSim:
    """Euler-integration billiards physics.

    Simulates ball motion, elastic ball-ball collisions, cushion reflections,
    and pocket detection until all balls stop or MAX_TICKS is reached.
    """

    def __init__(
        self,
        balls:   List[Dict],
        cue_dir: Tuple[float, float],
    ):
        self._tick_count = 0
        self.bodies: List[Dict] = []
        for b in balls:
            cm = b.get("center_cm", (0.0, 0.0))
            vel = [0.0, 0.0]
            if b.get("is_cue"):
                vel = [cue_dir[0] * SHOT_POWER, cue_dir[1] * SHOT_POWER]
            self.bodies.append(
                {
                    "pos":      [float(cm[0]), float(cm[1])],
                    "vel":      vel,
                    "color":    b["color"],
                    "is_cue":   b.get("is_cue", False),
                    "pocketed": False,
                }
            )

    # ── Main step ─────────────────────────────────────────────────────────────

    def step(self) -> bool:
        """Advance one physics tick.  Returns True while any ball is still moving."""
        self._tick_count += 1
        if self._tick_count >= SIM_MAX_TICKS:
            return False

        # Sub-stepping: split each tick into SIM_SUB_STEPS smaller steps so that
        # fast balls (SHOT_POWER > ball diameter) cannot tunnel through each other.
        for _ in range(SIM_SUB_STEPS):
            sub_active = [b for b in self.bodies if not b["pocketed"]]

            # 1. Integrate positions by vel / SIM_SUB_STEPS
            for b in sub_active:
                b["pos"][0] += b["vel"][0] / SIM_SUB_STEPS
                b["pos"][1] += b["vel"][1] / SIM_SUB_STEPS

            # 2. Cushion reflections
            r = BALL_RADIUS_CM
            for b in sub_active:
                if b["pos"][0] < r:
                    b["pos"][0] = r
                    b["vel"][0] = abs(b["vel"][0]) * SIM_WALL_RESTITUTION
                elif b["pos"][0] > TABLE_WIDTH_CM - r:
                    b["pos"][0] = TABLE_WIDTH_CM - r
                    b["vel"][0] = -abs(b["vel"][0]) * SIM_WALL_RESTITUTION
                if b["pos"][1] < r:
                    b["pos"][1] = r
                    b["vel"][1] = abs(b["vel"][1]) * SIM_WALL_RESTITUTION
                elif b["pos"][1] > TABLE_HEIGHT_CM - r:
                    b["pos"][1] = TABLE_HEIGHT_CM - r
                    b["vel"][1] = -abs(b["vel"][1]) * SIM_WALL_RESTITUTION

            # 3. Ball-ball elastic collisions (O(n²))
            for i in range(len(sub_active)):
                for j in range(i + 1, len(sub_active)):
                    self._resolve_collision(sub_active[i], sub_active[j])

            # 4. Pocket detection
            for b in sub_active:
                for px, py in POCKET_POSITIONS_CM:
                    dx = b["pos"][0] - px
                    dy = b["pos"][1] - py
                    if dx * dx + dy * dy < SIM_POCKET_RADIUS ** 2:
                        b["pocketed"] = True
                        b["vel"] = [0.0, 0.0]
                        break

        # 5. Friction — once per full tick to preserve deceleration rate
        for b in self.bodies:
            if not b["pocketed"]:
                b["vel"][0] *= SIM_FRICTION
                b["vel"][1] *= SIM_FRICTION

        return any(
            b["vel"][0] ** 2 + b["vel"][1] ** 2 > SIM_MIN_SPEED ** 2
            for b in self.bodies
            if not b["pocketed"]
        )

    # ── Collision helper ──────────────────────────────────────────────────────

    def _resolve_collision(self, b1: Dict, b2: Dict):
        dx = b2["pos"][0] - b1["pos"][0]
        dy = b2["pos"][1] - b1["pos"][1]
        dist_sq = dx * dx + dy * dy
        min_dist = 2.0 * BALL_RADIUS_CM
        if dist_sq >= min_dist * min_dist or dist_sq == 0.0:
            return
        dist = math.sqrt(dist_sq)
        nx, ny = dx / dist, dy / dist
        # Relative velocity along collision normal
        rvx = b1["vel"][0] - b2["vel"][0]
        rvy = b1["vel"][1] - b2["vel"][1]
        dot = rvx * nx + rvy * ny
        if dot <= 0.0:
            return  # already separating
        impulse = dot * (1.0 + SIM_RESTITUTION) / 2.0
        b1["vel"][0] -= impulse * nx
        b1["vel"][1] -= impulse * ny
        b2["vel"][0] += impulse * nx
        b2["vel"][1] += impulse * ny
        # Positional correction to prevent sustained overlap
        overlap = min_dist - dist
        b1["pos"][0] -= overlap / 2.0 * nx
        b1["pos"][1] -= overlap / 2.0 * ny
        b2["pos"][0] += overlap / 2.0 * nx
        b2["pos"][1] += overlap / 2.0 * ny

    # ── Result accessors ──────────────────────────────────────────────────────

    def get_active_balls(self) -> List[Dict]:
        """Ball dicts for all non-pocketed bodies (same format as _fake_balls)."""
        return [
            {
                "center_cm": (b["pos"][0], b["pos"][1]),
                "color":     b["color"],
                "is_cue":    b["is_cue"],
                "center":    (0, 0),
                "radius":    18,
            }
            for b in self.bodies
            if not b["pocketed"]
        ]

    def get_pocketed_colors(self) -> List[str]:
        return [b["color"] for b in self.bodies if b["pocketed"]]


# ── Main debug window ──────────────────────────────────────────────────────────
class DebugWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Billiards Debug Mode")

        # ── Game state ────────────────────────────────────────────────────────
        self.gt = GameTracker(confirm_frames=1)   # fast confirmation for debug
        self._fake_balls: List[Dict] = []
        self._selected_color: str = "white"
        self._drag_idx: Optional[int] = None

        # Physics simulation (None when idle)
        self._sim: Optional[_PhysicsSim] = None

        # Cached trajectory from last suggest_best_shot (used by Space key)
        self._cue_path_cache:    List[Tuple[float, float]] = []
        self._target_path_cache: List[Tuple[float, float]] = []

        # Static green frame — never changes so _check_motion() always returns
        # False, meaning the DISTURBED state is never auto-triggered.
        self._static_frame = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        self._static_frame[:] = TABLE_GREEN

        # ── UI layout ─────────────────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label = _ClickableLabel()
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setMinimumSize(CANVAS_W // 2, CANVAS_H // 2)
        layout.addWidget(self._label)

        self._status = self.statusBar()
        self._set_hint()

        # ── Signal connections ─────────────────────────────────────────────────
        self._label.pressed.connect(self._on_press)
        self._label.dragged.connect(self._on_drag)
        self._label.released.connect(self._on_release)
        self._label.right_clicked.connect(self._on_right_click)

        # ── Timer ─────────────────────────────────────────────────────────────
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)

    # ── Status helper ─────────────────────────────────────────────────────────

    def _set_hint(self):
        state = self.gt.state
        if self._sim is not None:
            msg = "Simulating physics..."
        elif state in (ST_WAITING_FOR_BALLS, ST_GAME_OVER):
            msg = (
                "1-5: colour  |  Click: place  |  Drag: move  |  RC: remove  "
                "|  Space: start game  |  R: reset"
            )
        elif state == ST_TRACKING and self.gt.cue_ball_pocketed:
            msg = "SCRATCH — click anywhere to re-place white ball"
        else:
            msg = (
                "Click: shoot toward point  |  Space: fire suggested shot  "
                "|  F: instant shot (no physics)  |  R: reset"
            )
        self._status.showMessage(msg)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _tick(self):
        if self._sim is not None:
            still_moving = self._sim.step()
            self._fake_balls = self._sim.get_active_balls()
            if not still_moving:
                self._on_sim_end()
        else:
            # Update game state machine (state only; we don't use GT's returned balls)
            self.gt.update(self._static_frame, list(self._fake_balls))

            # Trajectory computation uses fake_balls directly (no GT filtering delay)
            self._cue_path_cache    = []
            self._target_path_cache = []
            display_balls = list(self._fake_balls)

            if (
                self.gt.state in (ST_TRACKING, ST_DISTURBED)
                and len(display_balls) >= 2
            ):
                cue = next(
                    (b for b in display_balls if b.get("is_cue")), None
                )
                if cue and cue.get("center_cm") is not None:
                    remaining = [
                        b for b in display_balls
                        if not b.get("is_cue")
                        and b.get("center_cm") is not None
                        and b.get("color") in self.gt.remaining_balls
                    ]
                    if remaining:
                        best, cue_path, target_path = suggest_best_shot(
                            cue["center_cm"], remaining, all_balls=display_balls
                        )
                        self._cue_path_cache    = cue_path
                        self._target_path_cache = target_path
                        if best is not None:
                            self.gt.select_target(best["color"])
                        else:
                            self.gt.selected_target_color = None

        canvas = self._render(
            list(self._fake_balls),
            [] if self._sim is not None else self._cue_path_cache,
            [] if self._sim is not None else self._target_path_cache,
        )
        self._push(canvas)

    # ── Simulation end ────────────────────────────────────────────────────────

    def _on_sim_end(self):
        pocketed_colors = self._sim.get_pocketed_colors()
        self._fake_balls = self._sim.get_active_balls()
        self._sim = None
        self.gt.state = ST_TRACKING    # mirrors real GT: calm restored → TRACKING

        # Sync GameTracker's confirmed state to reflect the physics result
        self.gt._confirmed_balls = list(self._fake_balls)
        # Clear stale hidden-frame counters so GT doesn't misread pocketed balls
        # as suddenly-absent on the first update() call after the shot.
        self.gt._ball_hidden_frames = {}

        cue_pocketed = "white" in pocketed_colors

        if cue_pocketed:
            # Scratch: apply shot stroke + penalty; handle any co-pocketed colored balls
            self.gt.stroke_count += 1 + CUE_BALL_PENALTY
            self.gt.cue_ball_pocketed = True
            current_colors = {b["color"] for b in self._fake_balls}
            for color in list(self.gt.remaining_balls):
                if color not in current_colors:
                    self.gt.remaining_balls.remove(color)
                    self.gt.pocketed_balls.append(color)
            if not self.gt.remaining_balls:
                self.gt.state = ST_GAME_OVER
            self._status.showMessage(
                f"SCRATCH!  +{CUE_BALL_PENALTY} penalty strokes.  "
                "Click anywhere to re-place the white ball."
            )
        else:
            # Normal shot: let force_shot() count the stroke and handle pockets
            self.gt.force_shot()
            self._status.showMessage(
                f"Strokes: {self.gt.stroke_count}   "
                f"Remaining: {self.gt.remaining_balls or 'none — GAME OVER!'}"
            )

    # ── Shooting helpers ──────────────────────────────────────────────────────

    def _start_sim(self, dir_x: float, dir_y: float):
        """Launch the cue ball in direction (dir_x, dir_y)."""
        cue = next((b for b in self._fake_balls if b.get("is_cue")), None)
        if cue is None:
            self._status.showMessage("No white ball — place one first.")
            return
        self._sim = _PhysicsSim(list(self._fake_balls), (dir_x, dir_y))
        self.gt.state = ST_DISTURBED   # mirrors real GT: motion detected → DISTURBED
        self._cue_path_cache    = []
        self._target_path_cache = []

    def _fire_toward(self, target_cm: Tuple[float, float]):
        """Shoot cue ball toward target_cm."""
        cue = next((b for b in self._fake_balls if b.get("is_cue")), None)
        if cue is None:
            self._status.showMessage("No white ball — place one first.")
            return
        cx, cy = cue["center_cm"]
        dx = target_cm[0] - cx
        dy = target_cm[1] - cy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 0.5:
            return  # click on/very near cue ball — ignore
        self._start_sim(dx / dist, dy / dist)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render(
        self,
        balls:       List[Dict],
        cue_path:    List[Tuple[float, float]],
        target_path: List[Tuple[float, float]],
    ) -> np.ndarray:
        canvas = self._static_frame.copy()

        # Table border
        cv2.rectangle(
            canvas, (0, 0), (CANVAS_W - 1, CANVAS_H - 1), (200, 200, 200), 3
        )

        # Pockets
        for px, py in POCKET_POSITIONS_CM:
            cx, cy = int(px * SCALE), int(py * SCALE)
            cv2.circle(canvas, (cx, cy), POCKET_R_PX, (70, 70, 70), -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), POCKET_R_PX, (130, 130, 130), 1, cv2.LINE_AA)

        # Target path (orange)
        if len(target_path) >= 2:
            pts = [(int(x * SCALE), int(y * SCALE)) for x, y in target_path]
            for i in range(len(pts) - 1):
                cv2.line(canvas, pts[i], pts[i + 1], COLOR_TARGET, 2, cv2.LINE_AA)

        # Cue path (cyan) + backward aiming extension + ghost ball
        if len(cue_path) >= 2:
            pts = [(int(x * SCALE), int(y * SCALE)) for x, y in cue_path]
            dx = pts[0][0] - pts[1][0]
            dy = pts[0][1] - pts[1][1]
            ext = (int(pts[0][0] + dx * 3), int(pts[0][1] + dy * 3))
            cv2.line(canvas, ext, pts[-1], COLOR_CUE, 2, cv2.LINE_AA)
            cv2.circle(canvas, pts[-1], BALL_R_PX, COLOR_CUE, 1, cv2.LINE_AA)

        # Balls
        for ball in balls:
            cm = ball.get("center_cm")
            if cm is None:
                continue
            cx, cy = int(cm[0] * SCALE), int(cm[1] * SCALE)
            color_name = ball.get("color", "white")
            bgr = BALL_BGR.get(color_name, (200, 200, 200))

            cv2.circle(canvas, (cx, cy), BALL_R_PX, bgr, -1, cv2.LINE_AA)
            border = (80, 80, 80) if color_name == "white" else (220, 220, 220)
            cv2.circle(canvas, (cx, cy), BALL_R_PX, border, 1, cv2.LINE_AA)

            if color_name == self.gt.selected_target_color:
                cv2.circle(
                    canvas, (cx, cy), BALL_R_PX + 4, (0, 255, 255), 2, cv2.LINE_AA
                )

        # HUD — top-left: state / strokes / remaining
        state = self.gt.state
        strokes = self.gt.stroke_count
        remaining_str = (
            ", ".join(self.gt.remaining_balls) if self.gt.remaining_balls else "-"
        )
        state_label = {
            ST_WAITING_FOR_BALLS: "WAITING",
            ST_TRACKING:          "TRACKING",
            ST_DISTURBED:         "DISTURBED",
            ST_GAME_OVER:         "GAME OVER",
        }.get(state, state)
        if self._sim is not None:
            state_label = "SIMULATING..."

        hud_lines = [
            f"State:   {state_label}",
            f"Strokes: {strokes}",
            f"Remain:  {remaining_str}",
        ]
        for i, line in enumerate(hud_lines):
            y = 22 + i * 22
            cv2.putText(
                canvas, line, (8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_HUD_BG, 3, cv2.LINE_AA,
            )
            cv2.putText(
                canvas, line, (8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_HUD_FG, 1, cv2.LINE_AA,
            )

        # HUD — top-right: colour palette (visible only in placement mode)
        if state in (ST_WAITING_FOR_BALLS, ST_GAME_OVER):
            pal_x = CANVAS_W - 28
            for i, col in enumerate(COLOR_ORDER):
                cy_p = 18 + i * 36
                bgr_p = BALL_BGR[col]
                cv2.circle(canvas, (pal_x, cy_p), 13, bgr_p, -1, cv2.LINE_AA)
                border_p = (80, 80, 80) if col == "white" else (200, 200, 200)
                thickness = 3 if col == self._selected_color else 1
                cv2.circle(
                    canvas, (pal_x, cy_p), 13, border_p, thickness, cv2.LINE_AA
                )
                cv2.putText(
                    canvas, str(i + 1), (pal_x - 7, cy_p + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_HUD_BG, 2, cv2.LINE_AA,
                )
                cv2.putText(
                    canvas, str(i + 1), (pal_x - 7, cy_p + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_HUD_FG, 1, cv2.LINE_AA,
                )
            # Ball count indicator
            count_text = f"{len(self._fake_balls)}/5"
            cv2.putText(
                canvas, count_text, (pal_x - 14, 18 + 5 * 36 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HUD_BG, 3, cv2.LINE_AA,
            )
            cv2.putText(
                canvas, count_text, (pal_x - 14, 18 + 5 * 36 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HUD_FG, 1, cv2.LINE_AA,
            )

        # Game-over overlay
        if state == ST_GAME_OVER:
            overlay = canvas.copy()
            x0, y0 = CANVAS_W // 4, CANVAS_H // 3
            x1, y1 = 3 * CANVAS_W // 4, 2 * CANVAS_H // 3
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.8, canvas, 0.2, 0, canvas)
            text = f"GAME OVER  -  {strokes} strokes"
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)
            tx = (CANVAS_W - tw) // 2
            cv2.putText(
                canvas, text, (tx, (y0 + y1) // 2 + 8),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (80, 255, 255), 2, cv2.LINE_AA,
            )

        return canvas

    def _push(self, canvas: np.ndarray):
        h, w = canvas.shape[:2]
        rgb  = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg)
        self._label.setPixmap(
            pix.scaled(
                self._label.width(), self._label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
        )

    # ── Coordinate conversion ─────────────────────────────────────────────────

    def _label_to_cm(self, lx: int, ly: int) -> Optional[Tuple[float, float]]:
        """Convert a label pixel coordinate to table centimetres."""
        lw = self._label.width()
        lh = self._label.height()
        if lw == 0 or lh == 0:
            return None
        scale  = min(lw / CANVAS_W, lh / CANVAS_H)
        off_x  = (lw - CANVAS_W * scale) / 2
        off_y  = (lh - CANVAS_H * scale) / 2
        x_cm   = (lx - off_x) / scale / SCALE
        y_cm   = (ly - off_y) / scale / SCALE
        x_cm   = max(0.0, min(float(TABLE_WIDTH_CM),  x_cm))
        y_cm   = max(0.0, min(float(TABLE_HEIGHT_CM), y_cm))
        return (x_cm, y_cm)

    def _find_ball_near(self, cm: Tuple[float, float]) -> Optional[int]:
        """Return index of fake ball within DRAG_RADIUS_CM, or None."""
        best_idx, best_d2 = None, DRAG_RADIUS_CM ** 2
        for i, b in enumerate(self._fake_balls):
            bx, by = b["center_cm"]
            d2 = (bx - cm[0]) ** 2 + (by - cm[1]) ** 2
            if d2 < best_d2:
                best_d2, best_idx = d2, i
        return best_idx

    # ── Mouse handlers ────────────────────────────────────────────────────────

    def _on_press(self, lx: int, ly: int):
        if self._sim is not None:
            return  # ignore all input during simulation

        cm = self._label_to_cm(lx, ly)
        if cm is None:
            return

        state = self.gt.state

        if state in (ST_WAITING_FOR_BALLS, ST_GAME_OVER):
            # Placement mode: place or grab-to-drag
            idx = self._find_ball_near(cm)
            if idx is not None:
                self._drag_idx = idx
            else:
                self._place_ball(cm)

        elif state == ST_TRACKING:
            if self.gt.cue_ball_pocketed:
                # Scratch recovery: re-place white ball
                self._fake_balls = [b for b in self._fake_balls if not b["is_cue"]]
                self._fake_balls.append(
                    {
                        "center_cm": cm,
                        "color":     "white",
                        "is_cue":    True,
                        "center":    (0, 0),
                        "radius":    18,
                    }
                )
                self.gt.cue_ball_pocketed = False
                self.gt._confirmed_balls = list(self._fake_balls)
                self._status.showMessage("White ball placed — click to shoot.")
            else:
                # Fire cue ball toward clicked point
                self._fire_toward(cm)

    def _on_drag(self, lx: int, ly: int):
        if self._drag_idx is None or self._sim is not None:
            return
        cm = self._label_to_cm(lx, ly)
        if cm is not None:
            self._fake_balls[self._drag_idx]["center_cm"] = cm

    def _on_release(self, _lx: int, _ly: int):
        self._drag_idx = None

    def _on_right_click(self, lx: int, ly: int):
        if self._sim is not None:
            return
        if self.gt.state not in (ST_WAITING_FOR_BALLS, ST_GAME_OVER):
            return  # no removal during active play
        cm = self._label_to_cm(lx, ly)
        if cm is None:
            return
        idx = self._find_ball_near(cm)
        if idx is not None:
            removed = self._fake_balls.pop(idx)
            self._status.showMessage(f"Removed {removed['color']} ball.")

    # ── Ball placement ────────────────────────────────────────────────────────

    def _place_ball(self, center_cm: Tuple[float, float]):
        color = self._selected_color
        if color == "white":
            # Only one cue ball — replace any existing white
            self._fake_balls = [b for b in self._fake_balls if not b["is_cue"]]
        else:
            colored = [b for b in self._fake_balls if not b["is_cue"]]
            if len(colored) >= 4:
                self._status.showMessage(
                    "Maximum 4 coloured balls already placed. "
                    "Right-click to remove one first."
                )
                return
        self._fake_balls.append(
            {
                "center_cm": center_cm,
                "color":     color,
                "is_cue":    color == "white",
                "center":    (0, 0),
                "radius":    18,
            }
        )

    # ── Game control ──────────────────────────────────────────────────────────

    def _start_game(self):
        if self.gt.state not in (ST_WAITING_FOR_BALLS, ST_GAME_OVER):
            self._status.showMessage("Game already running. Press R to reset first.")
            return

        n = len(self._fake_balls)
        if n != 5:
            self._status.showMessage(
                f"Need exactly 5 balls (1 white + 4 coloured); currently {n}."
            )
            return

        cue_balls     = [b for b in self._fake_balls if b["is_cue"]]
        colored_balls = [b for b in self._fake_balls if not b["is_cue"]]
        if len(cue_balls) != 1 or len(colored_balls) != 4:
            self._status.showMessage("Need exactly 1 white ball + 4 coloured balls.")
            return

        colored_colors = [b["color"] for b in colored_balls]
        self.gt.start_game(colored_colors, list(self._fake_balls))
        self._set_hint()

    def _reset(self):
        self.gt.reset()
        self._fake_balls.clear()
        self._drag_idx = None
        self._sim = None
        self._cue_path_cache    = []
        self._target_path_cache = []
        self._set_hint()

    # ── Keyboard ──────────────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        key = event.key()

        if self._sim is not None:
            # Block all input while physics is running (R resets immediately)
            if key == Qt.Key_R:
                self._reset()
            else:
                super().keyPressEvent(event)
            return

        state = self.gt.state

        if key in COLOR_KEYS:
            if state in (ST_WAITING_FOR_BALLS, ST_GAME_OVER):
                self._selected_color = COLOR_KEYS[key]
                self._status.showMessage(f"Selected colour: {self._selected_color}")

        elif key == Qt.Key_Space:
            if state in (ST_WAITING_FOR_BALLS, ST_GAME_OVER):
                self._start_game()
            elif state == ST_TRACKING and not self.gt.cue_ball_pocketed:
                # Fire along the suggested trajectory
                if len(self._cue_path_cache) >= 2:
                    p0, p1 = self._cue_path_cache[0], self._cue_path_cache[1]
                    dx = p1[0] - p0[0]
                    dy = p1[1] - p0[1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist > 0:
                        self._start_sim(dx / dist, dy / dist)
                else:
                    self._status.showMessage(
                        "No trajectory suggestion available — click to aim manually."
                    )

        elif key == Qt.Key_F:
            if state == ST_TRACKING:
                self.gt.force_shot()
                self._status.showMessage(
                    f"Force-shot — strokes: {self.gt.stroke_count}  "
                    f"remaining: {self.gt.remaining_balls}"
                )
            else:
                self._status.showMessage(
                    f"force_shot ignored in state: {state}"
                )

        elif key == Qt.Key_R:
            self._reset()

        elif key in (Qt.Key_Escape, Qt.Key_Q):
            self.close()

        else:
            super().keyPressEvent(event)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    win = DebugWindow()
    win.resize(CANVAS_W, CANVAS_H + 30)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
