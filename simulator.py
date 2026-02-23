"""
Billiards Table Simulator
=========================
Standalone tool — no project imports required.

Pick a table photo, click the 4 corners manually, and get:

  Output 1 — Warped top-down image (976×488 px)
             Exactly what the ball-detection step receives as input.

  Output 2 — Corner coordinates  (4×2 float32 numpy array, [TL,TR,BR,BL])
             Same format as get_table_corners() in scene_understanding.py.
             Shown as a semi-transparent polygon overlay on the original image.

Controls (corner-selection window)
-----------------------------------
  Left-click          Add corner (up to 4), or drag an existing one
  R                   Reset all corners
  D / Backspace       Remove last corner
  Space               Compute & display results (needs 4 corners)
  Esc / Q             Quit

Controls (result windows)
--------------------------
  S                   Save warped PNG + corners NPY beside the source file
  N                   Close results, pick a new image
  Esc / Q             Quit
"""

import os
import sys
import numpy as np
import cv2

# ── Optional tkinter file dialog (falls back to sys.argv if unavailable) ──────
try:
    import tkinter as tk
    from tkinter import filedialog
    _HAS_TK = True
except ImportError:
    _HAS_TK = False

# ─────────────────────────────────────────────────────────────────────────────
# Constants  (mirrors config.py + gui/app.py)
# ─────────────────────────────────────────────────────────────────────────────

TABLE_WIDTH_CM      = 122.0
TABLE_HEIGHT_CM     =  61.0
TABLE_DISPLAY_SCALE = 8            # pixels per cm  (same as GUI)

OUT_W = int(TABLE_WIDTH_CM  * TABLE_DISPLAY_SCALE)   # 976
OUT_H = int(TABLE_HEIGHT_CM * TABLE_DISPLAY_SCALE)   # 488

_DRAG_THRESHOLD  = 15   # px — click within this radius to start dragging
_SCREEN_FRACTION = 0.88 # max fraction of screen size for the display window

# Corner labels shown in the selection window
_CORNER_LABELS = ['1-TL', '2-TR', '3-BR', '4-BL']

# ─────────────────────────────────────────────────────────────────────────────
# File-picker helper
# ─────────────────────────────────────────────────────────────────────────────

def _pick_image() -> str:
    """Open a file dialog and return the chosen path, or '' if cancelled."""
    if _HAS_TK:
        root = tk.Tk()
        root.withdraw()
        root.lift()
        path = filedialog.askopenfilename(
            title='Select a table image',
            filetypes=[
                ('Image files', '*.jpg *.jpeg *.png *.bmp *.tiff *.tif'),
                ('All files',   '*.*'),
            ],
        )
        root.destroy()
        return path or ''

    # Fallback: use first command-line argument
    if len(sys.argv) > 1:
        return sys.argv[1]
    print('[Simulator] No tkinter available. Pass the image path as a command-line argument.')
    return ''


# ─────────────────────────────────────────────────────────────────────────────
# Mouse-callback state
# ─────────────────────────────────────────────────────────────────────────────

class _CornerPicker:
    """Maintains corner state and handles all mouse events for window 1."""

    def __init__(self, display_scale: float):
        self.display_scale = display_scale  # original → display pixel ratio
        self.points        = []   # list of (x, y) in *original* image coords
        self.drag_idx      = -1
        self.hover_idx     = -1

    # ── internal helpers ──────────────────────────────────────────────────────

    def _to_orig(self, dx: int, dy: int):
        """Convert a display-space point to original image coordinates."""
        return (int(round(dx / self.display_scale)),
                int(round(dy / self.display_scale)))

    def _nearest(self, dx: int, dy: int) -> int:
        """Return index of the closest existing corner (display coords), or -1."""
        thresh = _DRAG_THRESHOLD   # threshold is in display pixels
        best_i, best_d = -1, thresh ** 2
        for i, (ox, oy) in enumerate(self.points):
            px = ox * self.display_scale
            py = oy * self.display_scale
            d  = (dx - px) ** 2 + (dy - py) ** 2
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    # ── OpenCV mouse callback ─────────────────────────────────────────────────

    def __call__(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.hover_idx = self._nearest(x, y)
            if self.drag_idx >= 0:
                self.points[self.drag_idx] = self._to_orig(x, y)

        elif event == cv2.EVENT_LBUTTONDOWN:
            idx = self._nearest(x, y)
            if idx >= 0:
                self.drag_idx = idx
            elif len(self.points) < 4:
                self.points.append(self._to_orig(x, y))
                print(f'[Simulator] Corner {len(self.points)} placed: {self.points[-1]}')

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drag_idx >= 0:
                print(f'[Simulator] Corner {self.drag_idx + 1} moved to: {self.points[self.drag_idx]}')
                self.drag_idx = -1

    # ── draw overlay on a display-sized copy ──────────────────────────────────

    def draw(self, display_img: np.ndarray) -> np.ndarray:
        out = display_img.copy()
        S   = self.display_scale
        pts_disp = [(int(ox * S), int(oy * S)) for ox, oy in self.points]

        # Polygon outline
        if len(pts_disp) >= 2:
            arr = np.array(pts_disp, dtype=np.int32)
            cv2.polylines(out, [arr], len(pts_disp) == 4, (255, 80, 0), 2)

        # Corner markers
        for i, (px, py) in enumerate(pts_disp):
            if i == self.drag_idx:
                color = (0, 255, 255)   # yellow while dragging
            elif i == self.hover_idx:
                color = (255, 255, 0)   # cyan on hover
            else:
                color = (0, 220, 0)     # green normally
            cv2.circle(out, (px, py),  8, color, -1)
            cv2.circle(out, (px, py), 15, color,  2)
            cv2.putText(out, _CORNER_LABELS[i],
                        (px + 18, py + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

        # Status text
        n = len(self.points)
        cv2.putText(out, f'Corners: {n} / 4',
                    (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 230, 230), 2, cv2.LINE_AA)
        if n == 4:
            cv2.putText(out, 'Press SPACE to warp',
                        (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 0), 2, cv2.LINE_AA)
        elif n < 4:
            cv2.putText(out, f'Click corner {n + 1} ({_CORNER_LABELS[n]})',
                        (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.putText(out, 'R=reset  D=undo  Esc=quit',
                    (10, out.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1, cv2.LINE_AA)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Homography & warp  (mirrors scene_understanding + gui/app.py pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_warp(orig_img: np.ndarray, corners: np.ndarray):
    """
    Apply the same perspective warp as the main pipeline.

    Parameters
    ----------
    orig_img : np.ndarray  BGR image
    corners  : np.ndarray  shape (4, 2) float32, order [TL, TR, BR, BL]
               in *original* image pixel coordinates

    Returns
    -------
    warped  : np.ndarray (OUT_H × OUT_W × 3)
    H       : np.ndarray (3 × 3) camera → table-cm homography
    """
    dst = np.array([
        [0,             0],
        [TABLE_WIDTH_CM, 0],
        [TABLE_WIDTH_CM, TABLE_HEIGHT_CM],
        [0,             TABLE_HEIGHT_CM],
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)

    S = np.array([
        [TABLE_DISPLAY_SCALE, 0, 0],
        [0, TABLE_DISPLAY_SCALE, 0],
        [0, 0,                  1],
    ], dtype=np.float32)

    warped = cv2.warpPerspective(orig_img, S @ H, (OUT_W, OUT_H))
    return warped, H


def _build_mask_preview(orig_img: np.ndarray,
                        corners: np.ndarray,
                        display_scale: float) -> np.ndarray:
    """
    Return the original image (display-sized) with a semi-transparent
    green polygon marking the table corners — same region as the mask
    produced by get_table_corners() in scene_understanding.py.
    """
    dh = int(orig_img.shape[0] * display_scale)
    dw = int(orig_img.shape[1] * display_scale)
    disp = cv2.resize(orig_img, (dw, dh), interpolation=cv2.INTER_AREA)

    # Scale corners to display space
    pts_disp = (corners * display_scale).astype(np.int32)

    overlay = disp.copy()
    cv2.fillPoly(overlay, [pts_disp], (0, 180, 0))          # green fill
    cv2.addWeighted(overlay, 0.30, disp, 0.70, 0, disp)     # semi-transparent

    cv2.polylines(disp, [pts_disp], True, (0, 230, 0), 2, cv2.LINE_AA)  # border

    # Label each corner
    S = display_scale
    for i, (ox, oy) in enumerate(corners):
        px, py = int(ox * S), int(oy * S)
        cv2.circle(disp, (px, py), 6, (0, 255, 0), -1)
        cv2.putText(disp, _CORNER_LABELS[i], (px + 10, py + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(disp, 'Table mask  (get_table_corners() format)',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(disp, 'S=save  N=new image  Esc=quit',
                (10, disp.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1, cv2.LINE_AA)
    return disp


def _annotate_warped(warped: np.ndarray) -> np.ndarray:
    """Add info overlay to the warped result window."""
    out = warped.copy()
    # Table-boundary rectangle
    cv2.rectangle(out, (0, 0), (OUT_W - 1, OUT_H - 1), (0, 200, 200), 2)
    cv2.putText(out, f'{OUT_W}x{OUT_H} px  |  {TABLE_DISPLAY_SCALE} px/cm',
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(out, 'S=save  N=new image  Esc=quit',
                (8, OUT_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Save helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_outputs(image_path: str,
                  warped: np.ndarray,
                  corners: np.ndarray):
    base  = os.path.splitext(image_path)[0]
    wpath = base + '_warped.png'
    cpath = base + '_corners.npy'
    cv2.imwrite(wpath, warped)
    np.save(cpath, corners)
    print(f'[Simulator] Warped image saved → {wpath}')
    print(f'[Simulator] Corners saved      → {cpath}')
    print(f'[Simulator] corners.npy shape  : {corners.shape}  dtype: {corners.dtype}')


# ─────────────────────────────────────────────────────────────────────────────
# Single-image session
# ─────────────────────────────────────────────────────────────────────────────

def _run_session(image_path: str) -> bool:
    """
    Run one corner-selection + warp session.

    Returns True  → user wants to open a new image (pressed N)
            False → user wants to quit (Esc/Q)
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f'[Simulator] ERROR: cannot read image: {image_path}')
        return False

    ih, iw = img.shape[:2]

    # Compute display scale so the window fits nicely on screen
    try:
        # cv2.getWindowImageRect requires a named window to exist first;
        # use a heuristic based on common screen heights
        screen_h = 1080
        screen_w = 1920
    except Exception:
        screen_h, screen_w = 1080, 1920

    ds = min(_SCREEN_FRACTION * screen_w / iw,
             _SCREEN_FRACTION * screen_h / ih,
             1.0)  # never upscale
    disp_w = max(1, int(iw * ds))
    disp_h = max(1, int(ih * ds))
    display_base = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    picker = _CornerPicker(display_scale=ds)

    win1 = 'Simulator — click 4 table corners'
    cv2.namedWindow(win1, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win1, disp_w, disp_h)
    cv2.setMouseCallback(win1, picker)

    print(f'\n[Simulator] Image: {os.path.basename(image_path)}  ({iw}×{ih})')
    print('[Simulator] Click corners in order: TL → TR → BR → BL')
    print('[Simulator] Controls: R=reset | D/Backspace=undo | Space=warp | Esc=quit\n')

    warped_result = None
    corners_result = None
    win2 = win3 = None

    while True:
        # ── Draw corner selection overlay ──────────────────────────────────────
        frame = picker.draw(display_base)
        cv2.imshow(win1, frame)

        # Guard against the user closing the window with the OS close button
        try:
            if cv2.getWindowProperty(win1, cv2.WND_PROP_VISIBLE) < 1:
                return False
        except Exception:
            return False

        key = cv2.waitKey(30) & 0xFF

        # ── Key handling — corner window ───────────────────────────────────────
        if key in (ord('q'), ord('Q'), 27):   # Esc / Q
            cv2.destroyAllWindows()
            return False

        elif key in (ord('r'), ord('R')):
            picker.points = []
            picker.drag_idx = -1
            print('[Simulator] Corners reset.')

        elif key in (ord('d'), ord('D'), 8):   # D or Backspace
            if picker.points:
                picker.points.pop()
                print(f'[Simulator] Removed last corner. Remaining: {len(picker.points)}')

        elif key == ord('n') or key == ord('N'):
            cv2.destroyAllWindows()
            return True  # new image

        elif key == ord('s') or key == ord('S'):
            if warped_result is not None and corners_result is not None:
                _save_outputs(image_path, warped_result, corners_result)
            else:
                print('[Simulator] Nothing to save yet — press Space first.')

        elif key == 32 and len(picker.points) == 4:   # Space — compute warp
            corners_arr = np.array(picker.points, dtype=np.float32)
            warped_result, H = _compute_warp(img, corners_arr)
            corners_result = corners_arr

            # Print corners (same format as get_table_corners())
            print('\n[Simulator] ─── Outputs ──────────────────────────────────')
            print(f'Corner coordinates (shape {corners_arr.shape}, float32):')
            print(corners_arr)
            print(f'\nHomography H (camera → table cm):\n{H}')
            print(f'\nWarped image size: {OUT_W}×{OUT_H} px  '
                  f'({TABLE_WIDTH_CM}×{TABLE_HEIGHT_CM} cm @ {TABLE_DISPLAY_SCALE} px/cm)')
            print('[Simulator] Press S to save outputs, N for new image.\n')

            # Window 2 — warped result
            win2 = 'Warped result  (ball-detection input)'
            annotated = _annotate_warped(warped_result)
            cv2.namedWindow(win2, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win2, OUT_W, OUT_H)
            cv2.imshow(win2, annotated)

            # Window 3 — mask preview
            win3 = 'Table mask  (get_table_corners() format)'
            mask_img = _build_mask_preview(img, corners_arr, ds)
            cv2.namedWindow(win3, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win3, disp_w, disp_h)
            cv2.imshow(win3, mask_img)

        # ── Also handle keys in result windows ─────────────────────────────────
        # (cv2.waitKey catches keys from any focused OpenCV window)
        if win2 is not None:
            try:
                if cv2.getWindowProperty(win2, cv2.WND_PROP_VISIBLE) < 1:
                    win2 = None
            except Exception:
                win2 = None
        if win3 is not None:
            try:
                if cv2.getWindowProperty(win3, cv2.WND_PROP_VISIBLE) < 1:
                    win3 = None
            except Exception:
                win3 = None


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('  Billiards Table Simulator')
    print(f'  Output size: {OUT_W}×{OUT_H} px  '
          f'({TABLE_DISPLAY_SCALE} px/cm,  {TABLE_WIDTH_CM}×{TABLE_HEIGHT_CM} cm)')
    print('=' * 60)

    while True:
        path = _pick_image()
        if not path:
            print('[Simulator] No image selected. Exiting.')
            break

        want_new = _run_session(path)
        cv2.destroyAllWindows()
        if not want_new:
            break

    print('[Simulator] Done.')


if __name__ == '__main__':
    main()
