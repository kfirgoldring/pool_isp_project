"""
Scene Understanding Module
===========================
Handles all camera-side scene analysis:
  - Table detection (HSV masking + contour analysis)
  - Camera → table homography computation
  - Manual camera calibration (4-corner click UI)
  - Auto-calibration from a static image

Public API
----------
detect_table_green(img)                        -> binary mask
get_table_corners(image_path)                  -> np.ndarray (4×2) or None
compute_homography_from_corners(corners, ...)  -> np.ndarray (3×3)
calibrate_from_image(image_path)               -> np.ndarray or None
CameraCalibrator                               (class)
"""

import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from typing import Optional, Tuple

import config


# ─────────────────────────────────────────────────────────────────────────────
# Table detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_table_green(img: np.ndarray) -> np.ndarray:
    """
    Return a binary mask of the green felt area.

    Uses HSV thresholding (Hue 60–85) after a Gaussian blur.
    """
    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    lower_green = np.array([60,  0,   0])
    upper_green = np.array([85, 255, 255])

    return cv2.inRange(hsv, lower_green, upper_green)


def get_table_corners(image_or_path) -> Optional[np.ndarray]:
    """
    Detect the four corners of the billiards table in a camera image.

    Parameters
    ----------
    image_or_path : str or np.ndarray
        Either a file path (str) to a BGR image, or a BGR numpy array
        captured directly from the camera.

    Returns
    -------
    np.ndarray of shape (4, 2) and dtype float32, ordered
        [Top-Left, Top-Right, Bottom-Right, Bottom-Left],
    or None if detection fails.
    """
    if isinstance(image_or_path, str):
        img = cv2.imread(image_or_path)
        if img is None:
            print(f'[Scene_Understanding] ERROR: Cannot load image: {image_or_path}')
            return None
    else:
        img = image_or_path

    mask = detect_table_green(img)

    if mask is None or mask.size == 0:
        print('[Scene_Understanding] ERROR: Could not create green mask.')
        return None

    # Close ball holes and pocket gaps with a large rectangular kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 60))
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Largest contour = table surface
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print('[Scene_Understanding] ERROR: No contours found.')
        return None
    table_contour = max(contours, key=cv2.contourArea)

    # Convex hull → extreme-point corners
    hull = cv2.convexHull(table_contour)
    pts  = hull.reshape(-1, 2)

    rect = np.zeros((4, 2), dtype='float32')
    s          = pts.sum(axis=1)
    rect[0]    = pts[np.argmin(s)]   # Top-Left
    rect[2]    = pts[np.argmax(s)]   # Bottom-Right
    diff       = np.diff(pts, axis=1)
    rect[1]    = pts[np.argmin(diff)] # Top-Right
    rect[3]    = pts[np.argmax(diff)] # Bottom-Left

    return rect


# ─────────────────────────────────────────────────────────────────────────────
# Homography computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_homography_from_corners(
        corners: np.ndarray,
        table_width:  float = config.TABLE_WIDTH_CM,
        table_height: float = config.TABLE_HEIGHT_CM) -> np.ndarray:
    """
    Compute the camera → table homography from 4 detected/clicked corners.

    Parameters
    ----------
    corners : np.ndarray, shape (4, 2)
        Camera-pixel coordinates ordered [TL, TR, BR, BL].
    table_width, table_height : float
        Physical table dimensions in the target coordinate system (cm by default).

    Returns
    -------
    np.ndarray, shape (3, 3) — the perspective transform matrix H such that
        table_pt = H @ camera_pt  (in homogeneous coords).
    """
    dst = np.array([
        [0,           0],
        [table_width, 0],
        [table_width, table_height],
        [0,           table_height],
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return H


def save_camera_homography(H: np.ndarray,
                           filepath: str = config.CAMERA_CALIB_FILE):
    """Save a homography matrix to disk as a .npy file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, H)
    print(f'[Scene_Understanding] Camera homography saved to {filepath}')


def load_camera_homography(filepath: str = config.CAMERA_CALIB_FILE) -> np.ndarray:
    """Load a previously saved camera homography from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'Camera calibration not found: {filepath}')
    return np.load(filepath)


def calibrate_from_camera(camera_index: int = config.CAMERA_INDEX,
                          save: bool = True) -> Optional[np.ndarray]:
    """
    Capture a live frame from the camera, auto-detect table corners,
    and compute the camera → table homography.

    Parameters
    ----------
    camera_index : int
        OpenCV camera index.
    save : bool
        If True, write the homography to config.CAMERA_CALIB_FILE.

    Returns
    -------
    np.ndarray (3×3) homography, or None if detection fails.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f'[Scene_Understanding] Cannot open camera {camera_index}.')
        return None

    # Let the camera settle (auto-exposure, white balance, etc.)
    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print('[Scene_Understanding] Cannot capture frame from camera.')
        return None

    corners = get_table_corners(frame)
    if corners is None:
        print('[Scene_Understanding] Auto-calibration failed: table corners not detected.')
        return None

    H = compute_homography_from_corners(corners)
    print('[Scene_Understanding] Camera homography computed from live camera frame.')

    if save:
        save_camera_homography(H)

    return H


def calibrate_from_image(image_path: str,
                         save: bool = True) -> Optional[np.ndarray]:
    """
    Auto-calibrate the camera from a static image.

    Detects the table corners automatically (no user interaction),
    computes the camera → table homography, and optionally saves it.

    Parameters
    ----------
    image_path : str
        Path to a BGR image of the table.
    save : bool
        If True, write the homography to config.CAMERA_CALIB_FILE.

    Returns
    -------
    np.ndarray (3×3) homography, or None if detection fails.
    """
    corners = get_table_corners(image_path)
    if corners is None:
        print('[Scene_Understanding] Auto-calibration failed: corners not detected.')
        return None

    H = compute_homography_from_corners(corners)
    print('[Scene_Understanding] Camera homography computed from image corners.')

    if save:
        save_camera_homography(H)

    return H


# ─────────────────────────────────────────────────────────────────────────────
# Manual camera calibration (interactive 4-corner click UI)
# ─────────────────────────────────────────────────────────────────────────────

class CameraCalibrator:
    """
    Manual camera calibration via a live camera feed.

    The user clicks (and optionally drags) 4 table corners on a captured
    frame; the class then computes and saves the camera → table homography.

    This is the Scene_Understanding version of ManualCameraCalibrator from
    billiards_calibration_merged.py, updated to use config.py constants.

    Usage
    -----
    calibrator = CameraCalibrator()
    H = calibrator.calibrate(camera_index=0)
    if H is not None:
        calibrator.save()
    """

    def __init__(self,
                 table_width:  float = config.TABLE_WIDTH_CM,
                 table_height: float = config.TABLE_HEIGHT_CM):
        self.table_width   = table_width
        self.table_height  = table_height
        self.image_points  = []
        self.homography    = None
        self.dragging_idx  = -1
        self.hover_idx     = -1

    # ── Geometry helpers ──────────────────────────────────────────────────────

    def distance(self, p1, p2) -> float:
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def find_closest_point(self, x: int, y: int, threshold: int = 15) -> int:
        """Return index of the point within threshold of (x, y), or -1."""
        for i, pt in enumerate(self.image_points):
            if self.distance((x, y), pt) < threshold:
                return i
        return -1

    # ── Mouse callback ────────────────────────────────────────────────────────

    def mouse_callback(self, event, x: int, y: int, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.hover_idx = self.find_closest_point(x, y)
            if self.dragging_idx >= 0:
                self.image_points[self.dragging_idx] = (x, y)

        elif event == cv2.EVENT_LBUTTONDOWN:
            idx = self.find_closest_point(x, y)
            if idx >= 0:
                self.dragging_idx = idx
                print(f'Dragging point {idx + 1}')
            elif len(self.image_points) < 4:
                self.image_points.append((x, y))
                print(f'Added point {len(self.image_points)}: ({x}, {y})')

        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging_idx >= 0:
                pt = self.image_points[self.dragging_idx]
                print(f'Point {self.dragging_idx + 1} placed at: ({pt[0]}, {pt[1]})')
                self.dragging_idx = -1

    # ── Main calibration routine ──────────────────────────────────────────────

    def calibrate(self, camera_index: int = config.CAMERA_INDEX
                  ) -> Optional[np.ndarray]:
        """
        Open the camera, capture one frame, and let the user click 4 corners.

        Controls
        --------
        Left-click        Add a corner point (up to 4)
        Click + drag      Reposition an existing point
        R                 Reset all points
        D / Backspace     Remove last point
        Space             Confirm and compute homography (needs 4 points)
        Esc               Cancel

        Returns
        -------
        np.ndarray (3×3) homography or None if cancelled.
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f'[CameraCalibrator] ERROR: Cannot open camera {camera_index}')
            return None

        ret, frame = cap.read()
        cap.release()

        if not ret:
            print('[CameraCalibrator] ERROR: Cannot read from camera.')
            return None

        win = 'Camera Calibration — click 4 table corners'
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, self.mouse_callback)

        print('\n=== Camera Calibration ===')
        print('Click the 4 table corners in order:')
        print('  1. Top-left  2. Top-right  3. Bottom-right  4. Bottom-left')
        print('Controls: R=reset | D/Backspace=undo | Space=confirm | Esc=cancel\n')

        while True:
            display = frame.copy()

            # Draw points
            for i, pt in enumerate(self.image_points):
                if i == self.dragging_idx:
                    color = (255, 255,   0)   # cyan while dragging
                elif i == self.hover_idx:
                    color = (0,   255, 255)   # yellow on hover
                else:
                    color = (0,   255,   0)   # green normally
                cv2.circle(display, pt, 8,  color, -1)
                cv2.circle(display, pt, 15, color,  2)
                cv2.putText(display, str(i + 1), (pt[0] + 20, pt[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Draw outline polygon
            if len(self.image_points) >= 2:
                pts_arr = np.array(self.image_points, np.int32)
                cv2.polylines(display, [pts_arr],
                              len(self.image_points) == 4, (255, 0, 0), 2)

            # Status overlay
            cv2.putText(display, f'Points: {len(self.image_points)}/4',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            if len(self.image_points) == 4:
                cv2.putText(display, 'Press SPACE to confirm',
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow(win, display)

            # Guard against window being closed by the user
            try:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    cv2.destroyAllWindows()
                    return None
            except Exception:
                pass

            key = cv2.waitKey(30) & 0xFF

            if key in (ord('r'), ord('R')):
                self.image_points = []
                self.dragging_idx = -1
                print('Reset all points.')

            elif key in (ord('d'), ord('D'), 8):   # D or Backspace
                if self.image_points:
                    self.image_points.pop()
                    print(f'Removed last point. Remaining: {len(self.image_points)}')

            elif key == 27:   # Esc
                print('[CameraCalibrator] Cancelled.')
                cv2.destroyAllWindows()
                return None

            elif key == 32 and len(self.image_points) == 4:   # Space
                break

        cv2.destroyAllWindows()

        # Compute homography: camera pixels → table cm
        src = np.array(self.image_points, dtype=np.float32)
        dst = np.array([
            [0,                 0],
            [self.table_width,  0],
            [self.table_width,  self.table_height],
            [0,                 self.table_height],
        ], dtype=np.float32)
        self.homography, _ = cv2.findHomography(src, dst)
        print('[CameraCalibrator] Homography computed.')
        return self.homography

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, filepath: str = config.CAMERA_CALIB_FILE):
        """Save the computed homography to disk."""
        if self.homography is None:
            print('[CameraCalibrator] ERROR: No homography to save.')
            return
        save_camera_homography(self.homography, filepath)

    @staticmethod
    def load(filepath: str = config.CAMERA_CALIB_FILE) -> np.ndarray:
        """Load a previously saved camera homography from disk."""
        return load_camera_homography(filepath)


# ─────────────────────────────────────────────────────────────────────────────
# Debug utilities
# ─────────────────────────────────────────────────────────────────────────────

def show_hue_histogram(image_path: str = r'table_pics\WIN_20260219_10_29_03_Pro.jpg'):
    """
    Plot the Hue-channel histogram of an image.
    Useful for tuning the HSV thresholds in detect_table_green().
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f'[Scene_Understanding] Failed to load image: {image_path}')
        return
    hue = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0]
    plt.figure(figsize=(8, 4))
    plt.hist(hue.ravel(), bins=180, range=[0, 180], color='g', alpha=0.7)
    plt.title('Hue Channel Histogram')
    plt.xlabel('Hue Value (0–179)')
    plt.ylabel('Pixel Count')
    plt.grid(True)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

def main():
    image_path = r'table_pics\WIN_20260219_10_30_02_Pro.jpg'
    print(f'[Scene_Understanding] Detecting table corners from: {image_path}')
    corners = get_table_corners(image_path)
    if corners is not None:
        print(f'Corners detected:\n{corners}')
        H = compute_homography_from_corners(corners)
        print(f'Camera→table homography:\n{H}')
    else:
        print('No corners detected.')


if __name__ == '__main__':
    main()
