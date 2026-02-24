"""
Hough-only pool ball detection for a single frame using OpenCV.

This keeps the same general preprocessing pipeline as ball_detection.py
but uses Hough circles as the sole detector.
"""
from typing import Dict, Iterable, List, Optional, Tuple
import cv2
import numpy as np
Point = Tuple[float, float]
BBox = Tuple[int, int, int, int]  # x, y, w, h

_ref_cache: Dict[str, np.ndarray] = {}


def _load_ref_image(ref_path: str) -> Optional[np.ndarray]:
    """Load and cache the reference image so it's read from disk only once."""
    cached = _ref_cache.get(ref_path)
    if cached is not None:
        return cached
    img = cv2.imread(ref_path)
    if img is not None:
        _ref_cache[ref_path] = img
    return img


class BallDetection:
    def __init__(self, center: Tuple[float, float], radius_px: float, bbox: BBox) -> None:
        self.center = center
        self.radius_px = radius_px
        self.bbox = bbox


class BallDetectionConfig:
    def __init__(
        self,
        green_hue_window: int = 12,
        green_min_sat: int = 200,
        green_min_val: int = 50,
        clahe_enabled: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: int = 8,
        kernel_size: int = 3,
        hough_dp: float = 1,
        hough_param1: float = 200,
        hough_param2: float = 15,
        min_circularity: float = 0.35,
        min_area_ratio: float = 0.002,
        max_area_ratio: float = 0.004,
        diff_ratio: float = 0.6,
        hue_std_max: float = 12.0,
        skin_ratio_max: float = 0.2,
        edge_margin: float = 0.02,
        hue_similarity_thresh: float = 10.0,
        # Color classification thresholds (HSV
        yellow_hue: Tuple[int, int] = (25, 30),
        blue_hue: Tuple[int, int] = (100, 110),
        purple_hue: Tuple[int, int] = (70, 90),
        red1_hue: Tuple[int, int] = (0, 10),
        red2_hue: Tuple[int, int] = (170, 179),
        black_val_max: int = 75,
        black_pixel_val_max: int = 90,
        black_dark_ratio_min: float = 0.45,
        black_sat_for_dark: int = 110,
        black_inner_scale: float = 0.7,
        white_sat_diff_thresh: int = 100,
        enable_color_filter: bool = True,
    ) -> None:
        # HSV thresholds for green table (will be adapted around dominant hue)
        self.green_hue_window = green_hue_window
        self.green_min_sat = green_min_sat
        self.green_min_val = green_min_val
        # CLAHE on HSV V channel
        self.clahe_enabled = clahe_enabled
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        # Morphology
        self.kernel_size = kernel_size
        # Circle detection params
        self.hough_dp = hough_dp
        self.hough_param1 = hough_param1
        self.hough_param2 = hough_param2
        # Fallback contour filtering
        self.min_circularity = min_circularity
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        # Non-green coverage threshold for circle filtering
        self.diff_ratio = diff_ratio
        # Color/texture filters
        self.hue_std_max = hue_std_max
        self.skin_ratio_max = skin_ratio_max
        # Edge filter margin in normalized table coordinates
        self.edge_margin = edge_margin
        # If bbox mean hue is within this threshold of table hue, drop it
        self.hue_similarity_thresh = hue_similarity_thresh
        # Color classification thresholds
        self.yellow_hue = yellow_hue
        self.blue_hue = blue_hue
        self.purple_hue = purple_hue
        self.red1_hue = red1_hue
        self.red2_hue = red2_hue
        self.white_sat_diff_thresh = white_sat_diff_thresh
        self.black_val_max = black_val_max
        self.black_pixel_val_max = black_pixel_val_max
        self.black_dark_ratio_min = black_dark_ratio_min
        self.black_sat_for_dark = black_sat_for_dark
        self.black_inner_scale = black_inner_scale
        self.enable_color_filter = enable_color_filter


def _order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    """Return points ordered as: top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32)
    c = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    order = np.argsort(angles)
    pts = pts[order]
    # Ensure the first point is top-left
    s = pts.sum(axis=1)
    start = np.argmin(s)
    pts = np.roll(pts, -start, axis=0)
    return pts


def _table_mask(frame_shape: Tuple[int, int], corners: Iterable[Point]) -> np.ndarray:
    h, w = frame_shape[:2]
    corners = _order_points_clockwise(np.array(list(corners), dtype=np.float32))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, corners.astype(np.int32), 255)
    return mask


def _circle_mask(
    shape: Tuple[int, int],
    center: Tuple[float, float],
    radius: float,
    inner_scale: float = 0.0,
) -> np.ndarray:
    h, w = shape
    if radius <= 0:
        return np.zeros((h, w), dtype=bool)
    cx, cy = float(center[0]), float(center[1])
    yy, xx = np.ogrid[:h, :w]
    r2 = radius * radius
    outer = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r2
    if inner_scale <= 0.0:
        return outer
    inner_r = max(0.0, radius * inner_scale)
    inner = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (inner_r * inner_r)
    return outer & (~inner)


def _median_hue_in_circle(
    hsv: np.ndarray,
    center: Tuple[float, float],
    radius: float,
    min_sat: int,
    min_val: int,
) -> Optional[float]:
    """
    Circular median hue in [0, 180) for robust color classification.
    Uses the sample hue that minimizes summed circular distance.
    """
    h_img, w_img = hsv.shape[:2]
    if radius <= 0:
        return None
    mask = _circle_mask((h_img, w_img), center, radius, inner_scale=0.0)
    if not np.any(mask):
        return None
    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]
    valid = mask & (s_ch >= min_sat) & (v_ch >= min_val)
    if not np.any(valid):
        return None

    hues = h_ch[valid].astype(np.float32).ravel()
    if hues.size == 0:
        return None
    if hues.size == 1:
        return float(hues[0])

    max_samples = 500
    if hues.size > max_samples:
        idx = np.linspace(0, hues.size - 1, max_samples, dtype=np.int32)
        hues = hues[idx]

    a = hues[:, None]
    b = hues[None, :]
    d = np.abs(a - b)
    circ_d = np.minimum(d, 180.0 - d)
    best_idx = int(np.argmin(np.sum(circ_d, axis=1)))
    return float(hues[best_idx])


def _mean_sv_in_circle(
    hsv: np.ndarray,
    center: Tuple[float, float],
    radius: float,
) -> Optional[Tuple[float, float]]:
    h_img, w_img = hsv.shape[:2]
    if radius <= 0:
        return None
    mask = _circle_mask((h_img, w_img), center, radius, inner_scale=0.0)
    if not np.any(mask):
        return None
    s_ch = hsv[:, :, 1].astype(np.float32)
    v_ch = hsv[:, :, 2].astype(np.float32)
    return float(np.mean(s_ch[mask])), float(np.mean(v_ch[mask]))


def _hue_std_in_circle(
    hsv: np.ndarray,
    center: Tuple[float, float],
    radius: float,
    min_sat: int,
    min_val: int,
) -> Optional[float]:
    h_img, w_img = hsv.shape[:2]
    if radius <= 0:
        return None
    mask = _circle_mask((h_img, w_img), center, radius, inner_scale=0.0)
    if not np.any(mask):
        return None
    h_ch = hsv[:, :, 0].astype(np.float32)
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]
    valid = mask & (s_ch >= min_sat) & (v_ch >= min_val)
    if not np.any(valid):
        return None
    hues = h_ch[valid].ravel()
    if hues.size < 2:
        return 0.0
    return float(np.std(hues))


def _skin_ratio_in_circle(
    frame_bgr: np.ndarray,
    center: Tuple[float, float],
    radius: float,
) -> Optional[float]:
    h_img, w_img = frame_bgr.shape[:2]
    if radius <= 0:
        return None
    mask = _circle_mask((h_img, w_img), center, radius, inner_scale=0.0)
    if not np.any(mask):
        return None
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    skin = (cr >= 135) & (cr <= 180) & (cb >= 85) & (cb <= 135)
    valid = mask
    total = int(np.count_nonzero(valid))
    if total == 0:
        return None
    skin_count = int(np.count_nonzero(skin & valid))
    return float(skin_count / float(total))


def _classify_color(
    hsv: np.ndarray,
    center: Tuple[float, float],
    radius: float,
    cfg: BallDetectionConfig,
) -> str:
    sv = _mean_sv_in_circle(hsv, center, radius)
    if sv is not None:
        mean_s, mean_v = sv
        if mean_v - mean_s >= cfg.white_sat_diff_thresh:
            return "white"
    median_hue = _median_hue_in_circle(hsv, center, radius, cfg.green_min_sat, cfg.green_min_val)
    if median_hue is None:
        return "unknown"

    h = float(median_hue)

    def _in_range(val: float, hue_range: Tuple[int, int]) -> bool:
        lo, hi = hue_range
        if lo <= hi:
            return lo <= val <= hi
        # Wrap-around range
        return val >= lo or val <= hi

    if _in_range(h, cfg.yellow_hue):
        return "yellow"
    if _in_range(h, cfg.blue_hue):
        return "blue"
    if _in_range(h, cfg.purple_hue):
        return "purple"
    if _in_range(h, cfg.red1_hue) or _in_range(h, cfg.red2_hue):
        return "bordeaux"
    return "unknown"


def _estimate_ball_radius_px(
    corners: np.ndarray,
    ball_diameter_cm=3,
    table_size_cm=120.0,
) -> Optional[float]:
    if ball_diameter_cm is None or table_size_cm is None:
        return None
    corners = _order_points_clockwise(corners)
    # Edge lengths in px
    top = np.linalg.norm(corners[1] - corners[0])
    bottom = np.linalg.norm(corners[2] - corners[3])
    left = np.linalg.norm(corners[3] - corners[0])
    right = np.linalg.norm(corners[2] - corners[1])
    width_px = 0.5 * (top + bottom)
    height_px = 0.5 * (left + right)
    width_cm, height_cm = table_size_cm
    if width_cm <= 0 or height_cm <= 0:
        return None
    px_per_cm = 0.5 * (width_px / width_cm + height_px / height_cm)
    return 0.5 * ball_diameter_cm * px_per_cm


def _circle_diff_ratio(
    diff_mask: np.ndarray, center: Tuple[float, float], radius: float
) -> float:
    h, w = diff_mask.shape
    x, y = int(round(center[0])), int(round(center[1]))
    r = int(round(radius))
    if r <= 0:
        return 0.0
    x0, x1 = max(0, x - r), min(w - 1, x + r)
    y0, y1 = max(0, y - r), min(h - 1, y + r)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    roi = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
    cv2.circle(roi, (x - x0, y - y0), r, 255, -1)
    region = diff_mask[y0 : y1 + 1, x0 : x1 + 1]
    total = np.count_nonzero(roi)
    if total == 0:
        return 0.0
    return float(np.count_nonzero(region & roi) / total)


def _load_table_corners_from_scene_understanding(img_path: str) -> Optional[np.ndarray]:
    """
    use scene_understanding module to get table corners for masking.
    :return:
    """
    try:
        import scene_understanding as su
        corners = su.get_table_corners(img_path)
        corners = np.asarray(corners, dtype=np.float32)
        if corners.shape == (4, 2):
            return corners
    except Exception as e:
        raise RuntimeError("Failed to load table corners from scene_understanding.")
    return None


def detect_balls_hough(
    frame_bgr: np.ndarray,
    table_corners: Optional[Iterable[Point]] = None,
    ref_path: str = "ref.jpeg",
    ball_diameter_cm: Optional[float] = 3,
    table_size_cm: Optional[Tuple[float, float]] = [120.0, 60.0],
    config: Optional[BallDetectionConfig] = None,
) -> List[BallDetection]:
    cfg = config or BallDetectionConfig()
    if frame_bgr is None or frame_bgr.size == 0:
        return []
    elif table_corners is None:
        corners_arr = _load_table_corners_from_scene_understanding(ref_path)
        if corners_arr is None:
            raise ValueError("table_corners not provided and scene_understanding failed to detect corners.")
    else:
        corners_arr = np.array(list(table_corners), dtype=np.float32)
    if corners_arr.shape[0] != 4:
        raise ValueError("table_corners must contain exactly 4 points.")

    ref_bgr = _load_ref_image(ref_path)
    if ref_bgr is None:
        raise ValueError(f"Failed to read reference image: {ref_path}")
    if ref_bgr.shape[:2] != frame_bgr.shape[:2]:
        raise ValueError("Reference and current frame must have the same resolution.")

    corners_arr = _order_points_clockwise(corners_arr)
    table_mask = _table_mask(frame_bgr.shape, corners_arr)

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    if cfg.clahe_enabled:
        clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip_limit,
            tileGridSize=(cfg.clahe_grid_size, cfg.clahe_grid_size),
        )
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])

    # Estimate dominant table hue inside the table mask.
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    valid = (table_mask > 0) & (s >= cfg.green_min_sat) & (v >= cfg.green_min_val)
    if np.any(valid):
        hist = cv2.calcHist([h], [0], valid.astype(np.uint8), [180], [0, 180])
        dominant_hue = int(np.argmax(hist))
    else:
        dominant_hue = 60

    hw = cfg.green_hue_window
    lower1 = np.array([max(0, dominant_hue - hw), cfg.green_min_sat, cfg.green_min_val], dtype=np.uint8)
    upper1 = np.array([min(179, dominant_hue + hw), 255, 255], dtype=np.uint8)
    green_mask = cv2.inRange(hsv, lower1, upper1)
    if dominant_hue - hw < 0 or dominant_hue + hw > 179:
        lower2 = np.array([0, cfg.green_min_sat, cfg.green_min_val], dtype=np.uint8)
        upper2 = np.array([(dominant_hue + hw) % 180, 255, 255], dtype=np.uint8)
        lower3 = np.array([(dominant_hue - hw) % 180, cfg.green_min_sat, cfg.green_min_val], dtype=np.uint8)
        upper3 = np.array([179, 255, 255], dtype=np.uint8)
        green_mask = cv2.inRange(hsv, lower2, upper2) | cv2.inRange(hsv, lower3, upper3)

    non_green = cv2.bitwise_and(table_mask, cv2.bitwise_not(green_mask))
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    gray = cv2.bitwise_and(gray, gray, mask=non_green)

    # Build diff mask for diff_ratio filtering
    blur_frame = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
    blur_ref = cv2.GaussianBlur(ref_bgr, (5, 5), 0)
    diff_bgr = cv2.absdiff(blur_frame, blur_ref)
    diff_gray = cv2.cvtColor(diff_bgr, cv2.COLOR_BGR2GRAY)
    diff_gray = cv2.medianBlur(diff_gray, 3)
    diff_gray = cv2.bitwise_and(diff_gray, diff_gray, mask=table_mask)
    _, diff_mask = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)
    k = max(3, cfg.kernel_size | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel)

    expected_radius = _estimate_ball_radius_px(corners_arr, ball_diameter_cm, table_size_cm)
    if expected_radius is not None:
        min_r = max(6, int(round(0.75 * expected_radius)))
        max_r = int(round(1.35 * expected_radius))
        min_dist = int(round(2.0 * expected_radius))
    else:
        min_r = 8
        max_r = 40
        min_dist = 20

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=cfg.hough_dp,
        minDist=min_dist,
        param1=cfg.hough_param1,
        param2=cfg.hough_param2,
        minRadius=min_r,
        maxRadius=max_r,
    )

    detections: List[BallDetection] = []
    if circles is not None:
        for c in circles[0]:
            x, y, r = float(c[0]), float(c[1]), float(c[2])
            if _circle_diff_ratio(diff_mask, (x, y), r) < cfg.diff_ratio:
                continue
            x0 = int(round(x - r))
            y0 = int(round(y - r))
            detections.append(
                BallDetection(center=(x, y), radius_px=r, bbox=(x0, y0, int(round(2 * r)), int(round(2 * r))))
            )

    if not detections:
        return []

    dst = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(corners_arr.astype(np.float32), dst)
    filtered: List[BallDetection] = []
    edge = cfg.edge_margin
    for d in detections:
        pt = np.array([[[d.center[0], d.center[1]]]], dtype=np.float32)
        uv = cv2.perspectiveTransform(pt, H)[0][0]
        u, v = float(uv[0]), float(uv[1])
        if u < edge or u > 1.0 - edge or v < edge or v > 1.0 - edge:
            continue
        filtered.append(d)
    if not filtered:
        return []

    if not cfg.enable_color_filter:
        return filtered
    color_filtered: List[BallDetection] = []
    for d in filtered:
        color = _classify_color(hsv, d.center, d.radius_px, cfg)
        if color != "unknown":
            color_filtered.append(d)
        else:
            median_hue = _median_hue_in_circle(hsv, d.center, d.radius_px, cfg.green_min_sat, cfg.green_min_val)
            print(
                "dropped_by_color",
                "center", (round(d.center[0], 2), round(d.center[1], 2)),
                "radius", round(d.radius_px, 2),
                "median_hue", None if median_hue is None else round(median_hue, 2),
                "ranges",
                "yellow", cfg.yellow_hue,
                "blue", cfg.blue_hue,
                "purple", cfg.purple_hue,
                "red1", cfg.red1_hue,
                "red2", cfg.red2_hue,
            )
    return color_filtered


def draw_detections(
    frame_bgr: np.ndarray,
    detections: Iterable[BallDetection],
    color_bbox: Tuple[int, int, int] = (0, 255, 255),
    color_center: Tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    if frame_bgr is None:
        return None
    vis = frame_bgr.copy()
    for det in detections:
        cx, cy = det.center
        r = int(round(det.radius_px))
        if r > 0:
            cv2.circle(vis, (int(round(cx)), int(round(cy))), r, color_bbox, 2)
        cv2.circle(vis, (int(round(cx)), int(round(cy))), 4, color_center, -1)
    return vis


def save_hue_histogram(
    frame_bgr: np.ndarray,
    table_mask: np.ndarray,
    cfg: BallDetectionConfig,
    out_path: str = "table_hue_hist.png",
) -> None:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    valid = (table_mask > 0) & (s >= cfg.green_min_sat) & (v >= cfg.green_min_val)
    if not np.any(valid):
        return
    hist = cv2.calcHist([h], [0], valid.astype(np.uint8), [180], [0, 180]).ravel()
    hist = hist / (hist.max() + 1e-6)
    height = 220
    width = 360  # 2px per bin
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    for i in range(180):
        x0 = i * 2
        x1 = x0 + 2
        h_val = int(hist[i] * (height - 10))
        cv2.rectangle(img, (x0, height - 1), (x1 - 1, height - 1 - h_val), (0, 0, 0), -1)
    cv2.putText(img, "Hue Histogram (table)", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    # X-axis labels every 30 degrees
    for val in range(0, 181, 30):
        x = min(width - 1, val * 2)
        cv2.line(img, (x, height - 20), (x, height - 10), (0, 0, 0), 1)
        cv2.putText(img, str(val), (x - 6, height - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.imwrite(out_path, img)


def save_edge_debug(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    cfg: BallDetectionConfig,
    out_path: str = "hough_edges.png",
) -> None:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    gray = cv2.bitwise_and(gray, gray, mask=mask)
    edges = cv2.Canny(gray, cfg.hough_param1 * 0.5, cfg.hough_param1)
    cv2.imwrite(out_path, edges)


if __name__ == "__main__":
    ref_path = "runs/2026-02-23_18-13-09/ref.jpeg"
    img_path = "runs/2026-02-23_18-13-09/capture_20260223_181437.jpeg"
    img = cv2.imread(img_path)
    if img is None:
        raise SystemExit("Failed to read pool_table.jpeg")
    cfg = BallDetectionConfig()
    corners = _load_table_corners_from_scene_understanding(ref_path)
    if corners is None:
        raise SystemExit("Failed to load corners from scene_understanding")

    table_mask = _table_mask(img.shape, corners)
    save_hue_histogram(img, table_mask, cfg, out_path="table_hue_hist.png")
    # Build the same non-green mask used by Hough for edge debugging
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if cfg.clahe_enabled:
        clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip_limit,
            tileGridSize=(cfg.clahe_grid_size, cfg.clahe_grid_size),
        )
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    valid = (table_mask > 0) & (s >= cfg.green_min_sat) & (v >= cfg.green_min_val)
    if np.any(valid):
        hist = cv2.calcHist([h], [0], valid.astype(np.uint8), [180], [0, 180])
        dominant_hue = int(np.argmax(hist))
    else:
        dominant_hue = 60
    hw = cfg.green_hue_window
    lower1 = np.array([max(0, dominant_hue - hw), cfg.green_min_sat, cfg.green_min_val], dtype=np.uint8)
    upper1 = np.array([min(179, dominant_hue + hw), 255, 255], dtype=np.uint8)
    green_mask = cv2.inRange(hsv, lower1, upper1)
    if dominant_hue - hw < 0 or dominant_hue + hw > 179:
        lower2 = np.array([0, cfg.green_min_sat, cfg.green_min_val], dtype=np.uint8)
        upper2 = np.array([(dominant_hue + hw) % 180, 255, 255], dtype=np.uint8)
        lower3 = np.array([(dominant_hue - hw) % 180, cfg.green_min_sat, cfg.green_min_val], dtype=np.uint8)
        upper3 = np.array([179, 255, 255], dtype=np.uint8)
        green_mask = cv2.inRange(hsv, lower2, upper2) | cv2.inRange(hsv, lower3, upper3)
    non_green = cv2.bitwise_and(table_mask, cv2.bitwise_not(green_mask))
    save_edge_debug(img, non_green, cfg, out_path="hough_edges.png")

    dets = detect_balls_hough(img, table_corners=corners, ref_path=ref_path, config=cfg)
    vis = draw_detections(img, dets)
    for (x, y) in corners:
        cv2.circle(vis, (int(x), int(y)), 8, (255, 0, 255), -1)
    for i, d in enumerate(dets):
        x0, y0, w, h = d.bbox
        label_pos = (max(0, x0), max(15, y0 - 6))
        cv2.putText(
            vis,
            str(i),
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite("pool_table_annotated_hough.jpeg", vis)
    print(f"detections: {len(dets)}")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    # Build diff mask for diff_ratio reporting (same preprocessing as main pipeline)
    ref_bgr = _load_ref_image(ref_path)
    blur_frame = cv2.GaussianBlur(img, (5, 5), 0)
    blur_ref = cv2.GaussianBlur(ref_bgr, (5, 5), 0) if ref_bgr is not None else blur_frame
    diff_bgr = cv2.absdiff(blur_frame, blur_ref)
    gray = cv2.cvtColor(diff_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    gray = cv2.bitwise_and(gray, gray, mask=table_mask)
    _, diff_mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    k = max(3, cfg.kernel_size | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel)
    for i, d in enumerate(dets):
        cx, cy = d.center
        r = d.radius_px
        color = _classify_color(hsv, d.center, d.radius_px, cfg)
        h_img, w_img = hsv.shape[:2]
        yy, xx = np.ogrid[:h_img, :w_img]
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r * r)
        if not np.any(mask):
            continue
        median_hue = _median_hue_in_circle(hsv, (cx, cy), r, cfg.green_min_sat, cfg.green_min_val)
        mean_s = float(np.mean(s[mask]))
        mean_v = float(np.mean(v[mask]))
        pixel_area = float(np.count_nonzero(mask))
        circularity = None
        if r > 0:
            contour_mask = (mask.astype(np.uint8)) * 255
            cnts, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cnt = max(cnts, key=cv2.contourArea)
                area_c = cv2.contourArea(cnt)
                perim_c = cv2.arcLength(cnt, True)
                if perim_c > 1e-3:
                    circularity = float(4.0 * np.pi * area_c / (perim_c * perim_c))
        diff_ratio = _circle_diff_ratio(diff_mask, (cx, cy), r)
        # Hue std on current HSV, but only on diff_mask pixels inside the circle
        hue_std = None
        if diff_mask is not None:
            h_img, w_img = hsv.shape[:2]
            yy, xx = np.ogrid[:h_img, :w_img]
            mask_r = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r * r)
            mask_r = mask_r & (diff_mask > 0)
            if np.any(mask_r):
                hues = hsv[:, :, 0].astype(np.float32)[mask_r].ravel()
                if hues.size >= 2:
                    if hues.size > 500:
                        idx = np.linspace(0, hues.size - 1, 500, dtype=np.int32)
                        hues = hues[idx]
                    a = hues[:, None]
                    b = hues[None, :]
                    dists = np.abs(a - b)
                    circ_d = np.minimum(dists, 180.0 - dists)
                    hue_std = float(np.std(hues))
                elif hues.size == 1:
                    hue_std = 0.0
        skin_ratio = _skin_ratio_in_circle(img, (cx, cy), r)
        print(
            i,
            "center", (round(cx, 2), round(cy, 2)),
            "radius", round(r, 2),
            "pixel_area", round(pixel_area, 1),
            "circularity", None if circularity is None else round(circularity, 3),
            "diff_ratio", None if diff_ratio is None else round(diff_ratio, 3),
            "hue_std", None if hue_std is None else round(hue_std, 2),
            "skin_ratio", None if skin_ratio is None else round(skin_ratio, 3),
            "color", color,
            "median_hue", None if median_hue is None else round(median_hue, 2),
            "mean_s", round(mean_s, 2),
            "mean_v", round(mean_v, 2),
        )
