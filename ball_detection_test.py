"""
Hough-only pool ball detection for a single frame using OpenCV.
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


def clear_ref_cache(path: str) -> None:
    """Remove a cached reference image so the next call reloads from disk."""
    _ref_cache.pop(path, None)


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
        blue_hue: Tuple[int, int] = (100, 115),
        purple_hue: Tuple[int, int] = (70, 95),
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
    # Sample the inner 70% of the ball to avoid green table bleed at edges.
    inner_r = max(3.0, radius * 0.7)

    sv = _mean_sv_in_circle(hsv, center, radius)
    if sv is not None:
        mean_s, mean_v = sv
        if mean_v - mean_s >= cfg.white_sat_diff_thresh:
            return "white"
    median_hue = _median_hue_in_circle(hsv, center, radius, min_sat=80, min_val=30)
    if median_hue is None:
        median_hue = _median_hue_in_circle(hsv, center, inner_r, min_sat=80, min_val=30)
    if median_hue is None:
        return "unknown"

    h = float(median_hue)

    def _in_range(val: float, hue_range: Tuple[int, int]) -> bool:
        lo, hi = hue_range
        if lo <= hi:
            return lo <= val <= hi
        return val >= lo or val <= hi

    if _in_range(h, cfg.yellow_hue):
        return "yellow"
    if _in_range(h, cfg.red1_hue) or _in_range(h, cfg.red2_hue):
        return "bordeaux"
    # Blue and purple overlap in hue (both appear in 70-115 range depending
    # on sampling), but are cleanly separable by saturation:
    #   Blue:   S ~185-220 (vivid)
    #   Purple: S ~104-142 (moderate)
    if sv is not None:
        mean_s, _ = sv
        if 65 <= h <= 120:
            if mean_s >= 160:
                return "blue"
            else:
                return "purple"
    if _in_range(h, cfg.blue_hue):
        return "blue"
    if _in_range(h, cfg.purple_hue):
        return "purple"
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
        ref_bgr = cv2.resize(ref_bgr, (frame_bgr.shape[1], frame_bgr.shape[0]))

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
    return color_filtered


def detect_balls_with_color(
    frame_bgr: np.ndarray,
    table_corners: Optional[Iterable[Point]] = None,
    ref_path: str = "ref.jpeg",
    ball_diameter_cm: Optional[float] = 3,
    table_size_cm: Optional[Tuple[float, float]] = None,
    config: Optional[BallDetectionConfig] = None,
    manual_corners: bool = False,
) -> np.ndarray:
    """
    Detect pool balls and classify colors (pipeline-compatible API).

    Returns np.ndarray of shape (N, 3): [x_cam_px, y_cam_px, color_string].
    This matches the contract expected by main.py / _adapt_detections().
    """
    cfg = config or BallDetectionConfig()
    detections = detect_balls_hough(
        frame_bgr=frame_bgr,
        table_corners=table_corners,
        ref_path=ref_path,
        ball_diameter_cm=ball_diameter_cm,
        table_size_cm=table_size_cm,
        config=cfg,
    )

    if not detections or frame_bgr is None or frame_bgr.size == 0:
        return np.empty((0, 3), dtype=object)

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    if cfg.clahe_enabled:
        clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip_limit,
            tileGridSize=(cfg.clahe_grid_size, cfg.clahe_grid_size),
        )
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])

    out: List[List[object]] = []
    for d in detections:
        color = _classify_color(hsv, d.center, d.radius_px, cfg)
        out.append([float(d.center[0]), float(d.center[1]), color])

    if not out:
        return np.empty((0, 3), dtype=object)
    return np.array(out, dtype=object)


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


def _debug_print_detection(
    idx: int,
    det: BallDetection,
    hsv: np.ndarray,
    diff_mask: np.ndarray,
    frame_bgr: np.ndarray,
    cfg: BallDetectionConfig,
) -> str:
    """Print detailed debug info for a single detection. Returns the color label."""
    cx, cy = det.center
    r = det.radius_px
    color = _classify_color(hsv, det.center, r, cfg)

    h_img, w_img = hsv.shape[:2]
    ir = int(round(r))
    x0c, x1c = max(0, int(cx) - ir), min(w_img, int(cx) + ir + 1)
    y0c, y1c = max(0, int(cy) - ir), min(h_img, int(cy) + ir + 1)
    rh, rw = y1c - y0c, x1c - x0c
    if rh > 0 and rw > 0:
        yy, xx = np.ogrid[:rh, :rw]
        circ = ((xx - (int(cx) - x0c)) ** 2 + (yy - (int(cy) - y0c)) ** 2) <= (ir * ir)
        roi_s = hsv[y0c:y1c, x0c:x1c, 1].astype(np.float32)
        roi_v = hsv[y0c:y1c, x0c:x1c, 2].astype(np.float32)
        mean_s = float(np.mean(roi_s[circ])) if np.any(circ) else 0.0
        mean_v = float(np.mean(roi_v[circ])) if np.any(circ) else 0.0
    else:
        mean_s, mean_v = 0.0, 0.0

    median_hue = _median_hue_in_circle(hsv, (cx, cy), r, cfg.green_min_sat, cfg.green_min_val)
    diff_ratio = _circle_diff_ratio(diff_mask, (cx, cy), r)
    skin_ratio = _skin_ratio_in_circle(frame_bgr, (cx, cy), r)

    print(
        f"    [{idx:2d}] center=({cx:7.1f},{cy:7.1f})  r={r:5.1f}  "
        f"color={color:<9s}  hue={'  None' if median_hue is None else f'{median_hue:6.1f}'}  "
        f"S={mean_s:5.1f}  V={mean_v:5.1f}  "
        f"diff_r={diff_ratio:.2f}  skin={'  None' if skin_ratio is None else f'{skin_ratio:.3f}'}"
    )
    return color


def _process_single_frame(
    img_path: str,
    ref_path: str,
    corners: np.ndarray,
    cfg: BallDetectionConfig,
    debug_dir: Optional[str],
) -> None:
    """Run Hough detection on one frame with full debug output."""
    import os
    basename = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path)
    if img is None:
        print(f"  [SKIP] Cannot read {img_path}")
        return

    print(f"\n  --- {basename} ({img.shape[1]}x{img.shape[0]}) ---")

    # Run detection
    dets = detect_balls_hough(img, table_corners=corners, ref_path=ref_path, config=cfg)
    print(f"  Hough detections: {len(dets)}")

    # Build supporting data for debug prints
    table_mask_arr = _table_mask(img.shape, corners)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if cfg.clahe_enabled:
        clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip_limit,
            tileGridSize=(cfg.clahe_grid_size, cfg.clahe_grid_size),
        )
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])

    ref_bgr = _load_ref_image(ref_path)
    blur_frame = cv2.GaussianBlur(img, (5, 5), 0)
    blur_ref = cv2.GaussianBlur(ref_bgr, (5, 5), 0) if ref_bgr is not None else blur_frame
    diff_bgr = cv2.absdiff(blur_frame, blur_ref)
    diff_gray = cv2.cvtColor(diff_bgr, cv2.COLOR_BGR2GRAY)
    diff_gray = cv2.medianBlur(diff_gray, 3)
    diff_gray = cv2.bitwise_and(diff_gray, diff_gray, mask=table_mask_arr)
    _, diff_mask = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)
    k = max(3, cfg.kernel_size | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel)

    # Print per-detection diagnostics
    colors = []
    if dets:
        print(f"  {'idx':>5s}  {'center':>17s}  {'r':>5s}  {'color':<9s}  "
              f"{'hue':>6s}  {'S':>5s}  {'V':>5s}  {'diff_r':>6s}  {'skin':>6s}")
        print(f"  {'-'*75}")
    for i, d in enumerate(dets):
        c = _debug_print_detection(i, d, hsv, diff_mask, img, cfg)
        colors.append(c)

    color_summary = ", ".join(colors) if colors else "none"
    print(f"  Result: {len(dets)} ball(s) -> [{color_summary}]")

    # Save annotated image
    if debug_dir is not None:
        import os
        os.makedirs(debug_dir, exist_ok=True)
        vis = draw_detections(img, dets)
        if vis is not None:
            for idx_d, (det, col) in enumerate(zip(dets, colors)):
                dcx, dcy = int(round(det.center[0])), int(round(det.center[1]))
                dr = int(round(det.radius_px))
                cv2.putText(
                    vis, f"{idx_d}:{col}",
                    (dcx - dr, dcy - dr - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA,
                )
            for (px, py) in corners:
                cv2.circle(vis, (int(px), int(py)), 8, (255, 0, 255), -1)
            out_path = os.path.join(debug_dir, f"{basename}_hough_test.jpeg")
            cv2.imwrite(out_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    import os
    import sys
    import glob
    import time

    cfg = BallDetectionConfig()

    if len(sys.argv) > 1:
        sessions = [sys.argv[1]]
    else:
        runs_dir = os.path.join(os.path.dirname(__file__) or ".", "runs")
        if not os.path.isdir(runs_dir):
            print(f"ERROR: runs directory not found at {runs_dir}")
            sys.exit(1)
        sessions = sorted([
            os.path.join(runs_dir, d)
            for d in os.listdir(runs_dir)
            if os.path.isdir(os.path.join(runs_dir, d)) and not d.startswith(".")
        ])

    print("=" * 80)
    print("ball_detection_test.py  --  Hough-only detection on all runs")
    print(f"Sessions: {len(sessions)}")
    print("=" * 80)

    total_frames = 0
    total_dets = 0
    t_start = time.time()

    for session_dir in sessions:
        session_name = os.path.basename(session_dir)
        ref_path = os.path.join(session_dir, "ref.jpeg")
        if not os.path.isfile(ref_path):
            print(f"\n[SKIP] No ref.jpeg in {session_dir}")
            continue

        captures = sorted(glob.glob(os.path.join(session_dir, "capture_*.*")))
        if not captures:
            print(f"\n[SKIP] No captures in {session_dir}")
            continue

        try:
            corners = _load_table_corners_from_scene_understanding(ref_path)
        except Exception as e:
            print(f"\n[ERROR] Corner detection failed for {session_name}: {e}")
            continue
        if corners is None:
            print(f"\n[ERROR] No corners for {session_name}")
            continue
        corners = _order_points_clockwise(corners)

        expected_radius = _estimate_ball_radius_px(corners, 3.0, (120.0, 60.0))
        table_area = float(abs(cv2.contourArea(corners)))

        print(f"\n{'=' * 80}")
        print(f"SESSION: {session_name}  ({len(captures)} frame(s))")
        print(f"  Corners: {corners.astype(int).tolist()}")
        print(f"  Table area: {table_area:.0f} px^2")
        print(f"  Expected ball radius: {expected_radius:.1f} px" if expected_radius else "  Expected ball radius: unknown")

        debug_dir = os.path.join(session_dir, "debug")
        for img_path in captures:
            _process_single_frame(img_path, ref_path, corners, cfg, debug_dir)
            total_frames += 1
    elapsed = time.time() - t_start
    print(f"\n{'=' * 80}")
    print(f"Done. {total_frames} frames processed in {elapsed:.2f}s "
          f"({elapsed / max(total_frames, 1):.3f}s/frame)")
    print(f"Debug images saved to <session>/debug/*_hough_test.jpeg")
