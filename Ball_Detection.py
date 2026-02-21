"""
Classical pool ball detection for a single frame using OpenCV.

Assumptions:
- Input is a BGR frame (numpy array).
- Table is green and its 4 corners are provided.
- Balls are colored and non-green; detection is based on color segmentation and circle finding.
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
        green_min_sat: int = 60,
        green_min_val: int = 40,
        clahe_enabled: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: int = 8,
        kernel_size: int = 5,
        hough_dp: float = 1.2,
        hough_param1: float = 120.0,
        hough_param2: float = 14.0,
        min_circularity: float = 0.6,
        min_area_px: int = 60,
        non_green_ratio: float = 0.3,
        edge_margin: float = 0.03,
        hue_similarity_thresh: float = 10.0,
        # Color classification thresholds (HSV
        yellow_hue: Tuple[int, int] = (10, 20),
        blue_hue: Tuple[int, int] = (95, 115),
        purple_hue: Tuple[int, int] = (120, 140),
        red1_hue: Tuple[int, int] = (0, 10),
        red2_hue: Tuple[int, int] = (170, 179),
        white_sat_max: int = 40,
        white_val_min: int = 180,
        black_val_max: int = 75,
        black_pixel_val_max: int = 90,
        black_dark_ratio_min: float = 0.45,
        black_sat_for_dark: int = 110,
        black_inner_scale: float = 0.7,
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
        self.min_area_px = min_area_px
        # Non-green coverage threshold for circle filtering
        self.non_green_ratio = non_green_ratio
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
        self.white_sat_max = white_sat_max
        self.white_val_min = white_val_min
        self.black_val_max = black_val_max
        self.black_pixel_val_max = black_pixel_val_max
        self.black_dark_ratio_min = black_dark_ratio_min
        self.black_sat_for_dark = black_sat_for_dark
        self.black_inner_scale = black_inner_scale


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


def _dominant_green_hue(hsv: np.ndarray, mask: np.ndarray, min_sat: int, min_val: int) -> int:
    """Estimate dominant table hue"""
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    valid = (mask > 0) & (s >= min_sat) & (v >= min_val)
    if not np.any(valid):
        return 60  # typical green hue
    hist = cv2.calcHist([h], [0], valid.astype(np.uint8), [180], [0, 180])
    return int(np.argmax(hist))

def _green_mask(
    hsv: np.ndarray, table_mask: np.ndarray, cfg: BallDetectionConfig, hue: Optional[int] = None
) -> np.ndarray:
    hue = _dominant_green_hue(hsv, table_mask, cfg.green_min_sat, cfg.green_min_val) if hue is None else hue
    hw = cfg.green_hue_window
    lower1 = np.array([max(0, hue - hw), cfg.green_min_sat, cfg.green_min_val], dtype=np.uint8)
    upper1 = np.array([min(179, hue + hw), 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower1, upper1)

    if hue - hw < 0 or hue + hw > 179:
        # Wrap around hue
        lower2 = np.array([0, cfg.green_min_sat, cfg.green_min_val], dtype=np.uint8)
        upper2 = np.array([(hue + hw) % 180, 255, 255], dtype=np.uint8)
        lower3 = np.array([(hue - hw) % 180, cfg.green_min_sat, cfg.green_min_val], dtype=np.uint8)
        upper3 = np.array([179, 255, 255], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower2, upper2) | cv2.inRange(hsv, lower3, upper3)

    return cv2.bitwise_and(mask1, table_mask)

def _mean_hue_in_bbox(
    hsv: np.ndarray,
    bbox: BBox,
    min_sat: int,
    min_val: int,
) -> Optional[float]:
    '''
    calculate mean hue value  in a given bbox
    :param hsv:
    :param bbox:
    :param min_sat:
    :param min_val:
    :return:
    '''
    x, y, w, h = bbox
    h_img, w_img = hsv.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w_img, x + w)
    y1 = min(h_img, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    roi = hsv[y0:y1, x0:x1]
    h_ch = roi[:, :, 0]
    s_ch = roi[:, :, 1]
    v_ch = roi[:, :, 2]
    valid = (s_ch >= min_sat) & (v_ch >= min_val)
    if not np.any(valid):
        return None
    hues = h_ch[valid].astype(np.float32)
    angles = hues * (2.0 * np.pi / 180.0)
    mean_x = float(np.mean(np.cos(angles)))
    mean_y = float(np.mean(np.sin(angles)))
    if mean_x == 0.0 and mean_y == 0.0:
        return None
    angle = float(np.arctan2(mean_y, mean_x))
    if angle < 0.0:
        angle += 2.0 * np.pi
    return float(angle * (180.0 / (2.0 * np.pi)))


def _median_hue_in_bbox(
    hsv: np.ndarray,
    bbox: BBox,
    min_sat: int,
    min_val: int,
) -> Optional[float]:
    """
    Circular median hue in [0, 180) for robust color classification.
    Uses the sample hue that minimizes summed circular distance.
    """
    x, y, w, h = bbox
    h_img, w_img = hsv.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w_img, x + w)
    y1 = min(h_img, y + h)
    if x1 <= x0 or y1 <= y0:
        return None

    roi = hsv[y0:y1, x0:x1]
    h_ch = roi[:, :, 0]
    s_ch = roi[:, :, 1]
    v_ch = roi[:, :, 2]
    valid = (s_ch >= min_sat) & (v_ch >= min_val)
    if not np.any(valid):
        return None

    hues = h_ch[valid].astype(np.float32).ravel()
    if hues.size == 0:
        return None
    if hues.size == 1:
        return float(hues[0])

    # Reduce compute while keeping robustness for large ROIs.
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

def _mean_sv_in_bbox(
    hsv: np.ndarray,
    bbox: BBox,
) -> Optional[Tuple[float, float]]:
    x, y, w, h = bbox
    h_img, w_img = hsv.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w_img, x + w)
    y1 = min(h_img, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    roi = hsv[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    s_ch = roi[:, :, 1].astype(np.float32)
    v_ch = roi[:, :, 2].astype(np.float32)
    return float(np.mean(s_ch)), float(np.mean(v_ch))


def _dark_ratio_in_bbox(
    hsv: np.ndarray,
    bbox: BBox,
    val_max: int,
    sat_max: int = 255,
    inner_scale: float = 1.0,
) -> Optional[float]:
    x, y, w, h = bbox
    h_img, w_img = hsv.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w_img, x + w)
    y1 = min(h_img, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    roi = hsv[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    v = roi[:, :, 2]
    s = roi[:, :, 1]

    if inner_scale < 1.0:
        h_roi, w_roi = v.shape[:2]
        cx = (w_roi - 1) * 0.5
        cy = (h_roi - 1) * 0.5
        r = max(1.0, 0.5 * min(w_roi, h_roi) * inner_scale)
        yy, xx = np.ogrid[:h_roi, :w_roi]
        inner = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r * r)
    else:
        inner = np.ones(v.shape, dtype=bool)

    valid = inner & (s <= sat_max)
    total = int(np.count_nonzero(valid))
    if total == 0:
        return None
    dark = int(np.count_nonzero(valid & (v <= val_max)))
    return float(dark / float(total))

def _classify_color(
    hsv: np.ndarray,
    bbox: BBox,
    cfg: BallDetectionConfig,
) -> str:
    sv = _mean_sv_in_bbox(hsv, bbox)
    if sv is not None:
        mean_s, mean_v = sv
        if mean_s <= cfg.white_sat_max and mean_v >= cfg.white_val_min:
            return "white"
        dark_ratio = _dark_ratio_in_bbox(
            hsv,
            bbox,
            cfg.black_pixel_val_max,
            sat_max=cfg.black_sat_for_dark,
            inner_scale=cfg.black_inner_scale,
        )
        if mean_v <= cfg.black_val_max or (
            dark_ratio is not None and dark_ratio >= cfg.black_dark_ratio_min
        ):
            return "black"

    median_hue = _median_hue_in_bbox(hsv, bbox, cfg.green_min_sat, cfg.green_min_val)
    if median_hue is None:
        return "unknown"

    h = float(median_hue)
    if cfg.yellow_hue[0] <= h <= cfg.yellow_hue[1]:
        return "yellow"
    if cfg.blue_hue[0] <= h <= cfg.blue_hue[1]:
        return "blue"
    if cfg.purple_hue[0] <= h <= cfg.purple_hue[1]:
        return "purple"
    if (cfg.red1_hue[0] <= h <= cfg.red1_hue[1]) or (cfg.red2_hue[0] <= h <= cfg.red2_hue[1]):
        return "bordeaux"

    return "unknown"

def _estimate_ball_radius_px(
    corners: np.ndarray,
    ball_diameter_cm:None,
    table_size_cm: None
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


def _circle_non_green_ratio(
    non_green_mask: np.ndarray, center: Tuple[float, float], radius: float
) -> float:
    h, w = non_green_mask.shape
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
    region = non_green_mask[y0 : y1 + 1, x0 : x1 + 1]
    total = np.count_nonzero(roi)
    if total == 0:
        return 0.0
    return float(np.count_nonzero(region & roi) / total)


def _load_table_corners_from_scene_understanding(img_path: str) -> Optional[np.ndarray]:
    '''
    use scene_understanding module to get table corners for masking.
    :return:
    '''
    try:
        import Scene_Understanding as su
        corners = su.get_table_corners(img_path)
        corners = np.asarray(corners, dtype=np.float32)
        if corners.shape == (4, 2):
            return corners
    except Exception as e:
        raise RuntimeError("Failed to load table corners from Scene_Understanding.")
    return None

def detect_balls(
    frame_bgr: np.ndarray,
    table_corners: Optional[Iterable[Point]] = None,
    ref_path: str = "ref.jpeg",
    ball_diameter_cm: Optional[float] = None,
    table_size_cm: Optional[Tuple[float, float]] = None,
    config: Optional[BallDetectionConfig] = None,
    manual_corners: bool = False,
) -> List[BallDetection]:
    """
    Detect pool balls in a single frame
    Args:
        frame_bgr: Input image (BGR).
        table_corners: 4 corner points of the table in image coordinates.
        ball_diameter_cm: Optional known ball diameter in cm.
        table_size_cm: Optional table size (width_cm, height_cm).
        config: Optional config parameters.
    Returns:
        List of BallDetection objects with center, radius, and bbox.
    """
    cfg = config or BallDetectionConfig()
    if frame_bgr is None or frame_bgr.size == 0:
        return []
    elif table_corners is None:
        corners_arr = _load_table_corners_from_scene_understanding(ref_path)
        if corners_arr is None:
            raise ValueError("table_corners not provided and Scene_Understanding failed to detect corners.")
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
    # Mild blur on both sides suppresses per-frame sensor noise / compression
    # artifacts that cause flickering false positives in video.
    blur_frame = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
    blur_ref   = cv2.GaussianBlur(ref_bgr,   (5, 5), 0)
    diff_bgr   = cv2.absdiff(blur_frame, blur_ref)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    if cfg.clahe_enabled:
        clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip_limit,
            tileGridSize=(cfg.clahe_grid_size, cfg.clahe_grid_size),
        )
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    k = max(3, cfg.kernel_size | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    # Prepare grayscale for Hough
    gray = cv2.cvtColor(diff_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    gray = cv2.bitwise_and(gray, gray, mask=table_mask)
    _, diff_mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
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
        circles = np.squeeze(circles, axis=0)
        for x, y, r in circles:
            if _circle_non_green_ratio(diff_mask, (x, y), r) < cfg.non_green_ratio:
                continue#ignore green circles which are likely to be the noise, not balls.
            x0 = int(round(x - r))
            y0 = int(round(y - r))
            w = int(round(2 * r))
            h = int(round(2 * r))
            detections.append(
                BallDetection(center=(float(x), float(y)), radius_px=float(r), bbox=(x0, y0, w, h))
            )

    # Fallback to contour-based detection on diff mask
    if not detections:
        contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < cfg.min_area_px:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 1e-3:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < cfg.min_circularity:
                continue
            (x, y), r = cv2.minEnclosingCircle(cnt)
            if expected_radius is not None:
                if r < 0.6 * expected_radius or r > 1.5 * expected_radius:
                    continue
            x0 = int(round(x - r))
            y0 = int(round(y - r))
            w = int(round(2 * r))
            h = int(round(2 * r))
            detections.append(
                BallDetection(center=(float(x), float(y)), radius_px=float(r), bbox=(x0, y0, w, h))
            )

    # Filter detections near the table edges using normalized table coordinates
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

    return filtered


def detect_balls_with_color(
    frame_bgr: np.ndarray,
    table_corners: Optional[Iterable[Point]] = None,
    ref_path: str = "ref.jpeg",
    ball_diameter_cm: Optional[float] = None,
    table_size_cm: Optional[Tuple[float, float]] = None,
    config: Optional[BallDetectionConfig] = None,
    manual_corners: bool = False,
) -> np.ndarray:
    """
    Detect pool balls and classify colors.
    Returns a numpy array of shape (N, 3): [x, y, color_string].
    """
    cfg = config or BallDetectionConfig()
    detections = detect_balls(
        frame_bgr=frame_bgr,
        table_corners=table_corners,
        ref_path=ref_path,
        ball_diameter_cm=ball_diameter_cm,
        table_size_cm=table_size_cm,
        config=cfg,
        manual_corners=manual_corners,
    )
    if frame_bgr is None or frame_bgr.size == 0:
        return np.empty((0, 3), dtype=object)

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    out: List[List[object]] = []
    for d in detections:
        color = _classify_color(hsv, d.bbox, cfg)
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
    """
    Create a new image with ball centroids and bounding boxes drawn.

    Args:
        frame_bgr: Input image (BGR).
        detections: Iterable of BallDetection.
        color_bbox: BGR color for bounding boxes.
        color_center: BGR color for centroids.

    Returns:
        Annotated BGR image.
    """
    if frame_bgr is None:
        return None
    vis = frame_bgr.copy()
    for det in detections:
        x, y, w, h = det.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), color_bbox, 2)
        cx, cy = det.center
        cv2.circle(vis, (int(round(cx)), int(round(cy))), 4, color_center, -1)
    return vis


if __name__ == "__main__":
    ref_path="ref.jpeg"
    img_path="img_1.jpeg"

    img = cv2.imread(img_path)
    if img is None:
        raise SystemExit("Failed to read pool_table.jpeg")
    cfg = BallDetectionConfig()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    corners = _load_table_corners_from_scene_understanding(ref_path)
    if corners is None:
        raise SystemExit("Failed to load corners from Scene_Understanding")
    dets = detect_balls(img, table_corners=corners, ref_path=ref_path, config=cfg)
    colored = detect_balls_with_color(img, table_corners=corners, ref_path=ref_path, config=cfg)
    vis = draw_detections(img, dets)
    for (x, y) in corners:
        cv2.circle(vis, (int(x), int(y)), 8, (255, 0, 255), -1)
    for i, d in enumerate(dets):
        x, y, w, h = d.bbox
        label_pos = (max(0, x), max(15, y - 6))
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
    cv2.imwrite("pool_table_annotated.jpeg", vis)
    print(f"detections: {len(dets)}")
    for i, d in enumerate(dets):
        median_hue = _median_hue_in_bbox(hsv, d.bbox, cfg.green_min_sat, cfg.green_min_val)
        print(i, d.center, d.bbox, d.radius_px, "median_hue", median_hue)
    print("colored detections:")
    print(colored)
