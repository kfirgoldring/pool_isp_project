"""
BILLIARDS PROJECTION SYSTEM - COMPLETE CALIBRATION TOOLKIT
All-in-one merged file containing all components for projector calibration

This file contains:
- Configuration settings
- Pattern generation (colored circles, checkerboard, test patterns)
- Circle detection algorithms
- Camera calibration (manual 4-point)
- Projector calibration (colored circles and checkerboard methods)
- Detection testing tools
- Calibration verification
- Component tests
- Main menu system

Usage:
    python billiards_calibration_merged.py
"""

# ============================================================================
# IMPORTS
# ============================================================================

import cv2
import numpy as np
import colorsys
import os
import time
from typing import Tuple, List, Dict, Optional


# ============================================================================
# CONFIGURATION
# ============================================================================

# Hardware settings
CAMERA_INDEX = 0
PROJECTOR_WIDTH = 1920
PROJECTOR_HEIGHT = 1080
MAIN_MONITOR_WIDTH = 1920  # Your laptop/main screen width (for window positioning)

# Calibration pattern settings - will try multiple configurations
PATTERN_CONFIGS = [
    {'rows': 4, 'cols': 6, 'radius': 40, 'margin': 50},   # First try: 24 circles
    {'rows': 3, 'cols': 5, 'radius': 50, 'margin': 80},   # Second try: 15 circles, larger
    {'rows': 5, 'cols': 7, 'radius': 35, 'margin': 40},   # Third try: 35 circles, smaller
    {'rows': 3, 'cols': 4, 'radius': 60, 'margin': 100},  # Fourth try: 12 circles, very large
]

INVERT_PATTERN = True  # Black circles on white = brighter projection

# Detection settings
MIN_CIRCLE_RADIUS = 8
MAX_CIRCLE_RADIUS = 100
MIN_REQUIRED_MATCHES = 12  # Minimum circles needed for calibration

# Table dimensions (in cm or any unit)
TABLE_WIDTH_CM = 122.0
TABLE_HEIGHT_CM = 61.0

# Table boundary margin (in table coordinates) - reject circles near edge
TABLE_MARGIN_CM = 5.0

# Calibration data storage
CAMERA_CALIB_FILE = 'data/calibration/camera_homography.npy'
PROJECTOR_CALIB_FILE = 'data/calibration/projector_homography.npy'


# ============================================================================
# PATTERN GENERATOR
# ============================================================================

class PatternGenerator:
    def __init__(self, width: int = PROJECTOR_WIDTH,
                 height: int = PROJECTOR_HEIGHT):
        self.width = width
        self.height = height

    def generate_unique_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """
        Generate n visually distinct colors
        Returns BGR tuples for OpenCV
        """
        colors = []
        for i in range(n):
            # Use HSV color space for even distribution
            hue = i / n
            saturation = 0.9
            value = 0.9

            # Convert to RGB
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)

            # Convert to BGR (OpenCV format) and scale to 0-255
            bgr = (int(b * 255), int(g * 255), int(r * 255))
            colors.append(bgr)

        return colors

    def generate_colored_circle_grid(self, rows: int, cols: int,
                                    radius: int, margin: int,
                                    invert: bool = INVERT_PATTERN) -> Tuple[np.ndarray, Dict[Tuple[int, int], Tuple[int, int, int]]]:
        """
        Generate grid of circles, each with unique color

        Returns:
            pattern: Image with colored circles
            circle_map: Dict mapping (x, y) center -> (B, G, R) color
        """
        # Background
        if invert:
            pattern = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        else:
            pattern = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Calculate positions
        usable_width = self.width - 2 * margin
        usable_height = self.height - 2 * margin

        x_spacing = usable_width / (cols - 1) if cols > 1 else 0
        y_spacing = usable_height / (rows - 1) if rows > 1 else 0

        # Generate unique colors
        n_circles = rows * cols
        colors = self.generate_unique_colors(n_circles)

        circle_map = {}
        idx = 0

        for row in range(rows):
            for col in range(cols):
                x = int(margin + col * x_spacing)
                y = int(margin + row * y_spacing)

                color = colors[idx]
                circle_map[(x, y)] = color

                # Draw colored circle with white/black border for visibility
                if invert:
                    cv2.circle(pattern, (x, y), radius, color, -1)
                    cv2.circle(pattern, (x, y), radius, (0, 0, 0), 2)  # Black border
                else:
                    cv2.circle(pattern, (x, y), radius, color, -1)
                    cv2.circle(pattern, (x, y), radius, (255, 255, 255), 2)  # White border

                idx += 1

        return pattern, circle_map

    def generate_test_pattern(self) -> np.ndarray:
        """Generate bright test pattern to check projector alignment"""
        pattern = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

        # Draw border
        cv2.rectangle(pattern, (10, 10), (self.width-10, self.height-10),
                     (255, 0, 0), 5)

        # Draw crosshair at center
        cx, cy = self.width // 2, self.height // 2
        cv2.line(pattern, (cx - 100, cy), (cx + 100, cy), (0, 0, 255), 3)
        cv2.line(pattern, (cx, cy - 100), (cx, cy + 100), (0, 0, 255), 3)

        # Add text
        cv2.putText(pattern, "PROJECTOR TEST", (cx - 200, cy - 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

        return pattern


# ============================================================================
# CIRCLE DETECTOR
# ============================================================================

class CircleDetector:
    def __init__(self, min_radius: int = MIN_CIRCLE_RADIUS,
                 max_radius: int = MAX_CIRCLE_RADIUS):
        self.min_radius = min_radius
        self.max_radius = max_radius

    def detect_colored_circles(self, image: np.ndarray,
                               invert: bool = INVERT_PATTERN) -> List[Tuple[Tuple[float, float], Tuple[int, int, int]]]:
        """
        Detect circles and extract their colors

        Returns:
            List of ((x, y), (B, G, R)) tuples
        """
        # Convert to grayscale for circle detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Adaptive threshold
        if invert:
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 51, 10)
        else:
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 51, -10)

        # Morphological cleanup
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        circles = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            min_area = np.pi * self.min_radius ** 2
            max_area = np.pi * self.max_radius ** 2

            if min_area < area < max_area:
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)

                    if circularity > 0.5:  # Reasonably circular
                        # Calculate centroid
                        M = cv2.moments(contour)
                        if M['m00'] > 0:
                            cx = M['m10'] / M['m00']
                            cy = M['m01'] / M['m00']

                            # Extract color at center
                            color = self.extract_circle_color(image, (cx, cy), contour)

                            if color is not None:
                                circles.append(((cx, cy), color))

        return circles

    def extract_circle_color(self, image: np.ndarray,
                            center: Tuple[float, float],
                            contour: np.ndarray) -> Tuple[int, int, int]:
        """
        Extract the dominant color of a circle
        """
        # Create mask for this circle
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # Get mean color within circle
        mean_color = cv2.mean(image, mask=mask)[:3]

        return (int(mean_color[0]), int(mean_color[1]), int(mean_color[2]))

    def match_circles_by_color(self,
                               detected_circles: List[Tuple[Tuple[float, float], Tuple[int, int, int]]],
                               projected_map: Dict[Tuple[int, int], Tuple[int, int, int]],
                               color_tolerance: int = 50) -> List[Tuple[Tuple[int, int], Tuple[float, float]]]:
        """
        Match detected circles to projected circles by color

        Returns:
            List of (projector_center, camera_center) pairs
        """
        matches = []

        for camera_center, camera_color in detected_circles:
            best_match = None
            best_distance = float('inf')

            for proj_center, proj_color in projected_map.items():
                # Calculate color distance (Euclidean in BGR space)
                color_dist = np.sqrt(
                    (camera_color[0] - proj_color[0])**2 +
                    (camera_color[1] - proj_color[1])**2 +
                    (camera_color[2] - proj_color[2])**2
                )

                if color_dist < best_distance and color_dist < color_tolerance:
                    best_distance = color_dist
                    best_match = proj_center

            if best_match is not None:
                matches.append((best_match, camera_center))

        return matches

    def visualize_colored_detection(self, image: np.ndarray,
                                   circles: List[Tuple[Tuple[float, float], Tuple[int, int, int]]]) -> np.ndarray:
        """Draw detected colored circles on image"""
        vis = image.copy()

        for i, ((x, y), color) in enumerate(circles):
            # Draw circle with detected color
            cv2.circle(vis, (int(x), int(y)), 8, color, 2)
            # Draw index
            cv2.putText(vis, str(i), (int(x)+15, int(y)+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return vis


# ============================================================================
# MANUAL CAMERA CALIBRATOR
# ============================================================================

class ManualCameraCalibrator:
    def __init__(self, table_width: float = TABLE_WIDTH_CM,
                 table_height: float = TABLE_HEIGHT_CM):
        self.table_width = table_width
        self.table_height = table_height
        self.image_points = []
        self.homography = None
        self.dragging_idx = -1  # Index of point being dragged (-1 = none)
        self.hover_idx = -1

    def distance(self, p1, p2):
        """Euclidean distance between two points"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def find_closest_point(self, x, y, threshold=15):
        """Find closest point within threshold"""
        for i, pt in enumerate(self.image_points):
            if self.distance((x, y), pt) < threshold:
                return i
        return -1

    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback with drag support"""

        if event == cv2.EVENT_MOUSEMOVE:
            # Update hover
            self.hover_idx = self.find_closest_point(x, y)

            # Update dragged point position
            if self.dragging_idx >= 0:
                self.image_points[self.dragging_idx] = (x, y)

        elif event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicking on existing point
            idx = self.find_closest_point(x, y)
            if idx >= 0:
                self.dragging_idx = idx
                print(f"Dragging point {idx+1}")
            elif len(self.image_points) < 4:
                # Add new point
                self.image_points.append((x, y))
                print(f"Added point {len(self.image_points)}: ({x}, {y})")

        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging_idx >= 0:
                pt = self.image_points[self.dragging_idx]
                print(f"Point {self.dragging_idx+1} positioned at: ({pt[0]}, {pt[1]})")
                self.dragging_idx = -1

    def calibrate(self, camera_index: int = 0) -> Optional[np.ndarray]:
        """
        Manual calibration by clicking 4 table corners
        """
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print(f"Error: Cannot open camera {camera_index}")
            return None

        # Capture frame
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("Error: Cannot read from camera")
            return None

        window_name = 'Camera Calibration'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print("\n=== Camera Calibration ===")
        print("Click 4 table corners:")
        print("  1. Top-left")
        print("  2. Top-right")
        print("  3. Bottom-right")
        print("  4. Bottom-left")
        print("\nClick to add point | Click & drag to move point")
        print("R=reset | D=delete last | SPACE=finish | ESC=cancel\n")

        while True:
            display = frame.copy()

            # Draw all points
            for i, pt in enumerate(self.image_points):
                # Choose color
                if i == self.dragging_idx:
                    color = (255, 255, 0)  # Cyan when dragging
                elif i == self.hover_idx:
                    color = (0, 255, 255)  # Yellow when hovering
                else:
                    color = (0, 255, 0)  # Green normally

                # Draw point
                cv2.circle(display, pt, 8, color, -1)
                cv2.circle(display, pt, 15, color, 2)  # Outer ring

                # Label
                cv2.putText(display, str(i+1), (pt[0]+20, pt[1]+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Draw polygon if we have multiple points
            if len(self.image_points) >= 2:
                pts = np.array(self.image_points, np.int32)
                cv2.polylines(display, [pts], len(self.image_points) == 4,
                            (255, 0, 0), 2)

            # Status text
            cv2.putText(display, f"Points: {len(self.image_points)}/4",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            if len(self.image_points) == 4:
                cv2.putText(display, "Press SPACE to finish", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow(window_name, display)

            # Check window closure
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("Window closed")
                    cv2.destroyAllWindows()
                    return None
            except:
                pass

            key = cv2.waitKey(30) & 0xFF

            if key == ord('r') or key == ord('R'):
                self.image_points = []
                self.dragging_idx = -1
                print("Reset all points")

            elif key == ord('d') or key == ord('D') or key == 8:  # D or Backspace
                if self.image_points:
                    removed = self.image_points.pop()
                    print(f"Removed point. Remaining: {len(self.image_points)}")

            elif key == 27:  # ESC
                print("Cancelled")
                cv2.destroyAllWindows()
                return None

            elif key == 32 and len(self.image_points) == 4:  # SPACE
                break

        cv2.destroyAllWindows()

        # Compute homography
        src_points = np.array(self.image_points, dtype=np.float32)
        dst_points = np.array([
            [0, 0],
            [self.table_width, 0],
            [self.table_width, self.table_height],
            [0, self.table_height]
        ], dtype=np.float32)

        self.homography, _ = cv2.findHomography(src_points, dst_points)
        print("✓ Homography computed")

        return self.homography

    def save(self, filepath: str = CAMERA_CALIB_FILE):
        """Save homography"""
        if self.homography is None:
            print("Error: No homography to save")
            return
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.save(filepath, self.homography)
        print(f"✓ Saved to {filepath}")

    @staticmethod
    def load(filepath: str = CAMERA_CALIB_FILE) -> np.ndarray:
        """Load homography"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Calibration file not found: {filepath}")
        return np.load(filepath)


# ============================================================================
# DETECTION TESTER
# ============================================================================

class DetectionTester:
    def __init__(self):
        self.detector = CircleDetector()
        self.pattern_gen = PatternGenerator()
        self.camera_homography = None
        self.projector_window = 'Projector'

    def load_camera_calibration(self) -> bool:
        """Load camera calibration"""
        try:
            self.camera_homography = ManualCameraCalibrator.load()
            print("✓ Loaded camera calibration")
            return True
        except FileNotFoundError:
            print("✗ Camera not calibrated! Run option 1 first.")
            return False
        except Exception as e:
            print(f"✗ Error loading calibration: {e}")
            return False

    def setup_projector_window(self):
        """Create fullscreen window on projector"""
        cv2.namedWindow(self.projector_window, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.projector_window, cv2.WND_PROP_FULLSCREEN,
                             cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow(self.projector_window, MAIN_MONITOR_WIDTH, 0)

    def get_table_boundary_in_camera(self):
        """Get table corners transformed to camera coordinates"""
        table_corners = np.array([
            [0, 0],
            [TABLE_WIDTH_CM, 0],
            [TABLE_WIDTH_CM, TABLE_HEIGHT_CM],
            [0, TABLE_HEIGHT_CM]
        ], dtype=np.float32)

        try:
            inv_H = np.linalg.inv(self.camera_homography)
            camera_corners = cv2.perspectiveTransform(
                table_corners.reshape(-1, 1, 2), inv_H
            ).reshape(-1, 2)
            return camera_corners.astype(np.int32)
        except:
            return None

    def warp_to_table_view(self, frame):
        """Warp camera image to top-down table view"""
        # Output dimensions for warped view (maintain aspect ratio)
        aspect_ratio = TABLE_WIDTH_CM / TABLE_HEIGHT_CM
        warp_height = 600
        warp_width = int(warp_height * aspect_ratio)

        # Destination points (rectangle in output image)
        dst_points = np.array([
            [0, 0],
            [warp_width, 0],
            [warp_width, warp_height],
            [0, warp_height]
        ], dtype=np.float32)

        # Source points (table corners in camera view)
        table_boundary = self.get_table_boundary_in_camera()
        if table_boundary is None:
            return None

        src_points = table_boundary.astype(np.float32)

        # Compute perspective transform
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        # Warp the image
        warped = cv2.warpPerspective(frame, M, (warp_width, warp_height))

        return warped

    def transform_to_table(self, camera_point):
        """Transform camera point to table coordinates"""
        try:
            pt = np.array([[camera_point]], dtype=np.float32)
            table_pt = cv2.perspectiveTransform(pt, self.camera_homography)[0][0]
            return table_pt
        except:
            return None

    def is_in_table(self, table_point, margin=5):
        """Check if point is within table bounds"""
        if table_point is None:
            return False
        x, y = table_point
        return (margin <= x <= TABLE_WIDTH_CM - margin and
                margin <= y <= TABLE_HEIGHT_CM - margin)

    def run_live_test(self, camera_index: int = CAMERA_INDEX):
        """Run live detection test with projection"""

        # Load calibration
        if not self.load_camera_calibration():
            return

        # Setup projector
        self.setup_projector_window()

        # Open camera
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"✗ Cannot open camera {camera_index}")
            return

        print("\n=== Live Detection Test with Projection ===")
        print("\nThis shows:")
        print("  1. Raw camera view with detection overlay")
        print("  2. Perspective-corrected table view (top-down)")
        print("\nControls:")
        print("  1-4     - Switch pattern (grid size)")
        print("  I       - Toggle invert (bright/dark circles)")
        print("  P       - Toggle projection ON/OFF")
        print("  SPACE   - Pause/Resume")
        print("  Q/ESC   - Quit\n")

        camera_window = 'Camera View (Raw)'
        table_window = 'Table View (Corrected)'
        cv2.namedWindow(camera_window)
        cv2.namedWindow(table_window)

        # State
        paused = False
        projection_on = True
        pattern_idx = 0
        invert = INVERT_PATTERN
        current_pattern = None
        circle_map = {}

        # Generate initial pattern
        config = PATTERN_CONFIGS[pattern_idx]
        current_pattern, circle_map = self.pattern_gen.generate_colored_circle_grid(
            config['rows'], config['cols'], config['radius'], config['margin'], invert
        )

        frozen_frame = None

        while True:
            # Capture frame
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("✗ Cannot read from camera")
                    break
                frozen_frame = frame.copy()
            else:
                frame = frozen_frame.copy()

            # Project pattern (or black screen)
            if projection_on and current_pattern is not None:
                cv2.imshow(self.projector_window, current_pattern)
            else:
                black = np.zeros((PROJECTOR_HEIGHT, PROJECTOR_WIDTH, 3), dtype=np.uint8)
                cv2.imshow(self.projector_window, black)

            # === Camera View ===
            camera_display = frame.copy()

            # Draw table boundary
            table_boundary = self.get_table_boundary_in_camera()
            if table_boundary is not None:
                cv2.polylines(camera_display, [table_boundary], True, (0, 255, 0), 3)

                # Label corners
                labels = ["TL", "TR", "BR", "BL"]
                for pt, label in zip(table_boundary, labels):
                    cv2.putText(camera_display, label, (pt[0]+10, pt[1]+10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Detect circles
            detected_circles = self.detector.detect_colored_circles(frame, invert=invert)

            # Process detected circles
            in_table = 0
            out_table = 0

            for (x, y), color in detected_circles:
                table_pt = self.transform_to_table((x, y))

                if table_pt is not None and self.is_in_table(table_pt):
                    # Inside table - green circle
                    cv2.circle(camera_display, (int(x), int(y)), 12, (0, 255, 0), 2)
                    cv2.circle(camera_display, (int(x), int(y)), 3, (0, 255, 0), -1)
                    in_table += 1

                    # Show table coordinates
                    coord_text = f"({table_pt[0]:.0f},{table_pt[1]:.0f})"
                    cv2.putText(camera_display, coord_text, (int(x)+15, int(y)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                else:
                    # Outside table - red X
                    cv2.circle(camera_display, (int(x), int(y)), 12, (0, 0, 255), 2)
                    cv2.line(camera_display, (int(x)-10, int(y)-10),
                            (int(x)+10, int(y)+10), (0, 0, 255), 2)
                    cv2.line(camera_display, (int(x)-10, int(y)+10),
                            (int(x)+10, int(y)-10), (0, 0, 255), 2)
                    out_table += 1

            # Status overlay for camera view
            y_pos = 30
            cv2.putText(camera_display, f"Detected: {len(detected_circles)} | In table: {in_table} | Out: {out_table}",
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_pos += 25

            config_text = f"Pattern: {config['rows']}x{config['cols']} ({pattern_idx+1}/{len(PATTERN_CONFIGS)})"
            cv2.putText(camera_display, config_text,
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_pos += 25

            mode_text = f"Mode: {'dark' if invert else 'bright'} circles | Projection: {'ON' if projection_on else 'OFF'}"
            cv2.putText(camera_display, mode_text,
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if paused:
                cv2.putText(camera_display, "PAUSED", (10, y_pos+25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # === Table View (Warped) ===
            table_view = self.warp_to_table_view(frame)

            if table_view is not None:
                # Detect circles in warped view too
                warped_circles = self.detector.detect_colored_circles(table_view, invert=invert)

                # Draw detected circles on warped view
                for (x, y), color in warped_circles:
                    cv2.circle(table_view, (int(x), int(y)), 10, (0, 255, 0), 2)
                    cv2.circle(table_view, (int(x), int(y)), 3, (0, 255, 0), -1)

                # Add scale reference
                cv2.putText(table_view, f"Table: {TABLE_WIDTH_CM:.0f} x {TABLE_HEIGHT_CM:.0f} cm",
                           (10, table_view.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.putText(table_view, f"Circles detected: {len(warped_circles)}",
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow(table_window, table_view)

            cv2.imshow(camera_window, camera_display)

            # Check window closure
            try:
                if (cv2.getWindowProperty(camera_window, cv2.WND_PROP_VISIBLE) < 1 or
                    cv2.getWindowProperty(table_window, cv2.WND_PROP_VISIBLE) < 1):
                    break
            except:
                pass

            key = cv2.waitKey(30) & 0xFF

            if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
                break

            elif key == 32:  # SPACE
                paused = not paused
                print("PAUSED" if paused else "RESUMED")

            elif key == ord('i') or key == ord('I'):
                invert = not invert
                print(f"Switched to: {'dark' if invert else 'bright'} circles")
                # Regenerate pattern with new invert setting
                current_pattern, circle_map = self.pattern_gen.generate_colored_circle_grid(
                    config['rows'], config['cols'], config['radius'], config['margin'], invert
                )

            elif key == ord('p') or key == ord('P'):
                projection_on = not projection_on
                print(f"Projection: {'ON' if projection_on else 'OFF'}")

            elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                # Switch pattern
                new_idx = int(chr(key)) - 1
                if 0 <= new_idx < len(PATTERN_CONFIGS):
                    pattern_idx = new_idx
                    config = PATTERN_CONFIGS[pattern_idx]
                    current_pattern, circle_map = self.pattern_gen.generate_colored_circle_grid(
                        config['rows'], config['cols'], config['radius'], config['margin'], invert
                    )
                    print(f"Switched to pattern {pattern_idx+1}: {config['rows']}x{config['cols']}")

        cap.release()
        cv2.destroyAllWindows()
        print("✓ Test complete")


# ============================================================================
# PROJECTOR CALIBRATOR
# ============================================================================

class ProjectorCalibrator:
    def __init__(self):
        self.pattern_gen = PatternGenerator()
        self.detector = CircleDetector()
        self.projector_homography = None
        self.camera_homography = None
        self.projector_window = 'Projector'

    def setup_projector_window(self):
        """Create fullscreen window on projector"""
        cv2.namedWindow(self.projector_window, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.projector_window, cv2.WND_PROP_FULLSCREEN,
                             cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow(self.projector_window, MAIN_MONITOR_WIDTH, 0)

    def is_point_in_table(self, point: Tuple[float, float],
                         margin: float = TABLE_MARGIN_CM) -> bool:
        """Check if a point (in table coordinates) is within table bounds"""
        x, y = point
        return (margin <= x <= TABLE_WIDTH_CM - margin and
                margin <= y <= TABLE_HEIGHT_CM - margin)

    def filter_points_by_table(self,
                               camera_points: List[Tuple[float, float]]) -> List[bool]:
        """Return mask of which camera points are within table bounds"""
        points_array = np.array(camera_points, dtype=np.float32).reshape(-1, 1, 2)
        table_points = cv2.perspectiveTransform(points_array, self.camera_homography)
        table_points = table_points.reshape(-1, 2)

        mask = [self.is_point_in_table(pt) for pt in table_points]

        return mask, table_points

    def capture_best_frame(self, cap, num_frames=10, settle_time=2.0):
        """
        Capture multiple frames and return the best one
        Gives camera time to adjust exposure/white balance
        """
        print(f"  Waiting {settle_time}s for camera to settle...")
        time.sleep(settle_time)

        frames = []
        print(f"  Capturing {num_frames} frames...")

        for i in range(num_frames):
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            time.sleep(0.1)  # 100ms between frames

        if not frames:
            return None

        # Pick frame with best average brightness (middle exposure)
        brightnesses = [np.mean(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in frames]
        median_brightness = np.median(brightnesses)
        best_idx = min(range(len(brightnesses)),
                      key=lambda i: abs(brightnesses[i] - median_brightness))

        print(f"  Selected frame {best_idx+1}/{num_frames} (brightness: {brightnesses[best_idx]:.1f})")
        return frames[best_idx]

    def try_calibration_pattern(self, config: Dict, camera_index: int) -> Optional[Tuple]:
        """Try one calibration pattern configuration"""
        rows = config['rows']
        cols = config['cols']
        radius = config['radius']
        margin = config['margin']

        print(f"\nTrying pattern: {rows}x{cols} grid, radius={radius}px, margin={margin}px")

        # Generate colored circle pattern
        pattern, circle_map = self.pattern_gen.generate_colored_circle_grid(
            rows, cols, radius, margin
        )

        print(f"  Projecting {len(circle_map)} colored circles...")

        # Project pattern
        cv2.imshow(self.projector_window, pattern)
        cv2.waitKey(100)  # Small delay for projection to start

        # Open camera
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return None

        # Capture best frame (with auto-exposure adjustment)
        frame = self.capture_best_frame(cap, num_frames=15, settle_time=2.5)
        cap.release()

        if frame is None:
            print("  ✗ Failed to capture frame")
            return None

        # Detect colored circles
        print("  Detecting circles...")
        detected_circles = self.detector.detect_colored_circles(frame)
        print(f"  Detected {len(detected_circles)} circles")

        if len(detected_circles) == 0:
            print("  ✗ No circles detected!")
            print("  TIP: Check lighting, focus, and try pressing 'I' in detection test")
            return None

        # Match by color
        print("  Matching by color...")
        matches = self.detector.match_circles_by_color(detected_circles, circle_map,
                                                       color_tolerance=80)
        print(f"  Matched {len(matches)} circles")

        if len(matches) < MIN_REQUIRED_MATCHES:
            print(f"  ✗ Too few matches (need {MIN_REQUIRED_MATCHES})")
            return None

        # Separate projector and camera points
        projector_points = [proj_pt for proj_pt, _ in matches]
        camera_points = [cam_pt for _, cam_pt in matches]

        # Filter by table bounds
        print("  Filtering by table bounds...")
        in_table_mask, table_points = self.filter_points_by_table(camera_points)

        # Keep only points within table
        valid_projector = [p for p, valid in zip(projector_points, in_table_mask) if valid]
        valid_table = [p for p, valid in zip(table_points, in_table_mask) if valid]

        print(f"  {len(valid_projector)} circles within table bounds")

        if len(valid_projector) < MIN_REQUIRED_MATCHES:
            print(f"  ✗ Too few points in table (need {MIN_REQUIRED_MATCHES})")
            return None

        # Visualize
        vis_circles = [(cam_pt, color) for (_, cam_pt), color in
                       zip(matches, [circle_map[proj_pt] for proj_pt, _ in matches])]
        vis = self.detector.visualize_colored_detection(frame, vis_circles)

        # Mark points outside table
        for i, (in_table, cam_pt) in enumerate(zip(in_table_mask, camera_points)):
            if not in_table:
                cv2.circle(vis, (int(cam_pt[0]), int(cam_pt[1])), 10, (0, 0, 255), 2)
                cv2.line(vis, (int(cam_pt[0])-10, int(cam_pt[1])-10),
                        (int(cam_pt[0])+10, int(cam_pt[1])+10), (0, 0, 255), 2)
                cv2.line(vis, (int(cam_pt[0])-10, int(cam_pt[1])+10),
                        (int(cam_pt[0])+10, int(cam_pt[1])-10), (0, 0, 255), 2)

        return (valid_projector, valid_table, vis)

    def calibrate(self, camera_index: int = CAMERA_INDEX) -> Optional[np.ndarray]:
        """Perform automatic projector calibration with auto-retry"""

        # Load camera calibration
        if not os.path.exists(CAMERA_CALIB_FILE):
            print("Error: Camera not calibrated! Run camera calibration first.")
            return None

        self.camera_homography = ManualCameraCalibrator.load()
        print("✓ Loaded camera calibration")

        # Setup projector
        self.setup_projector_window()

        print("\n=== Automatic Projector Calibration ===")
        print("Will try multiple patterns automatically...")
        print("Camera will take time to adjust exposure - be patient!")

        # Try each configuration
        for i, config in enumerate(PATTERN_CONFIGS):
            print(f"\n--- Attempt {i+1}/{len(PATTERN_CONFIGS)} ---")

            result = self.try_calibration_pattern(config, camera_index)

            if result is not None:
                projector_points, table_points, vis = result

                # Show detection result
                detection_window = f'Detection Result - {len(projector_points)} matches (SPACE=accept, R=retry, ESC=cancel)'
                cv2.imshow(detection_window, vis)

                print(f"\n✓ Success! Found {len(projector_points)} valid points")
                print("Green circles: detected | Red X: outside table")
                print("\nControls:")
                print("  SPACE - Accept and compute homography")
                print("  R     - Try next pattern")
                print("  ESC   - Cancel calibration")

                # Wait for user decision
                while True:
                    try:
                        if cv2.getWindowProperty(detection_window, cv2.WND_PROP_VISIBLE) < 1:
                            print("Cancelled (window closed)")
                            cv2.destroyAllWindows()
                            return None
                    except:
                        pass

                    key = cv2.waitKey(50) & 0xFF

                    if key == 32:  # SPACE - accept
                        # Compute homography
                        proj_array = np.array(projector_points, dtype=np.float32)
                        table_array = np.array(table_points, dtype=np.float32)

                        self.projector_homography, _ = cv2.findHomography(proj_array, table_array)

                        print("\n✓ Projector calibration complete!")
                        cv2.destroyAllWindows()
                        return self.projector_homography

                    elif key == ord('r') or key == ord('R'):  # Retry with next pattern
                        cv2.destroyWindow(detection_window)
                        break

                    elif key == 27:  # ESC - cancel
                        print("Cancelled (ESC)")
                        cv2.destroyAllWindows()
                        return None

        # All patterns failed
        print("\n✗ All patterns failed!")
        print("Tips:")
        print("  - Use Detection Test (option 2) to debug")
        print("  - Turn off room lights")
        print("  - Try checkerboard pattern (option 7)")

        cv2.destroyAllWindows()
        return None

    def save(self, filepath: str = PROJECTOR_CALIB_FILE):
        """Save projector homography"""
        if self.projector_homography is None:
            print("Error: No homography to save")
            return
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.save(filepath, self.projector_homography)
        print(f"✓ Saved to {filepath}")

    @staticmethod
    def load(filepath: str = PROJECTOR_CALIB_FILE) -> np.ndarray:
        """Load projector homography"""
        return np.load(filepath)


# ============================================================================
# CHECKERBOARD CALIBRATOR
# ============================================================================

class CheckerboardCalibrator:
    def __init__(self, board_size=(7, 5), square_size=80):
        """
        board_size: (cols, rows) - number of INNER corners
        square_size: size of each square in pixels
        """
        self.board_size = board_size
        self.square_size = square_size
        self.camera_homography = None
        self.projector_homography = None
        self.projector_window = 'Projector'

    def setup_projector_window(self):
        """Create fullscreen window on projector"""
        cv2.namedWindow(self.projector_window, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.projector_window, cv2.WND_PROP_FULLSCREEN,
                             cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow(self.projector_window, MAIN_MONITOR_WIDTH, 0)

    def generate_checkerboard(self):
        """Generate checkerboard pattern for projection"""
        cols, rows = self.board_size

        # Add 1 to get number of squares
        n_squares_x = cols + 1
        n_squares_y = rows + 1

        total_width = n_squares_x * self.square_size
        total_height = n_squares_y * self.square_size

        # Center on projector
        x_offset = (PROJECTOR_WIDTH - total_width) // 2
        y_offset = (PROJECTOR_HEIGHT - total_height) // 2

        # Create white background
        pattern = np.ones((PROJECTOR_HEIGHT, PROJECTOR_WIDTH, 3), dtype=np.uint8) * 255

        # Draw checkerboard
        for i in range(n_squares_y):
            for j in range(n_squares_x):
                if (i + j) % 2 == 0:
                    x1 = x_offset + j * self.square_size
                    y1 = y_offset + i * self.square_size
                    x2 = x1 + self.square_size
                    y2 = y1 + self.square_size
                    cv2.rectangle(pattern, (x1, y1), (x2, y2), (0, 0, 0), -1)

        # Calculate corner positions in projector coordinates
        projector_corners = []
        for i in range(rows):
            for j in range(cols):
                x = x_offset + (j + 1) * self.square_size
                y = y_offset + (i + 1) * self.square_size
                projector_corners.append((x, y))

        return pattern, projector_corners

    def is_point_in_table(self, point, margin=TABLE_MARGIN_CM):
        """Check if point is in table"""
        x, y = point
        return (margin <= x <= TABLE_WIDTH_CM - margin and
                margin <= y <= TABLE_HEIGHT_CM - margin)

    def calibrate(self, camera_index: int = CAMERA_INDEX) -> Optional[np.ndarray]:
        """Run checkerboard calibration"""

        # Load camera calibration
        if not os.path.exists(CAMERA_CALIB_FILE):
            print("Error: Camera not calibrated! Run camera calibration first.")
            return None

        self.camera_homography = ManualCameraCalibrator.load()
        print("✓ Loaded camera calibration")

        # Setup projector
        self.setup_projector_window()

        # Generate checkerboard
        pattern, projector_corners = self.generate_checkerboard()

        print(f"\n=== Checkerboard Calibration ===")
        print(f"Board size: {self.board_size[0]}x{self.board_size[1]} inner corners")
        print(f"Projecting checkerboard...")

        # Project pattern
        cv2.imshow(self.projector_window, pattern)
        cv2.waitKey(500)

        # Capture frame
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("✗ Cannot open camera")
            return None

        # Wait for camera to adjust
        print("Waiting for camera to adjust...")
        time.sleep(2.0)

        # Discard first frames
        for _ in range(10):
            cap.read()

        # Capture
        print("Capturing...")
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("✗ Cannot capture frame")
            cv2.destroyAllWindows()
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find checkerboard corners
        print(f"Detecting checkerboard corners...")
        ret, camera_corners = cv2.findChessboardCorners(
            gray, self.board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if not ret:
            print("✗ Could not detect checkerboard!")
            print("Tips:")
            print("  - Ensure checkerboard is fully visible")
            print("  - Check focus and lighting")
            print("  - Try adjusting square_size parameter")

            # Show what camera sees
            cv2.imshow('Camera View - No checkerboard detected', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return None

        # Refine corner positions to sub-pixel accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        camera_corners = cv2.cornerSubPix(gray, camera_corners, (11, 11), (-1, -1), criteria)

        # Flatten to list of tuples
        camera_corners = camera_corners.reshape(-1, 2)
        camera_corners_list = [(pt[0], pt[1]) for pt in camera_corners]

        print(f"✓ Detected {len(camera_corners_list)} corners")

        # Transform camera corners to table coordinates
        camera_array = np.array(camera_corners_list, dtype=np.float32).reshape(-1, 1, 2)
        table_corners = cv2.perspectiveTransform(camera_array, self.camera_homography)
        table_corners = table_corners.reshape(-1, 2)

        # Filter by table bounds
        valid_indices = [i for i, pt in enumerate(table_corners)
                        if self.is_point_in_table(pt)]

        print(f"  {len(valid_indices)} corners within table bounds")

        if len(valid_indices) < 12:
            print("✗ Too few corners in table!")
            vis = frame.copy()
            cv2.drawChessboardCorners(vis, self.board_size, camera_corners, True)
            cv2.imshow('Detection - Too few in table', vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return None

        # Keep only valid points
        valid_projector = [projector_corners[i] for i in valid_indices]
        valid_table = [table_corners[i] for i in valid_indices]

        # Visualize
        vis = frame.copy()
        cv2.drawChessboardCorners(vis, self.board_size, camera_corners, True)

        # Mark invalid corners in red
        for i, corner in enumerate(camera_corners_list):
            if i not in valid_indices:
                cv2.circle(vis, (int(corner[0]), int(corner[1])), 10, (0, 0, 255), 2)

        cv2.putText(vis, f"{len(valid_indices)} corners in table", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        detection_window = 'Checkerboard Detection (SPACE=accept, ESC=cancel)'
        cv2.imshow(detection_window, vis)

        print("\n✓ Detection successful!")
        print("Press SPACE to accept, ESC to cancel")

        while True:
            try:
                if cv2.getWindowProperty(detection_window, cv2.WND_PROP_VISIBLE) < 1:
                    cv2.destroyAllWindows()
                    return None
            except:
                pass

            key = cv2.waitKey(50) & 0xFF

            if key == 32:  # SPACE
                # Compute homography
                proj_array = np.array(valid_projector, dtype=np.float32)
                table_array = np.array(valid_table, dtype=np.float32)

                self.projector_homography, _ = cv2.findHomography(proj_array, table_array)

                print("\n✓ Calibration complete!")
                cv2.destroyAllWindows()
                return self.projector_homography

            elif key == 27:  # ESC
                print("Cancelled")
                cv2.destroyAllWindows()
                return None

    def save(self, filepath: str = PROJECTOR_CALIB_FILE):
        """Save homography"""
        if self.projector_homography is None:
            print("Error: No homography to save")
            return
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.save(filepath, self.projector_homography)
        print(f"✓ Saved to {filepath}")


# ============================================================================
# CALIBRATION VERIFIER
# ============================================================================

class CalibrationVerifier:
    def __init__(self):
        self.camera_homography = None
        self.projector_homography = None
        self.projector_window = 'Projector'
        self.click_points = []  # For interactive test

    def load_calibrations(self) -> bool:
        """Load both calibrations"""
        try:
            self.camera_homography = ManualCameraCalibrator.load()
            self.projector_homography = ProjectorCalibrator.load()
            print("✓ Loaded camera and projector calibrations")
            return True
        except FileNotFoundError as e:
            print(f"✗ Calibration file missing: {e}")
            return False
        except Exception as e:
            print(f"✗ Error loading calibrations: {e}")
            return False

    def setup_projector_window(self):
        """Create fullscreen window on projector"""
        cv2.namedWindow(self.projector_window, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.projector_window, cv2.WND_PROP_FULLSCREEN,
                             cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow(self.projector_window, MAIN_MONITOR_WIDTH, 0)

    def table_to_projector(self, table_point):
        """Transform table coordinates to projector pixels"""
        try:
            inv_H = np.linalg.inv(self.projector_homography)
            pt = np.array([[table_point]], dtype=np.float32)
            proj_pt = cv2.perspectiveTransform(pt, inv_H)[0][0]
            return (int(proj_pt[0]), int(proj_pt[1]))
        except:
            return None

    def camera_to_table(self, camera_point):
        """Transform camera coordinates to table coordinates"""
        try:
            pt = np.array([[camera_point]], dtype=np.float32)
            table_pt = cv2.perspectiveTransform(pt, self.camera_homography)[0][0]
            return table_pt
        except:
            return None

    def generate_table_grid(self, grid_spacing_cm=20):
        """
        Generate grid pattern in table coordinates
        Returns projection image with grid
        """
        # Create black background
        projection = np.zeros((PROJECTOR_HEIGHT, PROJECTOR_WIDTH, 3), dtype=np.uint8)

        # Vertical lines
        x = 0
        while x <= TABLE_WIDTH_CM:
            # Top and bottom points of vertical line
            top = self.table_to_projector((x, 0))
            bottom = self.table_to_projector((x, TABLE_HEIGHT_CM))

            if top and bottom:
                color = (0, 255, 0) if x % (grid_spacing_cm * 2) == 0 else (0, 150, 0)
                cv2.line(projection, top, bottom, color, 2 if x % (grid_spacing_cm * 2) == 0 else 1)

                # Label major gridlines
                if x % (grid_spacing_cm * 2) == 0:
                    cv2.putText(projection, f"{int(x)}", (top[0]-15, top[1]-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            x += grid_spacing_cm

        # Horizontal lines
        y = 0
        while y <= TABLE_HEIGHT_CM:
            # Left and right points of horizontal line
            left = self.table_to_projector((0, y))
            right = self.table_to_projector((TABLE_WIDTH_CM, y))

            if left and right:
                color = (0, 255, 0) if y % (grid_spacing_cm * 2) == 0 else (0, 150, 0)
                cv2.line(projection, left, right, color, 2 if y % (grid_spacing_cm * 2) == 0 else 1)

                # Label major gridlines
                if y % (grid_spacing_cm * 2) == 0:
                    cv2.putText(projection, f"{int(y)}", (left[0]+10, left[1]+5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            y += grid_spacing_cm

        return projection

    def generate_corner_markers(self):
        """Project markers at table corners"""
        projection = np.zeros((PROJECTOR_HEIGHT, PROJECTOR_WIDTH, 3), dtype=np.uint8)

        corners = [
            (0, 0, "TL"),
            (TABLE_WIDTH_CM, 0, "TR"),
            (TABLE_WIDTH_CM, TABLE_HEIGHT_CM, "BR"),
            (0, TABLE_HEIGHT_CM, "BL")
        ]

        for x, y, label in corners:
            proj_pt = self.table_to_projector((x, y))
            if proj_pt:
                # Draw crosshair
                cv2.line(projection, (proj_pt[0]-30, proj_pt[1]),
                        (proj_pt[0]+30, proj_pt[1]), (0, 255, 0), 3)
                cv2.line(projection, (proj_pt[0], proj_pt[1]-30),
                        (proj_pt[0], proj_pt[1]+30), (0, 255, 0), 3)
                cv2.circle(projection, proj_pt, 50, (255, 0, 0), 2)

                # Label
                cv2.putText(projection, label, (proj_pt[0]+40, proj_pt[1]+40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

        # Draw table boundary
        boundary_points = [
            self.table_to_projector((0, 0)),
            self.table_to_projector((TABLE_WIDTH_CM, 0)),
            self.table_to_projector((TABLE_WIDTH_CM, TABLE_HEIGHT_CM)),
            self.table_to_projector((0, TABLE_HEIGHT_CM))
        ]

        if all(boundary_points):
            pts = np.array(boundary_points, np.int32)
            cv2.polylines(projection, [pts], True, (0, 0, 255), 3)

        return projection

    def generate_center_target(self):
        """Project target at table center"""
        projection = np.zeros((PROJECTOR_HEIGHT, PROJECTOR_WIDTH, 3), dtype=np.uint8)

        center = (TABLE_WIDTH_CM / 2, TABLE_HEIGHT_CM / 2)
        proj_center = self.table_to_projector(center)

        if proj_center:
            # Concentric circles
            for radius in [80, 60, 40, 20]:
                cv2.circle(projection, proj_center, radius, (0, 255, 0), 2)

            # Crosshair
            cv2.line(projection, (proj_center[0]-100, proj_center[1]),
                    (proj_center[0]+100, proj_center[1]), (255, 0, 0), 3)
            cv2.line(projection, (proj_center[0], proj_center[1]-100),
                    (proj_center[0], proj_center[1]+100), (255, 0, 0), 3)

            # Label
            cv2.putText(projection, f"CENTER ({center[0]:.0f}, {center[1]:.0f})",
                       (proj_center[0]-80, proj_center[1]-120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        return projection

    def mouse_callback(self, event, x, y, flags, param):
        """Click on camera view to project at that location"""
        if event == cv2.EVENT_LBUTTONDOWN:
            table_pt = self.camera_to_table((x, y))
            if table_pt is not None:
                self.click_points.append(table_pt)
                print(f"Clicked table position: ({table_pt[0]:.1f}, {table_pt[1]:.1f}) cm")

    def generate_click_projection(self):
        """Generate projection for clicked points"""
        projection = np.zeros((PROJECTOR_HEIGHT, PROJECTOR_WIDTH, 3), dtype=np.uint8)

        for i, table_pt in enumerate(self.click_points):
            proj_pt = self.table_to_projector(table_pt)
            if proj_pt:
                # Draw marker
                cv2.circle(projection, proj_pt, 30, (0, 255, 255), -1)
                cv2.circle(projection, proj_pt, 35, (255, 255, 255), 3)

                # Label with number
                cv2.putText(projection, str(i+1),
                           (proj_pt[0]-10, proj_pt[1]+10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        return projection

    def run_verification(self, camera_index: int = CAMERA_INDEX):
        """Run interactive verification"""

        if not self.load_calibrations():
            return

        self.setup_projector_window()

        # Open camera
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("✗ Cannot open camera")
            return

        print("\n=== Calibration Verification ===")
        print("\nTests:")
        print("  1 - Grid pattern (aligned to table coordinates)")
        print("  2 - Corner markers (verify table boundaries)")
        print("  3 - Center target (verify center alignment)")
        print("  4 - Interactive test (click to project)")
        print("  5 - Clear projection")
        print("  Q - Quit\n")

        camera_window = 'Camera View - Press keys 1-5 for tests'
        cv2.namedWindow(camera_window)
        cv2.setMouseCallback(camera_window, self.mouse_callback)

        current_projection = np.zeros((PROJECTOR_HEIGHT, PROJECTOR_WIDTH, 3), dtype=np.uint8)
        current_test = "None"

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Display camera view
            display = frame.copy()

            # Add status overlay
            cv2.putText(display, f"Current test: {current_test}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display, "Press 1-5 for tests | Q to quit", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if current_test == "Interactive":
                cv2.putText(display, f"Click count: {len(self.click_points)} | C to clear",
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Draw clicked points on camera view
                for i, table_pt in enumerate(self.click_points):
                    # Transform back to camera for visualization
                    inv_cam_H = np.linalg.inv(self.camera_homography)
                    pt = np.array([[table_pt]], dtype=np.float32)
                    cam_pt = cv2.perspectiveTransform(pt, inv_cam_H)[0][0]

                    cv2.circle(display, (int(cam_pt[0]), int(cam_pt[1])), 10, (0, 255, 255), 2)
                    cv2.putText(display, str(i+1),
                               (int(cam_pt[0])+15, int(cam_pt[1])),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow(camera_window, display)
            cv2.imshow(self.projector_window, current_projection)

            # Handle keys
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q') or key == ord('Q') or key == 27:
                break

            elif key == ord('1'):
                print("\nTest 1: Grid Pattern")
                print("Green grid should align with table coordinate system")
                current_projection = self.generate_table_grid(grid_spacing_cm=20)
                current_test = "Grid (20cm spacing)"

            elif key == ord('2'):
                print("\nTest 2: Corner Markers")
                print("Crosshairs should appear at exact table corners")
                current_projection = self.generate_corner_markers()
                current_test = "Corner Markers"

            elif key == ord('3'):
                print("\nTest 3: Center Target")
                print("Target should appear at exact table center")
                current_projection = self.generate_center_target()
                current_test = "Center Target"

            elif key == ord('4'):
                print("\nTest 4: Interactive")
                print("Click anywhere on table in camera view")
                print("Yellow circles should appear where you click")
                current_test = "Interactive"

            elif key == ord('5'):
                print("\nClearing projection")
                current_projection = np.zeros((PROJECTOR_HEIGHT, PROJECTOR_WIDTH, 3), dtype=np.uint8)
                current_test = "None"

            elif key == ord('c') or key == ord('C'):
                if current_test == "Interactive":
                    self.click_points = []
                    print("Cleared click points")

            # Update interactive projection
            if current_test == "Interactive":
                current_projection = self.generate_click_projection()

        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Verification complete")


# ============================================================================
# COMPONENT TESTS
# ============================================================================

def test_camera():
    """Test if camera works"""
    print("\n=== Testing Camera ===")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"✗ Cannot open camera {CAMERA_INDEX}")
        return False

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("✗ Cannot read frame")
        return False

    print(f"✓ Camera works! Resolution: {frame.shape[1]}x{frame.shape[0]}")

    # Show frame
    cv2.imshow('Camera Test - Press any key', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return True


def test_mouse_callback():
    """Test if mouse callback works"""
    print("\n=== Testing Mouse Callback ===")

    points = []

    def callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f"Clicked at ({x}, {y})")

    # Create test image
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    window_name = 'Mouse Test - Click 3 times'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, callback)

    print("Click 3 times on the window...")

    while len(points) < 3:
        display = img.copy()

        for i, pt in enumerate(points):
            cv2.circle(display, pt, 10, (0, 255, 0), -1)
            cv2.putText(display, str(i+1), (pt[0]+15, pt[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(display, f"Clicks: {len(points)}/3", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow(window_name, display)

        if cv2.waitKey(30) & 0xFF == 27:  # ESC
            break

    cv2.destroyAllWindows()

    if len(points) == 3:
        print("✓ Mouse callback works!")
        return True
    else:
        print("✗ Mouse callback failed")
        return False


def test_circle_detection():
    """Test if circle detection works"""
    print("\n=== Testing Circle Detection ===")

    detector = CircleDetector()

    # Create test image with circles
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255  # White background

    # Draw some test circles
    test_circles = [(100, 100), (300, 200), (500, 300)]
    for pt in test_circles:
        cv2.circle(img, pt, 30, (255, 0, 0), -1)  # Blue circles

    detected = detector.detect_colored_circles(img, invert=True)

    print(f"Drew {len(test_circles)} circles")
    print(f"Detected {len(detected)} circles")

    if len(detected) > 0:
        print("✓ Circle detection works!")

        # Show result
        vis = detector.visualize_colored_detection(img, detected)
        cv2.imshow('Detection Test - Press any key', vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return True
    else:
        print("✗ Circle detection failed")
        cv2.imshow('Failed Detection - Press any key', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return False


def run_component_tests():
    """Run all component tests"""
    print("="*50)
    print("COMPONENT TESTING")
    print("="*50)

    results = {}

    results['camera'] = test_camera()
    results['mouse'] = test_mouse_callback()
    results['detection'] = test_circle_detection()

    print("\n" + "="*50)
    print("RESULTS:")
    print("="*50)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name.capitalize()}: {status}")


# ============================================================================
# MAIN CALIBRATION WORKFLOW FUNCTIONS
# ============================================================================

def print_menu():
    print("\n" + "="*60)
    print("BILLIARDS PROJECTION SYSTEM - CALIBRATION")
    print("="*60)
    print("1. Camera Calibration (4-point, drag & drop)")
    print("2. Test Detection (live preview with projection)")
    print("3. Projector Calibration - Colored Circles (automatic)")
    print("4. Projector Calibration - Checkerboard (more robust)")
    print("5. Verify Calibration (test accuracy)")
    print("6. Run Full Calibration (camera + projector)")
    print("7. Check Calibration Status")
    print("8. Run Component Tests")
    print("9. Exit")
    print("="*60)


def check_calibration_status():
    """Check which calibrations exist"""
    camera_exists = os.path.exists(CAMERA_CALIB_FILE)
    projector_exists = os.path.exists(PROJECTOR_CALIB_FILE)

    print("\n--- Calibration Status ---")
    print(f"Camera:    {'✓ Calibrated' if camera_exists else '✗ Not calibrated'}")
    print(f"Projector: {'✓ Calibrated' if projector_exists else '✗ Not calibrated'}")

    if camera_exists:
        print(f"  Camera file: {CAMERA_CALIB_FILE}")
    if projector_exists:
        print(f"  Projector file: {PROJECTOR_CALIB_FILE}")

    return camera_exists, projector_exists


def run_camera_calibration():
    """Run camera calibration"""
    print("\n--- Starting Camera Calibration ---")
    calibrator = ManualCameraCalibrator()
    H = calibrator.calibrate()

    if H is not None:
        calibrator.save()
        print("✓ Camera calibration completed!")
        return True
    else:
        print("✗ Camera calibration cancelled")
        return False


def run_detection_test():
    """Run live detection test"""
    if not os.path.exists(CAMERA_CALIB_FILE):
        print("\n✗ Error: Camera must be calibrated first!")
        print("Please run camera calibration (option 1)")
        return

    print("\n--- Starting Detection Test ---")
    tester = DetectionTester()
    tester.run_live_test()


def run_projector_calibration_circles():
    """Run projector calibration with colored circles"""
    if not os.path.exists(CAMERA_CALIB_FILE):
        print("\n✗ Error: Camera must be calibrated first!")
        print("Please run camera calibration (option 1)")
        return False

    print("\n--- Starting Projector Calibration (Circles) ---")
    calibrator = ProjectorCalibrator()
    H = calibrator.calibrate()

    if H is not None:
        calibrator.save()
        print("✓ Projector calibration completed!")
        return True
    else:
        print("✗ Projector calibration cancelled")
        return False


def run_projector_calibration_checkerboard():
    """Run projector calibration with checkerboard"""
    if not os.path.exists(CAMERA_CALIB_FILE):
        print("\n✗ Error: Camera must be calibrated first!")
        print("Please run camera calibration (option 1)")
        return False

    print("\n--- Starting Projector Calibration (Checkerboard) ---")
    print("Default settings: 7x5 inner corners, 80px squares")

    # Allow user to customize if desired
    customize = input("Use default settings? (y/n): ").strip().lower()

    if customize == 'n':
        try:
            cols = int(input("Enter number of inner corner columns (default 7): ") or "7")
            rows = int(input("Enter number of inner corner rows (default 5): ") or "5")
            square_size = int(input("Enter square size in pixels (default 80): ") or "80")
            calibrator = CheckerboardCalibrator(board_size=(cols, rows), square_size=square_size)
        except ValueError:
            print("Invalid input, using defaults")
            calibrator = CheckerboardCalibrator(board_size=(7, 5), square_size=80)
    else:
        calibrator = CheckerboardCalibrator(board_size=(7, 5), square_size=80)

    H = calibrator.calibrate()

    if H is not None:
        calibrator.save()
        print("✓ Projector calibration completed!")
        return True
    else:
        print("✗ Projector calibration cancelled")
        return False


def run_verification():
    """Run calibration verification"""
    camera_exists = os.path.exists(CAMERA_CALIB_FILE)
    projector_exists = os.path.exists(PROJECTOR_CALIB_FILE)

    if not camera_exists or not projector_exists:
        print("\n✗ Error: Both camera and projector must be calibrated!")
        if not camera_exists:
            print("  Missing: Camera calibration (run option 1)")
        if not projector_exists:
            print("  Missing: Projector calibration (run option 3 or 4)")
        return

    print("\n--- Starting Calibration Verification ---")
    verifier = CalibrationVerifier()
    verifier.run_verification()


def run_full_calibration():
    """Run complete calibration workflow"""
    print("\n" + "="*60)
    print("FULL CALIBRATION WORKFLOW")
    print("="*60)

    # Step 1: Camera calibration
    print("\n>>> STEP 1: Camera Calibration <<<")
    if os.path.exists(CAMERA_CALIB_FILE):
        use_existing = input("Camera already calibrated. Use existing? (y/n): ").strip().lower()
        if use_existing != 'y':
            if not run_camera_calibration():
                print("\nFull calibration aborted")
                return
    else:
        if not run_camera_calibration():
            print("\nFull calibration aborted")
            return

    # Optional: Test detection
    print("\n>>> Optional: Test Detection <<<")
    test = input("Run detection test to preview? (y/n): ").strip().lower()
    if test == 'y':
        run_detection_test()

    # Step 2: Projector calibration
    print("\n>>> STEP 2: Projector Calibration <<<")
    if os.path.exists(PROJECTOR_CALIB_FILE):
        use_existing = input("Projector already calibrated. Use existing? (y/n): ").strip().lower()
        if use_existing == 'y':
            print("Using existing projector calibration")
        else:
            print("\nChoose projector calibration method:")
            print("1 - Colored circles (automatic retry)")
            print("2 - Checkerboard (more robust, recommended)")
            method = input("Choice (1 or 2): ").strip()

            if method == '1':
                if not run_projector_calibration_circles():
                    print("\nProjector calibration failed")
                    return
            elif method == '2':
                if not run_projector_calibration_checkerboard():
                    print("\nProjector calibration failed")
                    return
            else:
                print("Invalid choice, aborting")
                return
    else:
        print("\nChoose projector calibration method:")
        print("1 - Colored circles (automatic retry)")
        print("2 - Checkerboard (more robust, recommended)")
        method = input("Choice (1 or 2): ").strip()

        if method == '1':
            if not run_projector_calibration_circles():
                print("\nProjector calibration failed")
                return
        elif method == '2':
            if not run_projector_calibration_checkerboard():
                print("\nProjector calibration failed")
                return
        else:
            print("Invalid choice, aborting")
            return

    # Step 3: Verification
    print("\n>>> STEP 3: Verification <<<")
    verify = input("Run verification tests? (y/n): ").strip().lower()
    if verify == 'y':
        run_verification()

    print("\n" + "="*60)
    print("FULL CALIBRATION COMPLETE!")
    print("="*60)
    print("Your system is ready to use.")
    print("You can always run verification (option 5) to test accuracy.")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*60)
    print("WELCOME TO BILLIARDS PROJECTION CALIBRATION")
    print("="*60)
    print("\nRecommended workflow:")
    print("  • First time: Choose option 6 (Full Calibration)")
    print("  • Quick start:")
    print("    1. Calibrate camera (option 1)")
    print("    2. Calibrate projector with checkerboard (option 4)")
    print("    3. Verify accuracy (option 5)")

    while True:
        print_menu()
        choice = input("\nEnter choice (1-9): ").strip()

        if choice == '1':
            run_camera_calibration()

        elif choice == '2':
            run_detection_test()

        elif choice == '3':
            run_projector_calibration_circles()

        elif choice == '4':
            run_projector_calibration_checkerboard()

        elif choice == '5':
            run_verification()

        elif choice == '6':
            run_full_calibration()

        elif choice == '7':
            check_calibration_status()

        elif choice == '8':
            run_component_tests()

        elif choice == '9':
            print("\nExiting...")
            print("Calibration files saved in data/calibration/")
            break

        else:
            print("Invalid choice! Please enter 1-9")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()
