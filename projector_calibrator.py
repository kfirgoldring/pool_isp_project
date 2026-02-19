"""Automatic projector calibration using camera feedback"""

import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict
import os
import time

from config import (CAMERA_INDEX, PATTERN_CONFIGS, PROJECTOR_WIDTH, PROJECTOR_HEIGHT,
                   MAIN_MONITOR_WIDTH, MIN_REQUIRED_MATCHES,
                   PROJECTOR_CALIB_FILE, CAMERA_CALIB_FILE,
                   TABLE_WIDTH_CM, TABLE_HEIGHT_CM, TABLE_MARGIN_CM)
from pattern_generator import PatternGenerator
from detector import CircleDetector
from camera_calibration import ManualCameraCalibrator


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