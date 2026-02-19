"""Test detection in real-time with projection and perspective correction"""

import cv2
import numpy as np
import os

from config import (CAMERA_INDEX, TABLE_WIDTH_CM, TABLE_HEIGHT_CM, CAMERA_CALIB_FILE,
                   PROJECTOR_WIDTH, PROJECTOR_HEIGHT, MAIN_MONITOR_WIDTH,
                   PATTERN_CONFIGS, INVERT_PATTERN)
from detector import CircleDetector
from pattern_generator import PatternGenerator
from camera_calibration import ManualCameraCalibrator


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


def main():
    """Standalone test"""
    tester = DetectionTester()
    tester.run_live_test()


if __name__ == '__main__':
    main()