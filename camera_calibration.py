"""Manual camera calibration using 4 corner points"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import os

from config import TABLE_WIDTH_CM, TABLE_HEIGHT_CM, CAMERA_CALIB_FILE


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


if __name__ == '__main__':
    calibrator = ManualCameraCalibrator()
    H = calibrator.calibrate()
    if H is not None:
        calibrator.save()