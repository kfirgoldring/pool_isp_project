"""Projector calibration using checkerboard pattern - more robust!"""

import cv2
import numpy as np
from typing import Optional, Tuple
import os
import time

from config import (CAMERA_INDEX, PROJECTOR_WIDTH, PROJECTOR_HEIGHT,
                   MAIN_MONITOR_WIDTH, PROJECTOR_CALIB_FILE, CAMERA_CALIB_FILE,
                   TABLE_WIDTH_CM, TABLE_HEIGHT_CM, TABLE_MARGIN_CM)
from camera_calibration import ManualCameraCalibrator


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


if __name__ == '__main__':
    calibrator = CheckerboardCalibrator()
    H = calibrator.calibrate()
    if H is not None:
        calibrator.save()