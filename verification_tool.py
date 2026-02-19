"""Verify projector calibration accuracy"""

import cv2
import numpy as np
import os

from config import (CAMERA_INDEX, PROJECTOR_WIDTH, PROJECTOR_HEIGHT,
                   MAIN_MONITOR_WIDTH, TABLE_WIDTH_CM, TABLE_HEIGHT_CM,
                   CAMERA_CALIB_FILE, PROJECTOR_CALIB_FILE)
from camera_calibration import ManualCameraCalibrator
from projector_calibrator import ProjectorCalibrator


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


def main():
    """Standalone verification"""
    verifier = CalibrationVerifier()
    verifier.run_verification()


if __name__ == '__main__':
    main()