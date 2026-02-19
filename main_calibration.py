"""Main script to run full calibration process with menu"""

import cv2
import os
from camera_calibration import ManualCameraCalibrator
from projector_calibrator import ProjectorCalibrator
from checkerboard_calibrator import CheckerboardCalibrator
from detection_tester import DetectionTester
from verification_tool import CalibrationVerifier
from config import CAMERA_CALIB_FILE, PROJECTOR_CALIB_FILE


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
    print("8. Exit")
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
        choice = input("\nEnter choice (1-8): ").strip()
        
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
            print("\nExiting...")
            print("Calibration files saved in data/calibration/")
            break
        
        else:
            print("Invalid choice! Please enter 1-8")
    
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