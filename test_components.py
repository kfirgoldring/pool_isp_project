"""Simple tests to verify each component works"""

import cv2
import numpy as np
from config import CAMERA_INDEX

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
    
    from detector import CircleDetector
    
    # Create test image with circles
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255  # White background
    
    # Draw some test circles
    test_circles = [(100, 100), (300, 200), (500, 300)]
    for pt in test_circles:
        cv2.circle(img, pt, 30, (255, 0, 0), -1)  # Blue circles
    
    detector = CircleDetector()
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


def main():
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


if __name__ == '__main__':
    main()