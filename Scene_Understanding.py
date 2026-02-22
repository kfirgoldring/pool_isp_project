import cv2
import numpy as np

import config


def detect_table_green(img):
    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    lower_green = np.array([60, 0, 0])
    upper_green = np.array([73, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    return mask


def get_table_corners(image):
    """Detect the 4 table corners from an image.

    Parameters
    ----------
    image : str or numpy.ndarray
        Either a file path (str) or a BGR numpy array.

    Returns
    -------
    numpy.ndarray of shape (4, 2) with corner order
    [top-left, top-right, bottom-right, bottom-left], or None on failure.
    """
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            return None
    elif isinstance(image, np.ndarray):
        img = image
    else:
        return None

    mask = detect_table_green(img)

    if mask is None or mask.size == 0:
        print("Error: Could not create mask.")
        return None

    # Morphology: Close the ball holes and bridge the pocket gaps
    # We use a large rectangular kernel to "straighten" the edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find the largest contour (the table surface)
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    table_contour = max(contours, key=cv2.contourArea)

    # Geometric Correction: Convex Hull
    hull = cv2.convexHull(table_contour)

    # find corners with extreme points

    # Reshape to a simple list of (x, y) coordinates
    pts = hull.reshape(-1, 2)

    # Initialize our 4 corner array
    rect = np.zeros((4, 2), dtype="float32")

    # Find Top-Left and Bottom-Right
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Top-Left
    rect[2] = pts[np.argmax(s)] # Bottom-Right

    # Find Top-Right and Bottom-Left
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Top-Right
    rect[3] = pts[np.argmax(diff)] # Bottom-Left

    # Visualization for debugging
    vis_img_2 = img.copy()
    for (x, y) in rect:
        cv2.circle(vis_img_2, (int(x), int(y)), 10, (0, 0, 255), -1) # Red dots
    
    return rect


def compute_homography_from_corners(corners: np.ndarray) -> np.ndarray:
    """Compute camera-pixel -> table-cm homography from 4 corner points.

    Parameters
    ----------
    corners : (4, 2) float32 array of camera-pixel corner coordinates,
              ordered [top-left, top-right, bottom-right, bottom-left].

    Returns
    -------
    (3, 3) float32 homography matrix mapping camera pixels to table cm.
    """
    src = np.array(corners, dtype=np.float32).reshape(4, 2)
    dst = np.array([
        [0.0,                  0.0],
        [config.TABLE_WIDTH_CM,  0.0],
        [config.TABLE_WIDTH_CM,  config.TABLE_HEIGHT_CM],
        [0.0,                  config.TABLE_HEIGHT_CM],
    ], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H.astype(np.float32)


# Utility function to show the histogram of the Hue channel
def show_hue_histogram():
    """
    Loads an image, converts it to HSV, and displays the histogram of the Hue channel.
    Helps to determine the range of green values for thresholding.
    """
    image_path = r'table_pics\\table\\new_cam_in_lib_ref.jpeg'
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    plt.figure(figsize=(8, 4))
    plt.hist(hue.ravel(), bins=180, range=[0, 180], color='g', alpha=0.7)
    plt.title('Hue Channel Histogram')
    plt.xlabel('Hue Value (0-179)')
    plt.ylabel('Pixel Count')
    plt.grid(True)
    plt.show()

def main():
    image_path = r'table_pics\\table\\new_cam_in_lib_ref.jpeg'
    get_table_corners(image_path)

if __name__ == "__main__":
    main()