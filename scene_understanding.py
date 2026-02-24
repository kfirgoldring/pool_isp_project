import cv2
import numpy as np
import matplotlib.pyplot as plt
import config


def masking(img):

    # step 1: mask green thresholding
    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower_green = np.array([60, 0, 0])
    upper_green = np.array([73, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    if mask is None or mask.size == 0:
        print("Error: Could not create mask.")
        return None
    
    #step 2: morphology open to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    final_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return final_mask


def get_table_corners_v1(image):
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

    # Morphology: open 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find the largest contour (the table surface)
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    table_contour = max(contours, key=cv2.contourArea)

    # Geometric Correction: Convex Hull
    hull = cv2.convexHull(table_contour)

    # show before and after
    vis_1 = cv2.resize(mask, (mask.shape[1] // 2, mask.shape[0] // 2))
    vis_2 = cv2.resize(closed_mask, (closed_mask.shape[1] // 2, closed_mask.shape[0] // 2))
    combined_vis = cv2.hconcat([vis_1, vis_2])
    cv2.imshow("Mask Before and After Morphology", combined_vis)
    cv2.waitKey(0)
    # show contours for debugging
    vis_3 = img.copy()
    cv2.drawContours(vis_3, [table_contour], -1, (0, 255, 0), 3) # Green contour
    cv2.imshow("Table Contour", vis_3)
    cv2.waitKey(0)
    # show hull for debugging
    vis_4 = img.copy()
    cv2.drawContours(vis_4, [hull], -1, (255, 0, 0), 3) # Blue hull
    cv2.imshow("Table Hull", vis_4)
    cv2.waitKey(0)

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
def show_hue_histogram(image_path: str):
    """
    Loads an image, converts it to HSV, and displays the histogram of the Hue channel.
    Helps to determine the range of green values for thresholding.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    plt.figure(figsize=(8, 4))
    plt.hist(hue.ravel(), bins=180, range=[0, 180], color='g', alpha=0.7)
    plt.title('Hue Channel Histogram')
    plt.xlabel('Hue Value (0-179)')
    plt.ylabel('Pixel Count')
    plt.grid(True)
    plt.show()

### DIFFERENT VERSION WITH CANNY + HOUGH ### 

def compute_intersection(line1, line2):
    """
    Finds the intersection of two lines given in Hesse normal form (rho, theta).
    Returns [x, y] or None if lines are parallel.
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    
    # Set up the linear system:
    # x * cos(theta) + y * sin(theta) = rho
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    
    # If determinant is 0, lines are parallel
    if np.abs(np.linalg.det(A)) < 1e-10:
        return None
        
    # Solve for [x, y]
    x0, y0 = np.linalg.solve(A, b)
    return [x0[0], y0[0]]

def order_points(pts):
    """
    Orders points in [top-left, top-right, bottom-right, bottom-left] order.
    """
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")

    # Sum of (x, y) - Top-Left has min sum, Bottom-Right has max sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Diff of (x, y) - Top-Right has min diff (x-y), Bottom-Left has max diff
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def get_table_corners(image):
    """Detect the 4 table corners from an image using Hull -> Canny -> Hough."""
    
    # --- Standard loading checks ---
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None: return None
    elif isinstance(image, np.ndarray):
        img = image
    else:
        return None

    # Get the Mask & Hull
    final_mask = masking(img)
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    table_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(table_contour)

    # --- Gradient Analysis ---

    # 2. Draw Hull on a blank canvas (1px wide)
    hull_canvas = np.zeros_like(final_mask)
    cv2.drawContours(hull_canvas, [hull], -1, 255, 1)

    # Dilate hull to create a "search zone" for gradients
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gradient_mask = cv2.dilate(hull_canvas, kernel_dilate, iterations=1)

    # DEBUG : show mask
    # cv2.imshow("Gradient Search Zone", gradient_mask)
    # cv2.waitKey(0)  

    # --- Targeted Hough Transform ---

    # for horizontal lines seperate bottom and top search
    x, y, w, h = cv2.boundingRect(hull)
    mid_y = y + h // 2
    top_mask = gradient_mask.copy()
    top_mask[mid_y:, :] = 0
    bottom_mask = gradient_mask.copy()
    bottom_mask[:mid_y, :] = 0
    
    ranges_deg = [(0,20) , (160,180) , (85,95) , (85,95)] # right, left, top, bottom
    edges = [gradient_mask,gradient_mask,top_mask,bottom_mask]
    threshold = 100

    lines = []
    for i, (min_deg, max_deg) in enumerate(ranges_deg):
        
        # Convert to radians 
        min_theta = np.deg2rad(min_deg)
        max_theta = np.deg2rad(max_deg)
        
        # Run Hough Transform constrained to the range
        Hough_lines = cv2.HoughLines(
            edges[i], 
            rho=1, 
            theta=np.pi/180, 
            threshold=threshold, 
            min_theta=min_theta, 
            max_theta=max_theta
        )
        
        # Find the "Best" line in this batch
        if Hough_lines is not None and len(Hough_lines) > 0:
            best_line = Hough_lines[0] # Take the strongest one
            lines.append(best_line)
            
            rho, theta = best_line[0]
            #print(f"Range {min_deg}-{max_deg}°: Found line at {np.rad2deg(theta):.2f}°")
        else:
            print(f"Range {min_deg}-{max_deg}°: No line found.")
    
    
    # show lines on otiginal image for debugging
    '''vis_img = img.copy()
    for line in lines:
        if line is not None:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            length = 5000
            pt1 = (int(x0 + length * (-b)), int(y0 + length * (a)))
            pt2 = (int(x0 - length * (-b)), int(y0 - length * (a)))
            cv2.line(vis_img, pt1, pt2, (255, 0, 255), 2) # Magenta lines

    # Visualization of detected lines
    cv2.imshow("Detected Lines", vis_img)
    cv2.waitKey(0)'''

    # find corners by intersecting the lines
    right_v = lines[0]
    left_v = lines[1]
    top_h = lines[2]
    bottom_h = lines[3]

    corners = []
    corners.append(compute_intersection(left_v, top_h))      # Corner 1
    corners.append(compute_intersection(right_v, top_h))     # Corner 2
    corners.append(compute_intersection(right_v, bottom_h))  # Corner 3
    corners.append(compute_intersection(left_v, bottom_h))   # Corner 4
    corners = [c for c in corners if c is not None]

    if len(corners) < 4:
        print("Error: Could not compute all 4 intersections.")
        return None

    # order corners
    final_corners = order_points(corners) 
    corner_img = img.copy()
    for i, pt in enumerate(final_corners):
        cv2.circle(corner_img, (int(pt[0]), int(pt[1])), 10, (0, 255, 0), -1)
        cv2.putText(corner_img, f"C{i}", (int(pt[0]), int(pt[1])-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # show: DEBUG
    # cv2.imshow("Final Corners", corner_img)
    # cv2.waitKey(0)

    return final_corners

def detect_disturbance(ref, image):
    """
    Detects if a disturbance (e.g., hand or cue) is present inside the table borders
    by comparing the current image to a reference image of the empty table.

    Parameters
    ----------
    ref : np.ndarray
        Reference image (empty table).
    image : np.ndarray
        Current frame to check for disturbance.
    diff_pixel_thresh : int
        Number of differing pixels required to trigger disturbance (tunable).
    dilation_size : int
        Size of dilation kernel for mask (tunable).

    Returns
    -------
    bool
        True if disturbance detected, False otherwise.
    """
    # Step 1: Get corners from reference image
    corners = get_table_corners(ref)
    if corners is None or len(corners) != 4:
        print("Could not find table corners in reference image.")
        return False

    '''# Step 2: Create border mask
    mask = np.zeros(ref.shape[:2], dtype=np.uint8)
    pts = np.array(corners, dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(mask, [pts], isClosed=True, color=255, thickness=1)
    dilation_size = 15
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
    mask = cv2.dilate(mask, kernel, iterations=1)'''

    # Step 1: Create filled table mask
    mask_full = np.zeros(ref.shape[:2], dtype=np.uint8)
    pts = np.array(corners, dtype=np.int32)
    cv2.fillPoly(mask_full, [pts], 255)

    # Step 2: Erode the mask to shrink it inward
    dilation_size = 30
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
    mask_eroded = cv2.erode(mask_full, kernel, iterations=1)

    # Step 3: Subtract eroded mask from full mask to get only the inner border
    mask = cv2.subtract(mask_full, mask_eroded)

    # Step 3: Mask both images
    ref_masked = cv2.bitwise_and(ref, ref, mask=mask)
    img_masked = cv2.bitwise_and(image, image, mask=mask)

    # Step 4: Compute absolute difference in masked region
    diff = cv2.absdiff(ref_masked, img_masked)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, diff_thresh = cv2.threshold(diff_gray, 100, 255, cv2.THRESH_BINARY)

    # Step 5: Count differing pixels
    num_diff_pixels = cv2.countNonZero(diff_thresh)

    # Debug visualization
    #print(f"Number of differing pixels: {num_diff_pixels}")
    #ref_masked_resize = cv2.resize(ref_masked, (ref_masked.shape[1] // 2, ref_masked.shape[0] // 2))
    #img_masked_resize = cv2.resize(img_masked, (img_masked.shape[1] // 2, img_masked.shape[0] // 2))
    #cv2.imshow("Reference Masked", ref_masked_resize)
    #cv2.imshow("Current Masked", img_masked_resize)
    #mask_resize = cv2.resize(mask, (mask.shape[1] // 2, mask.shape[0] // 2))
    #cv2.imshow("mask", mask_resize)
    #diff_resize = cv2.resize(diff, (diff.shape[1] // 2, diff.shape[0] // 2))
    #cv2.imshow("Difference", diff_resize)

    # Step 6: Decide disturbance
    diff_pixel_thresh = 100
    if num_diff_pixels > diff_pixel_thresh:
        return True
    else:
        return False

def main():
    ref = cv2.imread('runs\\2026-02-23_12-07-24\\ref.jpeg')
    image = cv2.imread('runs\\2026-02-23_12-07-24\\capture_20260223_121004.jpeg')
    print("Detecting disturbance in image_false:", detect_disturbance(ref,image))

if __name__ == "__main__":
    main()