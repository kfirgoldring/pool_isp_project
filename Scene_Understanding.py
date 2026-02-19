import matplotlib.pyplot as plt
import cv2
import numpy as np

def detect_table_green(img):
    # Pre-processing: Blur to reduce noise
    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    # Convert to HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Define the Green Threshold
    # Hue: 35-85 covers most green shades
    # Saturation: 40-255 filters out gray/white noise
    # Value: 40-255 filters out very dark shadows
    lower_green = np.array([60, 0, 0])
    upper_green = np.array([85, 255, 255])

    # Create the binary mask
    mask = cv2.inRange(hsv, lower_green, upper_green)

    return mask

# Usage:
# green_mask = detect_table_green('image_2ba55e.jpg')
# cv2.imshow('Table Mask', green_mask)
# cv2.waitKey(0)

def get_table_corners(image_path):

    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Load and mask the image
    mask = detect_table_green(img)

    if mask is None or mask.size == 0:
        print("Error: Could not create mask.")
        return None

    # Morphology: Close the ball holes and bridge the pocket gaps
    # We use a large rectangular kernel to "straighten" the edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 60))
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    ### DEBUG: show mask, image, and closed mask side by side ###
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    img_resized = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.title('Green Mask')
    mask_resized = cv2.resize(mask, (mask.shape[1] // 2, mask.shape[0] // 2))
    plt.imshow(mask_resized, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.title('Mask After Closing')
    closed_mask_resized = cv2.resize(closed_mask, (closed_mask.shape[1] // 2, closed_mask.shape[0] // 2))
    plt.imshow(closed_mask_resized, cmap='gray')
    plt.axis('off')
    plt.show()


    # Find the largest contour (the table surface)
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    table_contour = max(contours, key=cv2.contourArea)

    # Geometric Correction: Convex Hull
    hull = cv2.convexHull(table_contour)

    ### DEBUG: Show hull on the same image
    vis_img_1 = img.copy()
    cv2.drawContours(vis_img_1, [hull], -1, (0, 255, 0), 3)   
    vis_img_resized = cv2.resize(vis_img_1, (vis_img_1.shape[1] // 2, vis_img_1.shape[0] // 2))
    cv2.imshow("Hull", vis_img_resized)
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
    
    vis_img_2_resized = cv2.resize(vis_img_2, (vis_img_2.shape[1] // 2, vis_img_2.shape[0] // 2))
    cv2.imshow("Hull with Corners", vis_img_2_resized)
    cv2.waitKey(0)

    return rect

# Utility function to show the histogram of the Hue channel
def show_hue_histogram():
    """
    Loads an image, converts it to HSV, and displays the histogram of the Hue channel.
    Helps to determine the range of green values for thresholding.
    """
    image_path = 'table_pics\WIN_20260219_10_29_03_Pro.jpg'
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
    image_path = 'table_pics\WIN_20260219_10_30_02_Pro.jpg'
    get_table_corners(image_path)

if __name__ == "__main__":
    main()