"""Detect projected calibration patterns in camera image"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
import colorsys

from config import MIN_CIRCLE_RADIUS, MAX_CIRCLE_RADIUS, INVERT_PATTERN


class CircleDetector:
    def __init__(self, min_radius: int = MIN_CIRCLE_RADIUS, 
                 max_radius: int = MAX_CIRCLE_RADIUS):
        self.min_radius = min_radius
        self.max_radius = max_radius
    
    def detect_colored_circles(self, image: np.ndarray,
                               invert: bool = INVERT_PATTERN) -> List[Tuple[Tuple[float, float], Tuple[int, int, int]]]:
        """
        Detect circles and extract their colors
        
        Returns:
            List of ((x, y), (B, G, R)) tuples
        """
        # Convert to grayscale for circle detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Adaptive threshold
        if invert:
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 51, 10)
        else:
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 51, -10)
        
        # Morphological cleanup
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        circles = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            min_area = np.pi * self.min_radius ** 2
            max_area = np.pi * self.max_radius ** 2
            
            if min_area < area < max_area:
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    
                    if circularity > 0.5:  # Reasonably circular
                        # Calculate centroid
                        M = cv2.moments(contour)
                        if M['m00'] > 0:
                            cx = M['m10'] / M['m00']
                            cy = M['m01'] / M['m00']
                            
                            # Extract color at center
                            color = self.extract_circle_color(image, (cx, cy), contour)
                            
                            if color is not None:
                                circles.append(((cx, cy), color))
        
        return circles
    
    def extract_circle_color(self, image: np.ndarray, 
                            center: Tuple[float, float],
                            contour: np.ndarray) -> Tuple[int, int, int]:
        """
        Extract the dominant color of a circle
        """
        # Create mask for this circle
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Get mean color within circle
        mean_color = cv2.mean(image, mask=mask)[:3]
        
        return (int(mean_color[0]), int(mean_color[1]), int(mean_color[2]))
    
    def match_circles_by_color(self, 
                               detected_circles: List[Tuple[Tuple[float, float], Tuple[int, int, int]]],
                               projected_map: Dict[Tuple[int, int], Tuple[int, int, int]],
                               color_tolerance: int = 50) -> List[Tuple[Tuple[int, int], Tuple[float, float]]]:
        """
        Match detected circles to projected circles by color
        
        Returns:
            List of (projector_center, camera_center) pairs
        """
        matches = []
        
        for camera_center, camera_color in detected_circles:
            best_match = None
            best_distance = float('inf')
            
            for proj_center, proj_color in projected_map.items():
                # Calculate color distance (Euclidean in BGR space)
                color_dist = np.sqrt(
                    (camera_color[0] - proj_color[0])**2 +
                    (camera_color[1] - proj_color[1])**2 +
                    (camera_color[2] - proj_color[2])**2
                )
                
                if color_dist < best_distance and color_dist < color_tolerance:
                    best_distance = color_dist
                    best_match = proj_center
            
            if best_match is not None:
                matches.append((best_match, camera_center))
        
        return matches
    
    def visualize_colored_detection(self, image: np.ndarray, 
                                   circles: List[Tuple[Tuple[float, float], Tuple[int, int, int]]]) -> np.ndarray:
        """Draw detected colored circles on image"""
        vis = image.copy()
        
        for i, ((x, y), color) in enumerate(circles):
            # Draw circle with detected color
            cv2.circle(vis, (int(x), int(y)), 8, color, 2)
            # Draw index
            cv2.putText(vis, str(i), (int(x)+15, int(y)+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return vis