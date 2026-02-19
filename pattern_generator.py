"""Generate calibration patterns for projector calibration"""

import numpy as np
import cv2
from typing import Tuple, List, Dict
import colorsys

from config import PROJECTOR_WIDTH, PROJECTOR_HEIGHT, INVERT_PATTERN


class PatternGenerator:
    def __init__(self, width: int = PROJECTOR_WIDTH, 
                 height: int = PROJECTOR_HEIGHT):
        self.width = width
        self.height = height
    
    def generate_unique_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """
        Generate n visually distinct colors
        Returns BGR tuples for OpenCV
        """
        colors = []
        for i in range(n):
            # Use HSV color space for even distribution
            hue = i / n
            saturation = 0.9
            value = 0.9
            
            # Convert to RGB
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            
            # Convert to BGR (OpenCV format) and scale to 0-255
            bgr = (int(b * 255), int(g * 255), int(r * 255))
            colors.append(bgr)
        
        return colors
    
    def generate_colored_circle_grid(self, rows: int, cols: int,
                                    radius: int, margin: int,
                                    invert: bool = INVERT_PATTERN) -> Tuple[np.ndarray, Dict[Tuple[int, int], Tuple[int, int, int]]]:
        """
        Generate grid of circles, each with unique color
        
        Returns:
            pattern: Image with colored circles
            circle_map: Dict mapping (x, y) center -> (B, G, R) color
        """
        # Background
        if invert:
            pattern = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        else:
            pattern = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Calculate positions
        usable_width = self.width - 2 * margin
        usable_height = self.height - 2 * margin
        
        x_spacing = usable_width / (cols - 1) if cols > 1 else 0
        y_spacing = usable_height / (rows - 1) if rows > 1 else 0
        
        # Generate unique colors
        n_circles = rows * cols
        colors = self.generate_unique_colors(n_circles)
        
        circle_map = {}
        idx = 0
        
        for row in range(rows):
            for col in range(cols):
                x = int(margin + col * x_spacing)
                y = int(margin + row * y_spacing)
                
                color = colors[idx]
                circle_map[(x, y)] = color
                
                # Draw colored circle with white/black border for visibility
                if invert:
                    cv2.circle(pattern, (x, y), radius, color, -1)
                    cv2.circle(pattern, (x, y), radius, (0, 0, 0), 2)  # Black border
                else:
                    cv2.circle(pattern, (x, y), radius, color, -1)
                    cv2.circle(pattern, (x, y), radius, (255, 255, 255), 2)  # White border
                
                idx += 1
        
        return pattern, circle_map
    
    def generate_test_pattern(self) -> np.ndarray:
        """Generate bright test pattern to check projector alignment"""
        pattern = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        
        # Draw border
        cv2.rectangle(pattern, (10, 10), (self.width-10, self.height-10), 
                     (255, 0, 0), 5)
        
        # Draw crosshair at center
        cx, cy = self.width // 2, self.height // 2
        cv2.line(pattern, (cx - 100, cy), (cx + 100, cy), (0, 0, 255), 3)
        cv2.line(pattern, (cx, cy - 100), (cx, cy + 100), (0, 0, 255), 3)
        
        # Add text
        cv2.putText(pattern, "PROJECTOR TEST", (cx - 200, cy - 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        
        return pattern


if __name__ == '__main__':
    gen = PatternGenerator()
    pattern, circle_map = gen.generate_colored_circle_grid(4, 6, 40, 50)
    
    print(f"Generated pattern with {len(circle_map)} colored circles")
    
    # Display
    cv2.imshow('Colored Pattern Preview', pattern)
    cv2.waitKey(0)
    cv2.destroyAllWindows()