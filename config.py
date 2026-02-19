"""Configuration for projector calibration system"""

# Hardware settings
CAMERA_INDEX = 0
PROJECTOR_WIDTH = 1920
PROJECTOR_HEIGHT = 1080
MAIN_MONITOR_WIDTH = 1920  # Your laptop/main screen width (for window positioning)

# Calibration pattern settings - will try multiple configurations
PATTERN_CONFIGS = [
    {'rows': 4, 'cols': 6, 'radius': 40, 'margin': 50},   # First try: 24 circles
    {'rows': 3, 'cols': 5, 'radius': 50, 'margin': 80},   # Second try: 15 circles, larger
    {'rows': 5, 'cols': 7, 'radius': 35, 'margin': 40},   # Third try: 35 circles, smaller
    {'rows': 3, 'cols': 4, 'radius': 60, 'margin': 100},  # Fourth try: 12 circles, very large
]

INVERT_PATTERN = True  # Black circles on white = brighter projection

# Detection settings
MIN_CIRCLE_RADIUS = 8
MAX_CIRCLE_RADIUS = 100
MIN_REQUIRED_MATCHES = 12  # Minimum circles needed for calibration

# Table dimensions (in cm or any unit)
TABLE_WIDTH_CM = 122.0
TABLE_HEIGHT_CM = 61.0

# Table boundary margin (in table coordinates) - reject circles near edge
TABLE_MARGIN_CM = 5.0

# Calibration data storage
CAMERA_CALIB_FILE = 'data/calibration/camera_homography.npy'
PROJECTOR_CALIB_FILE = 'data/calibration/projector_homography.npy'