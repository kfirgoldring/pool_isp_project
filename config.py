"""Configuration for billiards assistance system"""

# Hardware settings
CAMERA_INDEX = 0 # index of the camera to use

# Table dimensions (in cm or any unit)
TABLE_WIDTH_CM = 122.0
TABLE_HEIGHT_CM = 61.0

#Ball radius (in cm)
BALL_RADIUS_CM = 1.5
# Table boundary margin (in table coordinates) - reject circles near edge
TABLE_MARGIN_CM = 5.0

# Calibration data storage
CAMERA_CALIB_FILE = 'data/calibration/camera_homography.npy'
