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

# Rail inset: distance from the outer frame edge to the cushion face, in cm.
# Bank shots reflect off the cushion, not the raw frame edge.  Adjust to
# match the physical table.
RAIL_INSET_CM = 3.0
