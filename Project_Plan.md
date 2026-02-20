# Billiards Assistance System - Project Plan
**Duration:** 1 week | **Team Size:** 4 | **Stack:** Python + OpenCV

---

## Quick Status Reference
👉 **See `STATUS.md` for current status**
👉 **See `logs/` for progress history**

---

## Approach: MVP First

The goal is a working minimum viable system by end of week. Only add features if the MVP is stable and time allows.

**MVP (must have):**
- Capture a frame from the overhead camera
- Detect all balls and identify the cue ball
- User selects a target ball
- Calculate a direct straight-line path from cue ball to target
- Display the result in the GUI

**Stretch goals (only if time allows):**
- Real-time ball tracking
- Cue stick detection
- Bank shot calculation
- Shot suggestions
- Pocket detection

---

## Team Roles & Responsibilities

### Person 1: Ball Detection Lead
**Primary:** Ball detection and color identification
**Contact for:** Detection accuracy, false positives, color classification

### Person 2: Scene Understanding Lead
**Primary:** Table calibration, homography, pocket detection
**Contact for:** Calibration issues, coordinate transformations

### Person 3: Physics & Trajectory Lead
**Primary:** Straight-line trajectory calculation, collision math
**Contact for:** Path calculation, angle math, stretch goal physics

### Person 4: GUI Lead
**Primary:** GUI that displays the camera view, detected balls, and trajectory overlays
**Contact for:** Visualization bugs, integration issues, UI/UX

---

## Week Timeline

### Days 1-2: Individual Modules
**Goal:** Each person has their module working on static images

- Person 1: Ball detection + color identification on a test image
- Person 2: Camera calibration + table homography
- Person 3: Straight-line path from cue ball to target ball
- Person 4: Basic GUI window showing camera feed + overlays

### Days 3-4: Integration
**Goal:** End-to-end pipeline on a static image

- Connect detection → trajectory → GUI
- User can click to select target ball
- Path is drawn on the GUI

### Day 5: Testing & Polish
**Goal:** System works reliably on real table images

- Fix bugs from integration
- Clean up UI
- If time: start on stretch goals

---

## Critical Path

```
Days 1-2: All modules developed in parallel
Day 3:    Integration — depends on ALL Day 1-2 modules
Days 4-5: Polish and stretch goals
```

⚠️ **Integration on Day 3 is the bottleneck.** Each person must have a working, callable module by end of Day 2.

---

## Success Criteria

### MVP ✔
- [ ] Ball detection working on static image
- [ ] Cue ball correctly identified
- [ ] Straight-line trajectory from cue ball to selected target ball
- [ ] GUI displays camera view with balls and path overlaid

### Stretch Goals ⭐ (only if MVP is done)
- [ ] Real-time ball tracking
- [ ] Cue stick detection
- [ ] Bank shot calculation
- [ ] Shot suggestions
- [ ] Pocket detection

---

## Risk Register

### Risk 1: Calibration Accuracy 🔴 HIGH
**Impact:** High | **Probability:** Medium
**Mitigation:** Start calibration Day 1, use existing calibration code, manual fallback
**Owner:** Person 2

### Risk 2: Integration Delays 🔴 HIGH
**Impact:** High | **Probability:** Medium
**Mitigation:** Define interfaces between modules on Day 1, test with dummy data
**Owner:** All

### Risk 3: Detection Robustness 🟡 MEDIUM
**Impact:** Medium | **Probability:** Medium
**Mitigation:** Use test images from actual table, tune HSV thresholds early
**Owner:** Person 1

---

## Git Workflow

```bash
# Main branches
main          - Working code only
develop       - Integration branch

# Feature branches
feature/ball-detection    (Person 1)
feature/calibration       (Person 2)
feature/physics           (Person 3)
feature/gui               (Person 4)
```

### Commit Convention:
```
[Module] Type: Description

Examples:
[Detection] feat: Add HSV color classification
[Physics] fix: Trajectory angle calculation
[GUI] feat: Add ball overlay on camera view
[Integration] merge: Connect detection to GUI
```

---

## Module Interfaces

Each module must expose a clean, simple function so integration is straightforward.

```python
# Person 1
detect_balls(image: np.ndarray) -> List[Dict]
# Returns: [{'center': (x,y), 'radius': int, 'color': str, 'is_cue': bool}, ...]

# Person 2
get_homography(camera_index: int) -> np.ndarray
transform_to_table(point, homography) -> (x, y)

# Person 3
calculate_path(cue_ball: Dict, target_ball: Dict) -> List[Tuple]
# Returns: list of (x, y) points representing the path

# Person 4
draw_overlay(image: np.ndarray, balls: List[Dict], path: List[Tuple]) -> np.ndarray
# Returns: image with balls and path drawn on it
```

---

## Project Structure

```
billiards_project/
├── PROJECT_PLAN.md          ← This file
├── STATUS.md                ← Current status (update daily)
├── BLOCKERS.md              ← Active blockers
│
├── calibration/             ← Calibration code (Person 2)
│   ├── table_calibration.py
│   └── billiards_calibration_merged.py
│
├── detection/               ← Ball detection (Person 1)
│   └── ball_detector.py
│
├── physics/                 ← Trajectory calculation (Person 3)
│   └── trajectory.py
│
├── gui/                     ← GUI (Person 4)
│   └── app.py
│
├── utils/                   ← Shared utilities
│   └── geometry.py
│
├── data/                    ← Test images and calibration data
│   ├── test_images/
│   └── calibration_data/
│
├── main.py                  ← Main application entry point
├── config.py                ← Configuration settings
└── requirements.txt         ← Dependencies
```

---

## Getting Help from Claude

1. Describe what you're working on and what's broken
2. Share the relevant code snippet or error message
3. Reference the specific file and function

**Example:**
```
"Working on ball detection. detect_balls() returns empty list on test image.
See detection/ball_detector.py line 45.
Error: no circles detected after HoughCircles."
```

---

## Coordinate System Contract

All modules share four distinct coordinate spaces. Each module may only read/write the spaces listed in its column.

| Space | Unit | Range | Used by | Notes |
|---|---|---|---|---|
| **Camera pixels** | px | 0..frame_w × 0..frame_h | `Ball_Detection.py` output, `app.py` click→table transform | Raw camera frame, NOT warped |
| **Table centimeters** | cm | X: 0..122, Y: 0..61 | `golf_game.py`, `trajectory.py`, `main.py` adapter, `app.render()` inputs | Origin = top-left pocket |
| **Display pixels** | px | 0..976 × 0..488 | `app.py` rendering internals only | Table cm × `TABLE_DISPLAY_SCALE` (8) |
| **Projector pixels** | px | 0..1920 × 0..1080 | `app.py` projection output only | Via `proj_H_inv` homography |

### Rules

1. **`main.py` is the only module that converts camera pixels → table cm.**
   This conversion uses `cam_H` (the camera→table homography) via `cv2.perspectiveTransform`.
   The adapter function `_adapt_detections(raw, cam_H)` does this conversion.

2. **`golf_game.py` and `trajectory.py` never see camera pixels or display pixels.**
   Every argument they receive is already in table centimeters.

3. **`app.py` converts table cm → display px internally** (multiply by `TABLE_DISPLAY_SCALE = 8`).
   No other module performs this conversion.

4. **Pocket positions are always derived from table dimensions** — never hardcoded as arbitrary numbers:

   ```python
   TABLE_WIDTH_CM  = 122.0
   TABLE_HEIGHT_CM = 61.0
   # 6 pockets: corners + mid-long-edges
   POCKET_POSITIONS_CM = [
       (k * TABLE_WIDTH_CM / 2, j * TABLE_HEIGHT_CM)
       for j in (0, 1) for k in (0, 1, 2)
   ]
   # → [(0,0), (61,0), (122,0), (0,61), (61,61), (122,61)]
   ```

### Architecture (as-implemented)

```
main.py  ←→  app.py          (grab_frame, render, consume_pending_clicks)
  │
  ├── Ball_Detection.py       detect_balls_with_color(frame, table_corners, ...)
  │   └─ returns (N,3) numpy: [x_cam_px, y_cam_px, color_str]
  │
  ├── _adapt_detections()     converts numpy → List[Dict] with center_cm
  │
  ├── golf_game.py            GolfGame.update(balls)      — all cm
  └── trajectory.py           calculate_path / suggest_best_shot  — all cm
```

### Ball detection interface (`Ball_Detection.py`)

```python
detect_balls_with_color(
    frame_bgr:      np.ndarray,
    table_corners:  Optional[Iterable],   # 4 camera-pixel corner points
    ref_path:       str = "ref.jpeg",     # empty-table reference image
    table_size_cm:  Optional[Tuple[float,float]] = None,
) -> np.ndarray   # shape (N, 3): [[x_cam_px, y_cam_px, color_str], ...]
```

Ball colors returned: `'white'`, `'orange'`, `'yellow'`, `'blue'`, `'bordeaux'`, `'unknown'`

**Important:** `Ball_Detection.py` must **not** be modified. It takes raw camera frames and corner points; it does not apply any homography.

### Adapter layer (`main.py → _adapt_detections`)

```python
def _adapt_detections(raw: np.ndarray, cam_H: Optional[np.ndarray]) -> List[Dict]:
    # Converts detect_balls_with_color() output to internal dict format.
    # Each output dict:
    #   'center':    (int, int)              # camera pixel coords
    #   'center_cm': (float, float) | None   # table cm coords (via cam_H)
    #   'color':     str
    #   'is_cue':    bool  (color == 'white')
    #   'radius':    int   (fixed at 18 px — Ball_Detection does not expose this)
```

### Flags for real-world validation

- **`MOVEMENT_THRESHOLD_CM = 0.625`** — derived from 5 px / 8 px-per-cm. Must be validated with real camera footage before finalising.
- **`ref.jpeg`** — Ball_Detection requires an empty-table reference image at the project root. Confirm filename with Person 1 before deployment.

---

## Resources

- Hough Circle Transform: https://docs.opencv.org/4.x/d4/d70/tutorial_hough_circle.html
- Camera Calibration: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- Homography: https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html

---

**Last Updated:** 2026-02-19
