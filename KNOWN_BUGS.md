# OptiCue — Known Bugs & Pending Work

## UI / Layout

### 1. Font still small in places
**Symptom:** Despite increasing base font sizes (QLabel 13 px, QPushButton 14 px, headers 11 px),
text may still look thin or hard to read on high-DPI Windows displays.
**Likely cause:** Windows DPI scaling (125 % / 150 %) multiplies logical pixels, making
point-sized fonts render smaller than expected; additionally, custom Playfair Display / Lora /
DM Sans fonts may not be loading from `assets/fonts/` and Qt falls back to a thinner system font.
**To investigate:** Confirm font files exist in `assets/fonts/`; add debug print in
`_load_opticue_fonts()` to log which IDs were loaded; increase font sizes further or switch
remaining labels to `font-weight: 600`.

### 2. Some sidebar buttons may still not appear (main view)
**Symptom:** In the main sidebar, the lower CONTROLS buttons (Capture, Switch to Projection,
Re-calibrate) may be cut off at the bottom of the window.
**Likely cause:** Even after compacting STROKES/REMAINING to side-by-side and collapsing STATUS
to 2 lines, the total sidebar content can still exceed the window height on smaller screens or
at higher DPI. The `btn_ghost` / `btn_accent` QSS styles may also cause buttons to render as
invisible if the object-name isn't reapplied after a style change.
**To fix:** Wrap the entire sidebar `QWidget` in a `QScrollArea` (vertical scroll, hidden
scrollbar) as a safety net; or increase `setMinimumSize` further; re-check that
`setObjectName` + `setStyleSheet('')` pattern correctly re-evaluates QSS on Windows.

### 3. UI does not adapt to window size (no responsive layout)
**Symptom:** Resizing the window does not reflow the sidebar or scale the camera feed
gracefully; at small sizes content is clipped, at large sizes the sidebar leaves dead space.
**To fix:** Switch the main sidebar from a fixed `setFixedWidth(260)` to a percentage-based
constraint or add a `QSplitter` between the camera view and sidebar so the user can
drag the divider; use `QScrollArea` for sidebar content.

---

## Features / Pipeline

### 4. Projection pipeline incomplete
**Symptom:** Switching to "Projection" mode shows the overlay on the projector window, but
the projector–table homography calibration (`billiards_calibration_merged`) is not always
available or may produce incorrect alignment.
**Details:**
- `ProjectorCalibrator.calibrate()` depends on `billiards_calibration_merged`, which may not
  be present in all environments.
- The fallback (`proj_H_inv = None`) causes "NO PROJECTOR CALIBRATION" to display on the
  projector canvas.
- Trajectory lines and ball markers in projection mode are untested end-to-end.
**To do:** Complete the projector calibration flow; add a "Test projection" button that draws
a fixed crosshair to verify alignment; expose `proj_H_inv` in the status panel.

### 5. Ball detection accuracy / robustness
**Symptom:** Ball detection (via `Ball_Detection.py`) can produce false positives (phantom
balls), miss balls under poor lighting, or mis-classify colors.
**Details:**
- Contour-based detection is sensitive to table cloth color, lighting uniformity, and ball
  occlusions.
- The reference-frame subtraction (auto-captured from the last calibration frame) may be
  stale if lighting changes between calibration and play.
**To do:** Add a "Recapture Reference" button in the main view; tune HSV thresholds in
`Ball_Detection.py`; consider adding confidence scores or minimum-area filters.

### 6. Mock / demo mode is minimal
**Symptom:** When no camera is connected and mock mode is active, `_make_mock_frame()` returns
a static synthetic frame with no ball positions — game logic cannot be exercised.
**To do:** Generate synthetic moving balls in `_make_mock_frame()` to allow full UI testing
without hardware.

---

## Minor / Style

### 7. QGroupBox title background may still show on some Qt/Windows themes
**Symptom:** Despite `background-color: transparent` on `QGroupBox::title`, some Qt styles
(e.g., WindowsVistaStyle) paint a white or grey rectangle behind the title text.
**To fix:** Force `QApplication.setStyle('Fusion')` at startup so the flat QSS rules take
effect uniformly across all Windows versions.

### 8. Start Game button object-name flicker on repeated state changes
**Symptom:** The `btn_start_game` text and colour (orange "Start Game" ↔ green "Pause/Resume")
update correctly but may flicker briefly because `setObjectName()` + `setStyleSheet('')`
forces a full QSS re-parse on every status-panel refresh (called at ~30 fps).
**To fix:** Cache the last button state string; only call `setObjectName` / `setStyleSheet`
when the state actually changes.
