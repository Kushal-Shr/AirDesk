# AirDesk demo rehearsal

For a word-for-word presentation with timing, stage directions, and a
ninety-second fallback, use [DEMO_SCRIPT.md](DEMO_SCRIPT.md). This file remains
the detailed gesture and recovery reference.

Use the `additional-features` branch and the existing `.venv`. Start from the
project root with `./run_airdesk.command`. Keep the laptop on power and use good
front lighting. Place the demo text field on the Mac's primary display; the
cursor, overlay, and palette target that display. If using a projector, screen
mirroring is the simplest setup.

## Before presenting

1. Quit any old AirDesk process and close other apps using the camera.
2. Run `PYTHONPATH=src .venv/bin/python -m airdesk.doctor --camera --native --gemini`.
   Each requested check should say PASS. Gemini requires internet and a working
   `.env` key; this command sends only a generated test image, not camera pixels.
3. Open a new plain-text document in TextEdit. Disable smart substitutions if
   you need literal punctuation. Keep personal documents out of the demo.
4. Start AirDesk: desktop controls start LIVE. Release your hands to neutral
   before trying gestures. Hide the gesture guide with P in the preview if it overlaps
   your target. Hand labels are permanently swapped for this camera setup.
5. Rehearse the same short line twice. Recognition is not guaranteed for every
   glyph or arbitrary handwriting; use your rehearsal to choose the demo text.

## Suggested three-minute sequence

| Demonstrate | Gesture / expected result |
| --- | --- |
| Pointer | Left index only, other fingers down. Move within the active rectangle. |
| Left click | Quick right thumb–index pinch, then release. The first click fires immediately. |
| Double click | Two quick right thumb–index pinches; release between them. The second is sent with the native macOS double-click count. |
| Drag / text selection | Hold the right pinch for about 0.32 seconds, move the left pointer, then release the right pinch. |
| Right click | Touch the right thumb to both index and middle fingertips and hold briefly. |
| Scroll | Left index and middle up, remaining fingers down; hold briefly, then move vertically. |
| Command palette | Hold the right index and middle fingers up with thumb, ring, and little folded for about 0.3 seconds. Highlight with the left pointer, then pinch with the right hand. Right fist cancels. |
| App switching | Right index, middle, ring up; thumb and little down. Hold briefly, swipe left/right. Swipe up for Mission Control. |
| Enter / Return | Hold only the right index finger up for about 0.3 seconds, with thumb and other fingers folded. Release before using it again. Works in Desktop and Air Write after the current line has inserted. |
| Air Write | Focus the text field, then hold both open palms for about 0.6 seconds. The preview disappears. |
| Write a short line | Right thumb–index pinch with middle, ring, little at least 80% extended. Release between strokes. Write separate printed characters left to right. |
| Undo ink | Hold a left thumb–index pinch with middle, ring, little extended for about 0.4 seconds. Release before undoing again. |
| Insert | Release the writing pinch and wait two seconds. Uncertain local output goes to Gemini; sufficiently confident output is inserted with one trailing space. |
| Recognize now / retry | Hold right thumbs-up about 0.35 seconds, then release. This retries a retained line without requiring another stroke. Confidence checks still apply. |
| Save document | Focus the intended document, then choose Save Document from the Air Command Palette. AirDesk never sends Command-S automatically after insertion. |
| Clear ink | Left open palm for about 0.55 seconds, with the right hand not open. |
| Return / stop | Both palms for about 0.6 seconds. Pending ink is inserted first; Air Write closes only after the text is safe in the focused app. AirDesk remains LIVE; lower both hands while a brief guard ignores the release motion, then use a fresh gesture. Uncertain ink stays visible for retry. Esc stops all output and returns to Desktop; press M only after an Esc stop. |
| Quit | Q with the preview focused, or Ctrl+C in the launching Terminal. |

## Recovery during the demo

- No mouse movement: release the gesture, then point again. If the display says
  STOPPED, click the preview and press M. A fist or missing hand locks that hand;
  a visible non-fist unlocks it after roughly 0.12 seconds. Moving into a screen corner triggers the
  PyAutoGUI stop; move away with the trackpad and re-enable M.
- Writing works but no text: confirm the destination field has keyboard focus.
  The cyan/orange dot is the drawing pointer, not a text-field focus indicator.
  Low-confidence output intentionally remains in the preview. Thumbs-up retries;
  the left palm clears it. Esc disarms text until you re-enter Air Write.
- Gemini unavailable: the ink and local preview are retained. Retry with
  thumbs-up after connectivity returns. Each remote request has a 10-second
  timeout and a fallback for temporary server/network errors. The configured
  primary is `gemini-3.5-flash-lite`; the app remembers a successful fallback
  for later lines. High-
  confidence local results can still insert without Gemini.
- Missing hand model: `.venv/bin/python scripts/download_hand_model.py`.
- Missing merged model: restore `models/airdesk_merged_characters.pt` from a
  backup. Retraining is available via `scripts/train_merged_characters.py`, but
  avoid changing the model immediately before presenting.
- Permission failures: enable Camera and Accessibility for the launching app
  (usually Terminal) in System Settings → Privacy & Security, then restart it.
  Global key monitoring may also require Input Monitoring if macOS requests it.

## What has been checked

The review includes deterministic regression tests for pointer mapping,
gesture timing, clicks, drag release, palette actions, undo, mode transitions,
text insertion, Gemini errors, and startup cleanup. Live checks exercised camera
tracking, model loading, native windows, the configured Gemini fallback, and an
isolated native text field for exact mixed-case/symbol insertion, double-click
event counts, and drag events.

These checks cannot certify your physical gestures or recognition of new
handwriting. The current merged checkpoint reports 98.6% on personal samples
also used for fitting and 73.2% on its EMNIST evaluation subset. The personal
score is not held-out accuracy. Separate characters clearly, make spaces
larger than letter gaps, keep the writing on one baseline, and avoid cursive
or overlapping characters. Gemini's reported confidence is also not an
accuracy guarantee.

The model files, `.env`, and personal samples are locally ignored by Git. A
Git checkout alone does not include them. Keep a local backup of `models/`
and `personal_data/`; keep your API key private.

For a repeatable isolated macOS input check, run
`.venv/bin/python scripts/smoke_macos_input.py`. It opens its own temporary
field, verifies focus before sending input, and restores your previous app.
