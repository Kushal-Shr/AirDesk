# AirDesk

## Run the complete local demo

```bash
cd /Users/kushal/Projects/AirDesk
./run_airdesk.command
```

Or use `source .venv/bin/activate` followed by
`PYTHONPATH=src python -m airdesk.air_writing`.
The launcher also works by double-clicking `run_airdesk.command` in Finder.

The integrated app starts in **LIVE** mode with desktop mouse/shortcut output enabled.
Click the camera preview and press **M** to stop or resume desktop output.
**P** shows/hides the gesture guide. **Esc** stops output and returns from Air
Write to the preview. **Q** in the preview or **Ctrl+C** in Terminal quits.
Air Write text output is armed separately when you enter using both palms.
When leaving with both palms, pending ink is recognized and inserted before the
overlay closes. If it cannot be inserted confidently, Air Write remains open
and keeps the ink visible for correction or retry. AirDesk stays **LIVE** while
the Air Write overlay is open; desktop gesture recognition is merely paused.
After closing, lower both hands—a short guard ignores the release motion before
accepting a fresh desktop gesture in the focused document.

Every successful Air Write insertion is also stored in a local recovery history
at `~/Library/Application Support/AirDesk/recovery.jsonl`. AirDesk sends one
debounced **Command-S** to the focused document one second after insertion. Save
a new document and choose its filename before the demo; otherwise macOS may show
the application's Save As dialog on the first automatic save. **Save Document**
is also available from the Air Command Palette for an immediate save.

Read [DEMO.md](DEMO.md) for the rehearsal sequence, gesture reference, known
handwriting limits, and recovery steps. These are the current integrated
controls; the stage-by-stage sections below describe the project's history.

Run preflight from the project root:

```bash
PYTHONPATH=src .venv/bin/python -m airdesk.doctor --camera --native --gemini
```

This checks both models, Camera/Accessibility, the global Escape listener,
native windows, and a Gemini request using generated test text. It does not
type or click into your applications and saves no camera frames. Omit
`--gemini` for offline checks. Permissions belong to the launching app, so run
this in the same Terminal you will use during the demo.

AirDesk is a beginner-friendly macOS computer-vision project. It will grow
stage by stage into a gesture-controlled desktop interface with air-writing.

AirDesk now draws through a click-through desktop overlay, splits a complete
mixed handwriting line into characters, classifies each character independently,
and inserts the joined result after a two-second no-ink pause.

## Planned project layout

```text
airdesk/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── airdesk/
│       ├── __init__.py
│       ├── webcam_preview.py
│       ├── hand_landmarks.py
│       ├── hand_lock.py
│       ├── virtual_cursor.py
│       ├── air_mouse.py
│       ├── desktop_controls.py
│       ├── shortcut_config.py
│       ├── system_shortcuts.py
│       ├── feature_config.py
│       ├── control_panel.py
│       ├── command_palette.py
│       ├── native_overlay.py
│       ├── character_recognition.py
│       ├── emnist_model.py
│       ├── personal_samples.py
│       ├── collect_samples.py
│       ├── air_writing.py
│       └── whiteboard.py
├── scripts/
│   └── download_hand_model.py
└── tests/
    ├── .gitkeep
    ├── test_hand_lock.py
    ├── test_virtual_cursor.py
    ├── test_air_mouse.py
    ├── test_desktop_controls.py
    ├── test_system_shortcuts.py
    ├── test_feature_config.py
    ├── test_control_panel.py
    ├── test_command_palette.py
    ├── test_character_recognition.py
    ├── test_personal_samples.py
    └── test_whiteboard.py
```

Application modules will be added under `src/airdesk/` only when their build
stage begins. Tests will live in `tests/`.

## Run Stage 1

Activate the virtual environment, then start the webcam preview:

```bash
source .venv/bin/activate
python src/airdesk/webcam_preview.py
```

Press `Q` while the preview window is active to quit. The program also releases
the camera if it encounters an error or the window is closed.

The first run may make macOS request Camera access for Terminal, Python, or VS
Code. If access was denied, enable the app you used to run AirDesk under
**System Settings → Privacy & Security → Camera**, then restart that app.

## Run Stage 2

Download the official MediaPipe Hand Landmarker model once:

```bash
source .venv/bin/activate
python scripts/download_hand_model.py
```

Then run the landmark preview:

```bash
python src/airdesk/hand_landmarks.py
```

Hold one or both hands inside the camera frame. AirDesk draws all 21 landmarks,
highlights landmark `8` at each index fingertip, and shows its pixel and
normalized coordinates. Press `Q` to quit.

## Run Stage 3

Run the hand-lock preview from the project root:

```bash
source .venv/bin/activate
PYTHONPATH=src python -m airdesk.hand_lock
```

An open hand must remain visible for 0.25 seconds before its status becomes
`READY`. A closed fist changes that hand to `LOCKED` immediately. A hand that
is not visible is also treated as locked. Press `Q` to quit.

Run the safety-logic tests with:

```bash
PYTHONPATH=src python -m unittest tests/test_hand_lock.py -v
```

## Run Stage 4

Run the safe on-screen cursor preview from the project root:

```bash
source .venv/bin/activate
PYTHONPATH=src python -m airdesk.virtual_cursor
```

Move an unlocked left index finger inside the central active rectangle. The
small magenta marker is the raw fingertip location; the larger cyan dot is the
mapped and smoothed virtual cursor. The pointer activates only when the index
finger is straight while the thumb, middle, ring, and little fingers are held
down. An open palm does not move the cursor. A left fist immediately hides it.
The right hand never moves it.

The labels refer to **your anatomical left and right hands**, not which side of
the preview they occupy. AirDesk starts with the camera's handedness labels
reversed to match this setup. That correction is fixed for every launch and
cannot be accidentally toggled back. Press `Q` to quit.

Run the mapping and smoothing tests with:

```bash
PYTHONPATH=src python -m unittest tests/test_virtual_cursor.py -v
```

## Run Stage 5

Before enabling real movement, allow the app that launches Python—usually
Terminal, iTerm, or VS Code—under:

```text
System Settings → Privacy & Security → Accessibility
```

Restart that app if macOS requests it. Then run:

```bash
source .venv/bin/activate
PYTHONPATH=src python -m airdesk.air_mouse
```

Real mouse output always starts in `SAFE PREVIEW`. Verify the virtual cursor
first, then press `M` once to change the display to `AIRDESK: ACTIVE`. Press
`Esc` to disable real output immediately. `M` toggles it deliberately, and `Q`
quits. PyAutoGUI's corner fail-safe is also enabled.

The Stage 4 command remains preview-only and cannot move the real mouse:

```bash
PYTHONPATH=src python -m airdesk.virtual_cursor
```

Run Stage 5's mouse tests without touching the real pointer:

```bash
PYTHONPATH=src python -m unittest tests/test_air_mouse.py -v
```

## Run Stage 6

Stage 6 uses the same Camera and Accessibility permissions as Stage 5. It also
starts a read-only macOS event monitor for `Esc`, so the emergency stop still
works after a click gives another app focus. If Accessibility permission is
missing, Stage 6 refuses to start. Permission is checked inside the exact
Terminal, iTerm, or VS Code process launching AirDesk because a different app's
permission does not carry over.

```bash
source .venv/bin/activate
PYTHONPATH=src python -m airdesk.desktop_controls
```

Real output begins in `SAFE PREVIEW`. Gesture feedback still appears there, so
test recognition before pressing `M`:

- Quick right thumb–index pinch: left click
- Two quick right thumb–index pinches: double click
- Hold the right thumb–index pinch for about half a second: mouse-down for
  drag, drop, or text selection; release the pinch for mouse-up
- Right thumb touching both index and middle fingertips: right click
- Left index + middle fingers extended, then move vertically: scroll

The controller uses normalized thumb/fingertip distance, so the pinch thresholds
scale with the apparent size of the hand. A fist or missing right hand safely
releases an active drag. `Esc`, leaving `ACTIVE`, quitting, or an output error
also releases every held mouse button. Scroll begins only after a short dwell
and movement threshold. `Esc` disables all real output, `M` toggles it, and `Q`
quits. Handedness remains permanently swapped for this camera setup.

The bottom diagnostic shows the normalized thumb-to-index and thumb-to-middle
distances followed by the recognized pinch pose.

Real pointer movement is applied before click recognition on each frame so a
click uses the latest smoothed position. This is important for small targets
such as macOS menu-bar icons.

For double-click detection, AirDesk briefly waits after the first quick pinch
to determine whether a second pinch follows. This prevents a double click from
also producing an unwanted single click.

Run Stage 6 tests without system input:

```bash
PYTHONPATH=src python -m unittest tests/test_desktop_controls.py -v
```

## Run Stage 7

Stage 7 keeps all prior controls and adds a right-hand swipe pose: index,
middle, and ring fingers up, with the thumb and little finger down. Hold the
pose briefly, then move the whole hand:

- Up: Mission Control (`Control + Up Arrow`)
- Right: next app (`Command + Tab`)
- Left: previous app (`Command + Shift + Tab`)

Mission Control uses the native macOS application in the integrated build, so
it still works when the keyboard shortcut is disabled or remapped. To press
Enter, hold only the right index finger up for about 0.3 seconds while keeping
the thumb and other fingers folded, then release.

The shortcuts are editable in `src/airdesk/shortcut_config.py`.

```bash
source .venv/bin/activate
PYTHONPATH=src python -m airdesk.system_shortcuts
```

Start in `SAFE PREVIEW`. A recognized pose shows `RIGHT: SWIPE`; after the
dwell, the bottom diagnostic shows palm displacement as `dx` and `up`. A swipe
must exceed the palm-size-normalized threshold and have a dominant direction.
It fires once, then requires releasing the three-finger pose and observes a
cooldown. Press `M` only after preview recognition is reliable. `Esc` globally
disables real output.

Run Stage 7 tests without sending shortcuts:

```bash
PYTHONPATH=src python -m unittest tests/test_system_shortcuts.py -v
```

## Run Stage 8

Stage 8 starts in Desktop Mode with real output off. First use AirDesk to select
the text field where recognized writing should eventually be inserted. Then hold
both open palms for one second to show a transparent ink layer over the desktop:

```bash
source .venv/bin/activate
PYTHONPATH=src python -m airdesk.air_writing
```

Air-writing controls:

- Tight right thumb–index pinch with the middle, ring, and little fingers at
  least 80% open: pen down and draw
- Release the pinch or curl any of those three fingers: pen up
- Cyan ring: smoothed right index-fingertip aim position, visible even while
  the pen is up
- Filled orange dot: the complete writing pose is active and the pen is drawing
- Close the right fist: immediately lift the pen
- Left open palm held for one second: clear the canvas
- Both open palms held for one second: hide the ink and return to Desktop Mode

The OpenCV camera preview hides while Air Write mode is active. The native macOS
overlay does not accept clicks and cannot become the active window, so the app
and text field selected beforehand should retain keyboard focus. Real system
output stays off; returning to Desktop Mode does not restore it automatically.
Press `M` deliberately if needed. Use `Ctrl+C` in the launching terminal to quit
while the camera preview is hidden.

Mode switching counts any two detected palms with all four non-thumb fingers
extended; it does not depend on MediaPipe's left/right labels or thumb angle.
The overlay shows `OPEN PALMS: 0/2`, `1/2`, or `2/2`, and brief tracking
dropouts do not restart the full hold timer. The visible strokes are also kept
on an off-screen black-and-white canvas so the next stage can recognize them.

Run the Stage 8 logic tests without camera or system output:

```bash
PYTHONPATH=src python -m unittest tests/test_whiteboard.py -v
```

## Control panel and Air Command Palette

The main Air Write launcher now opens a normal, clickable macOS utility panel
alongside the existing camera/ink interface. It shows `SAFE PREVIEW`, `ACTIVE`,
or `PAUSED`, both hand states, FPS, and estimated frame-processing time. Every
available desktop control and palette command is listed with its gesture or
keyboard action.

The panel is a read-only gesture guide. It does not enable or disable anything;
the controls are available automatically. Click **Hide** to dismiss it, then
press `P` while the camera preview is active to show it again. The `M`/`Esc`
safety controls remain authoritative.

To use the Air Command Palette:

1. Hold the right index and middle fingers up while folding the thumb, ring,
   and little finger. Keep the pose for about 0.3 seconds.
2. Move the left-hand pointer over a palette row.
3. Use one right thumb–index pinch to select the highlighted row.
4. Make a right fist or remove the right hand from view to cancel.

The palette works in `SAFE PREVIEW` for visual testing, but it sends no keyboard
shortcut until AirDesk is `ACTIVE`.

### Responsiveness

The integrated launcher requests a 640×480, 30 FPS camera stream and enforces a
640×480 maximum inference size if the camera ignores that request. Camera input
runs on a latest-frame-only capture thread, so slow inference drops stale frames
instead of building visible delay. Native window events are pumped once per
frame, while the gesture guide's live text refreshes at 5 Hz and performs no
label updates while hidden. Its processing-time value now reports the previous
complete frame, including native UI work.

Run the integrated app from the project root:

```bash
source .venv/bin/activate
PYTHONPATH=src python -m airdesk.air_writing
```

Before testing real OS control, grant the launcher (Terminal, Python, or VS
Code) access in **System Settings → Privacy & Security → Accessibility**. The
integrated application begins in `LIVE` mode. Press `M` to stop or resume real
mouse/keyboard output; `Esc` always stops it.

Run the new platform-neutral tests without opening a camera or native window:

```bash
PYTHONPATH=src python -m unittest \
  tests/test_control_panel.py tests/test_command_palette.py -v
```

## Run Stage 9

Stage 9 adds on-device OCR and deliberate text insertion to the same launcher.
It is intentionally limited to one clearly drawn uppercase letter:

1. In Desktop Mode, select the destination text field.
2. Hold both open palms to enter Air Write mode.
3. Draw one character with the right thumb–index pinch.
4. Release the pinch, then hold a right-hand thumbs-up for about 0.65 seconds.
5. AirDesk recognizes the character and types it into the focused field.

The overlay reports `INSERTED`, `NOT RECOGNIZED`, or an OCR/Accessibility
error. A successful insertion clears the ink so the next character can be
drawn. The thumbs-up fires only once and must be released before another
character can be accepted. This is not digit, full-word, or sentence
recognition yet.

The included model was trained on 124,800 EMNIST letter images with rotation,
translation, scale, and shear augmentation. Its held-out EMNIST accuracy was
93.57%. To reproduce the model training from the downloaded dataset, run:

```bash
PYTHONPATH=src python scripts/train_emnist_letters.py
```

Install the machine-learning dependencies and run the latest stage with:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=src python -m airdesk.air_writing
```

Run the recognition and gesture tests without typing into another app:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Personal Training Mode

Personal Training Mode collects labeled examples of your own air writing. It
does not type into other apps and does not require Accessibility permission.
Start with five examples per lowercase letter:

```bash
source .venv/bin/activate
PYTHONPATH=src python -m airdesk.collect_samples \
  --group lowercase --samples-per-character 5
```

Hold both palms open to enter the transparent training overlay. Draw the shown
character inside the guide, release the pinch, then hold a right thumbs-up to
save it. Hold the left palm open to clear a bad attempt. The collector resumes
from existing samples if it is restarted. A cyan ring shows the current aiming
point before the pinch; it turns into a filled orange dot while drawing.

Collect symbols separately so the prompted label is always clear:

```bash
PYTHONPATH=src python -m airdesk.collect_samples \
  --group symbols --samples-per-character 5
```

The starter symbol set is:

```text
. , ? ! @ # $ % & + - _ = ( ) [ ] { } / :
```

Use `--group uppercase` to improve the current capital-letter model, or
`--group digits` to collect `0`–`9`. Use `--group all` for a combined
collection session. Samples are stored under
`personal_data/` and excluded from Git. Each sample contains a 28×28 model
image, a guide-relative image that preserves punctuation position, and JSON
stroke-path metadata.

Train four guarded, mode-specific models from the collected samples and the
local EMNIST data:

```bash
PYTHONPATH=src python scripts/train_personal_models.py
```

The trainer reserves the final personal sample for every character as a strict
test example. Uppercase, lowercase, and digit models also have to pass a held-
out EMNIST test. A model that misses either accuracy threshold is saved only as
`*.candidate.pt`; only a validated model receives the runtime-ready
`airdesk_<mode>.pt` name. Detailed accuracy and confusion results are written
to `models/personal_training_report.json`.

Run live AirDesk with whole-line mixed recognition:

```bash
PYTHONPATH=src python -m airdesk.air_writing
```

The launcher refuses a line checkpoint that did not pass validation. Write all
character types together; there is no character-mode switch.

## Segmented mixed-character recognition

The merged character trainer uses the mapped uppercase, lowercase, and symbol
images plus local EMNIST letters and digits:

```bash
PYTHONPATH=src python scripts/train_merged_characters.py
```

At runtime, vertical projection finds blank columns between characters. Each
resulting region is resized to 28×28 and passed independently through the merged
83-class CNN. Large horizontal gaps become spaces. There is no word prediction,
sentence prediction, dictionary correction, or language model.

Digits use EMNIST until personal digit samples exist. To personalize them, run
the digit collector and retrain the merged classifier:

```bash
PYTHONPATH=src python -m airdesk.collect_samples --group digits --samples-per-character 3
PYTHONPATH=src python scripts/train_merged_characters.py
```

Write a complete
mixed sentence from left to right inside the horizontal guide, lifting the
pinch between strokes and words as needed. After two seconds without new ink,
AirDesk recognizes and inserts the complete line into the previously selected
text field. No per-character thumbs-up or character-mode switching is used.
Hold the left palm open to clear before submission; `Esc` blocks automatic
insertion immediately. Automatic insertion requires at least 90% mean character
confidence; a lower-confidence result is previewed but not typed. The status
line shows the normalized pinch distance and the least-open of the three
non-writing fingers so the strict pose can be adjusted in real time.

To undo the most recent stroke, make a left thumb–index pinch while keeping the
middle, ring, and little fingers up. Hold it for about 0.4 seconds. A pinched hand is
excluded from open-palm detection, so this pose cannot accidentally clear the
line or trigger the two-palm mode switch.
Undo removes one complete pen-down-to-pen-up stroke, rebuilds the recognition
canvas, and fires only once until the pose is released.

### Improving the merged model

`--samples-per-character` is a cumulative target. If a class already has three
samples, using a target of eight collects five new examples rather than starting
over. Collect each group separately so every class gets balanced coverage:

```bash
PYTHONPATH=src python -m airdesk.collect_samples --group uppercase --samples-per-character 8
PYTHONPATH=src python -m airdesk.collect_samples --group lowercase --samples-per-character 8
PYTHONPATH=src python -m airdesk.collect_samples --group digits --samples-per-character 8
PYTHONPATH=src python -m airdesk.collect_samples --group symbols --samples-per-character 8
```

Make the examples genuinely different: vary size, slant, starting position,
stroke speed, and lighting while keeping each character legible. Then retrain
from the complete cumulative dataset and review the report:

```bash
PYTHONPATH=src python scripts/train_merged_characters.py --epochs 8
cat models/merged_character_report.json
```

Add extra examples for repeatedly confused shapes before increasing epochs.
More varied data usually improves live recognition more reliably than repeatedly
training on the same small sample set.

### Optional Gemini low-confidence review

AirDesk can send only the cropped black-and-white ink canvas to Gemini when the
local merged model scores a line below 90%. The request also includes the local
text and its three strongest candidates for each segmented character. Camera
frames, desktop screenshots, and the selected text field are never included.

Install the optional SDK, create a Gemini API key, and place it in the ignored
project `.env` file:

```bash
python -m pip install -r requirements.txt
```

```dotenv
GEMINI_API_KEY="your-key-here"
GEMINI_MODEL="gemini-3.8-flash"
GEMINI_FALLBACK_MODEL="gemini-3.5-flash-lite"
```

Then launch AirDesk normally:

```bash
PYTHONPATH=src python -m airdesk.air_writing
```

Shell environment variables still take precedence over `.env` values.
`GEMINI_MODEL` can optionally select another compatible model; the default is
`gemini-3.8-flash`. Capacity errors (`429` or `503`) automatically retry once
with `GEMINI_FALLBACK_MODEL`. Never commit an API key to the repository. Gemini
corrections at 90% confidence or higher are inserted into the previously
focused text field; lower-confidence results remain labeled `PREVIEW ONLY`.
High-confidence local results continue to use the existing automatic insertion.
Every successful Air Write insertion adds exactly one trailing space so the
next recognized word or sentence does not run into the previous text.
If the key, network, SDK, or Gemini service is unavailable, local recognition
continues normally.

## Requirements

- macOS
- Python 3.11 or newer, with compatible package wheels available
- A webcam (not used until Stage 1)

Python 3.11 is a conservative choice for compatibility with computer-vision
packages, although your current Python 3.14 environment installed the Stage 0
packages successfully. Check your version with:

```bash
python3 --version
```

## Create a virtual environment

A virtual environment is an isolated folder containing the Python interpreter
and packages for one project. It prevents AirDesk's dependencies from changing
the packages used by your other Python projects.

From the repository root, run:

```bash
cd airdesk
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `python3.11` is unavailable but `python3 --version` reports Python 3.11 or
newer, create the environment with:

```bash
python3 -m venv .venv
```

When the environment is active, the shell prompt usually begins with
`(.venv)`. Confirm the interpreter and installed packages:

```bash
python --version
python -m pip list
```

Leave the environment when you are finished:

```bash
deactivate
```

Activate it again whenever you return to AirDesk:

```bash
cd airdesk
source .venv/bin/activate
```

## Libraries we will use

- **OpenCV** reads webcam frames and draws interface overlays.
- **MediaPipe** detects hands and their 21 landmarks.
- **NumPy** performs coordinate and gesture calculations.
- **PyAutoGUI** optionally sends real mouse movement beginning in Stage 5.

Real system input remains off by default. Before Stage 5 can control the
mouse or keyboard, macOS may require permission for Terminal, Python, or VS
Code under **System Settings → Privacy & Security → Accessibility**.

Camera permission is not needed during Stage 0. macOS may ask for it when the
Stage 1 webcam preview is first run.

## Safety roadmap

AirDesk will be developed in safe preview mode first. No real mouse or keyboard
events are implemented in this stage. Later stages will add explicit hand
locks, an emergency `Esc` stop, and a deliberate `M` toggle before system
control is enabled.
