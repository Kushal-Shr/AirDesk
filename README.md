# AirDesk

AirDesk is a beginner-friendly macOS computer-vision project. It will grow
stage by stage into a gesture-controlled desktop interface with air-writing.

This repository currently contains **Stage 9: One-Character Recognition**.
AirDesk draws through a click-through desktop overlay, recognizes one uppercase
letter with an EMNIST-trained neural network, and types it only after a
thumbs-up hold.

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
│       ├── native_overlay.py
│       ├── character_recognition.py
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
    ├── test_character_recognition.py
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
the preview they occupy. If your camera reports them in reverse, press `H`; the
status panel changes from `HAND LABELS: NORMAL` to `HAND LABELS: SWAPPED`.
Changing this setting safely relocks both hands. Press `Q` to quit.

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

- Right index finger only, held briefly: left click
- Right index + middle fingers, held briefly: right click
- Left index + middle fingers extended, then move vertically: scroll

Each click pose must dwell briefly, fires once per release, and has a cooldown.
Scroll begins only after a short dwell and movement threshold. A fist or missing
hand cancels that hand's gesture state. `Esc` disables all real output, `M`
toggles it, `H` corrects hand labels, and `Q` quits.

The bottom diagnostic shows whether the right index, middle, ring, little, and
thumb are interpreted as `UP` or `DOWN`, followed by the recognized click pose.
Brief one-frame landmark dropouts are tolerated.

Real pointer movement is applied before click recognition on each frame so a
click uses the latest smoothed position. This is important for small targets
such as macOS menu-bar icons.

Click poses contain at least one extended finger, so they no longer conflict
with closed-fist locking. Lower the extended finger or fingers after each click
to re-arm the one-shot detector.

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

- Right thumb–index pinch held: pen down and draw
- Release the right pinch: pen up
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
