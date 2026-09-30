# AirDesk

AirDesk is a beginner-friendly macOS computer-vision project. It will grow
stage by stage into a gesture-controlled desktop interface and an air-writing
whiteboard.

This repository currently contains **Stage 1: webcam preview**. It opens a safe,
mirrored camera preview with an FPS display. It does not track hands or control
macOS.

## Planned project layout

```text
airdesk/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── airdesk/
│       ├── __init__.py
│       └── webcam_preview.py
└── tests/
    └── .gitkeep
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
- **PyAutoGUI** will optionally send mouse and keyboard input in a later stage.

Real system input will remain off by default. Before Stage 5 can control the
mouse or keyboard, macOS may require permission for Terminal, Python, or VS
Code under **System Settings → Privacy & Security → Accessibility**.

Camera permission is not needed during Stage 0. macOS may ask for it when the
Stage 1 webcam preview is first run.

## Safety roadmap

AirDesk will be developed in safe preview mode first. No real mouse or keyboard
events are implemented in this stage. Later stages will add explicit hand
locks, an emergency `Esc` stop, and a deliberate `M` toggle before system
control is enabled.
