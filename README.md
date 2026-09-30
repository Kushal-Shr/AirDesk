# AirDesk

AirDesk is a beginner-friendly macOS computer-vision project. It will grow
stage by stage into a gesture-controlled desktop interface and an air-writing
whiteboard.

This repository currently contains **Stage 0 only: project setup**. It does not
open the webcam, track hands, or control macOS yet.

## Planned project layout

```text
airdesk/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── airdesk/
│       └── __init__.py
└── tests/
    └── .gitkeep
```

Application modules will be added under `src/airdesk/` only when their build
stage begins. Tests will live in `tests/`.

## Requirements

- macOS
- Python 3.11, 3.12, or 3.13
- A webcam (not used until Stage 1)

Python 3.11 is a conservative choice for compatibility with the computer-
vision packages used by this project. Check your version with:

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

If `python3.11` is unavailable but `python3 --version` reports Python 3.11,
3.12, or 3.13, create the environment with:

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

