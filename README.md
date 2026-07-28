# HackTronix 2.0 — Real-Time Ball Detection

A CPU-friendly webcam application for Task 1: Ball Detection. It runs a small
Ultralytics YOLO model on each frame, displays ball boxes and confidence, and
reports stable moving-average FPS plus per-frame inference latency.

The default `yolo11n.pt` model is a general COCO model. It detects the COCO
`sports ball` class, which is a strong immediate baseline but is not tuned to a
specific competition ball, lighting setup, or distance. A custom fine-tuned
model labelled `ball` can be placed at `models/yolo11n.pt` (or selected with
`BALL_DETECTOR_MODEL_PATH`) to improve F1 score.

## Folder structure

```text
BallDetection/
├── main.py
├── requirements.txt
├── README.md
├── models/                 # YOLO weights download here on first run
├── src/
│   ├── camera.py           # Webcam lifecycle and disconnect handling
│   ├── detector.py         # One-time YOLO load and ball inference
│   ├── draw.py             # Boxes and on-screen HUD
│   ├── fps.py              # Moving-average FPS counter
│   ├── config.py           # Central defaults and environment overrides
│   └── utils.py            # Logging and runtime setup
├── assets/
│   ├── images/
│   └── videos/
├── output/
│   ├── screenshots/
│   └── recordings/
└── runs/
```

## Installation

Use Python 3.11 or newer. From the project folder in VS Code's integrated
terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, run the remaining command
with the interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

## Run

```powershell
python main.py
```

On first launch, the application downloads the compact YOLO11 nano weights to
`models/yolo11n.pt`; an internet connection is needed only for that download.
It then opens webcam index 0. Press **Q** in the video window to exit cleanly.

Select another camera or tune the confidence threshold when needed:

```powershell
python main.py --camera 1 --confidence 0.40
```

The same settings can be configured without source edits using
`BALL_DETECTOR_CAMERA_ID`, `BALL_DETECTOR_CONFIDENCE`,
`BALL_DETECTOR_IOU`, `BALL_DETECTOR_FRAME_WIDTH`,
`BALL_DETECTOR_FRAME_HEIGHT`, and `BALL_DETECTOR_MODEL_PATH` environment
variables.

## Expected output

The window shows a green bounding box and label such as `sports ball: 82%` for
every detected ball. The top-left panel shows the ball count, moving-average
FPS, inference milliseconds, and the quit key. Startup logs report model
loading, camera selection, and any recoverable setup problem.

## Design notes

- The YOLO model loads once during startup, never per frame.
- Inference uses the CPU and the smallest YOLO11 model by default.
- Frames are passed directly to YOLO and annotated in place, avoiding needless
  full-frame copies.
- YOLO preserves source-frame coordinates, so boxes align with the webcam view.
- Camera and model failures are handled with readable messages and resources
  are always released on exit.

## Future improvements

- Fine-tune YOLO on labelled images from the exact challenge environment.
- Add a confidence/IoU calibration script against a held-out validation set.
- Add tracking to reduce flicker and improve temporal consistency.
- Add asynchronous capture and optional recording for higher-end CPUs.
- Benchmark input size and model variants for the target judging machine.
