"""Small shared utilities for application setup and diagnostics."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

from src.config import PROJECT_ROOT


def setup_logging() -> None:
    """Configure concise, timestamped console logging once per process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("ultralytics").setLevel(logging.WARNING)


def ensure_project_directories() -> None:
    """Create runtime output directories if a clean clone does not contain them."""
    for relative_path in (
        "models",
        "assets/images",
        "assets/videos",
        "output/screenshots",
        "output/recordings",
        "runs",
    ):
        (PROJECT_ROOT / relative_path).mkdir(parents=True, exist_ok=True)


def configure_opencv() -> None:
    """Apply safe CPU-oriented OpenCV settings before opening the webcam."""
    cv2.setUseOptimized(True)
