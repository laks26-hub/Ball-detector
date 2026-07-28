"""Configuration module for the YOLO11n real-time Ball Detection System.

This module organizes all settings into logical sections using Python dataclasses,
providing automatic hardware detection (CUDA) and input validation.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

# Detect hardware acceleration availability
try:
    import torch
    HAS_CUDA: bool = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False


@dataclass
class ModelConfig:
    """YOLO model inference configuration."""
    MODEL_PATH: Path = Path("models/yolo11n.pt")
    CONFIDENCE_THRESHOLD: float = 0.35
    IOU_THRESHOLD: float = 0.45
    TARGET_CLASS_NAME: str = "ball"
    DEVICE: str = "cuda" if HAS_CUDA else "cpu"
    HALF_PRECISION: bool = HAS_CUDA

    def validate(self) -> None:
        """Validates model configuration parameters."""
        if not (0.0 <= self.CONFIDENCE_THRESHOLD <= 1.0):
            raise ValueError(
                f"CONFIDENCE_THRESHOLD must be in range [0.0, 1.0], got {self.CONFIDENCE_THRESHOLD}"
            )
        if not (0.0 <= self.IOU_THRESHOLD <= 1.0):
            raise ValueError(
                f"IOU_THRESHOLD must be in range [0.0, 1.0], got {self.IOU_THRESHOLD}"
            )

    def __post_init__(self) -> None:
        self.validate()


@dataclass
class CameraConfig:
    """Camera hardware and capture configuration."""
    CAMERA_INDEX: int = 0
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    TARGET_FPS: int = 60
    BUFFER_SIZE: int = 1

    def validate(self) -> None:
        """Validates camera configuration parameters."""
        if self.CAMERA_INDEX < 0:
            raise ValueError(f"CAMERA_INDEX must be >= 0, got {self.CAMERA_INDEX}")
        if self.FRAME_WIDTH <= 0 or self.FRAME_HEIGHT <= 0:
            raise ValueError("FRAME_WIDTH and FRAME_HEIGHT must be positive integers")
        if self.TARGET_FPS <= 0:
            raise ValueError(f"TARGET_FPS must be a positive integer, got {self.TARGET_FPS}")
        if self.BUFFER_SIZE <= 0:
            raise ValueError(f"BUFFER_SIZE must be a positive integer, got {self.BUFFER_SIZE}")

    def __post_init__(self) -> None:
        self.validate()


@dataclass
class DisplayConfig:
    """GUI window and overlay rendering configurations."""
    SHOW_FPS: bool = True
    SHOW_CONFIDENCE: bool = True
    SHOW_CENTER_POINT: bool = True
    WINDOW_NAME: str = "Ball Detection"


@dataclass
class DrawingConfig:
    """Visualization drawing styles (colors in BGR format)."""
    BBOX_COLOR: Tuple[int, int, int] = (0, 255, 0)          # BGR: Green
    TEXT_COLOR: Tuple[int, int, int] = (255, 255, 255)      # BGR: White
    CENTER_POINT_COLOR: Tuple[int, int, int] = (0, 0, 255)  # BGR: Red
    FONT_SCALE: float = 0.5
    LINE_THICKNESS: int = 2
    CIRCLE_RADIUS: int = 5

    def validate(self) -> None:
        """Validates drawing parameters."""
        for name, color in [
            ("BBOX_COLOR", self.BBOX_COLOR),
            ("TEXT_COLOR", self.TEXT_COLOR),
            ("CENTER_POINT_COLOR", self.CENTER_POINT_COLOR),
        ]:
            if not isinstance(color, tuple) or len(color) != 3 or not all(0 <= c <= 255 for c in color):
                raise ValueError(
                    f"{name} must be a BGR Tuple[int, int, int] containing values 0-255, got {color}"
                )
        if self.FONT_SCALE <= 0.0:
            raise ValueError(f"FONT_SCALE must be positive, got {self.FONT_SCALE}")
        if self.LINE_THICKNESS <= 0:
            raise ValueError(f"LINE_THICKNESS must be positive, got {self.LINE_THICKNESS}")
        if self.CIRCLE_RADIUS <= 0:
            raise ValueError(f"CIRCLE_RADIUS must be positive, got {self.CIRCLE_RADIUS}")

    def __post_init__(self) -> None:
        self.validate()


@dataclass
class PerformanceConfig:
    """Pipeline performance and queue configurations."""
    FPS_AVERAGE_WINDOW: int = 30
    ENABLE_THREADING: bool = True
    ENABLE_TRACKER: bool = False
    MAX_QUEUE_SIZE: int = 2

    def validate(self) -> None:
        """Validates performance metrics."""
        if self.FPS_AVERAGE_WINDOW <= 0:
            raise ValueError(
                f"FPS_AVERAGE_WINDOW must be positive, got {self.FPS_AVERAGE_WINDOW}"
            )
        if self.MAX_QUEUE_SIZE <= 0:
            raise ValueError(
                f"MAX_QUEUE_SIZE must be positive, got {self.MAX_QUEUE_SIZE}"
            )

    def __post_init__(self) -> None:
        self.validate()


@dataclass
class OutputConfig:
    """Video/screenshot saving and file logging settings."""
    SAVE_VIDEO: bool = False
    SAVE_SCREENSHOTS: bool = False
    OUTPUT_DIRECTORY: Path = Path("output")
    LOG_DIRECTORY: Path = Path("output/logs")


@dataclass
class DebugConfig:
    """Logging, verbosity, and runtime profiling settings."""
    DEBUG_MODE: bool = False
    VERBOSE: bool = False
    PRINT_INFERENCE_TIME: bool = False


@dataclass
class SystemConfig:
    """Root configuration aggregator for the entire ball detection system."""
    model: ModelConfig = field(default_factory=ModelConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    drawing: DrawingConfig = field(default_factory=DrawingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    def validate(self) -> None:
        """Trigger-validates all sub-configurations."""
        self.model.validate()
        self.camera.validate()
        self.drawing.validate()
        self.performance.validate()

    def __repr__(self) -> str:
        """Returns a formatted representation of the current configuration state."""
        return (
            "SystemConfig(\n"
            f"  model={self.model},\n"
            f"  camera={self.camera},\n"
            f"  display={self.display},\n"
            f"  drawing={self.drawing},\n"
            f"  performance={self.performance},\n"
            f"  output={self.output},\n"
            f"  debug={self.debug}\n"
            ")"
        )


# Global singleton CONFIG object accessible across modules
CONFIG = SystemConfig()

# Environment-overridable settings used by the webcam application. CONFIG
# remains available above for code that uses the detailed configuration API.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
CAMERA_ID = int(os.getenv("BALL_DETECTOR_CAMERA_ID", "0"))
MODEL_PATH = Path(
    os.getenv("BALL_DETECTOR_MODEL_PATH", str(MODELS_DIR / "yolo11n.pt"))
)
CONFIDENCE_THRESHOLD = float(os.getenv("BALL_DETECTOR_CONFIDENCE", "0.35"))
IOU_THRESHOLD = float(os.getenv("BALL_DETECTOR_IOU", "0.45"))
WINDOW_NAME = "HackTronix 2.0 - Real-Time Ball Detection"
FRAME_WIDTH = int(os.getenv("BALL_DETECTOR_FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.getenv("BALL_DETECTOR_FRAME_HEIGHT", "480"))
INFERENCE_IMAGE_SIZE = int(os.getenv("BALL_DETECTOR_IMAGE_SIZE", "640"))
BALL_CLASS_NAMES = {"sports ball", "ball"}
