"""YOLO model loading and ball-only inference."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import shutil
from time import perf_counter
from typing import Mapping

import numpy as np
from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset

from src.config import (
    BALL_CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
    INFERENCE_IMAGE_SIZE,
    IOU_THRESHOLD,
)


class ModelLoadError(RuntimeError):
    """Raised when the selected YOLO weights cannot be prepared or loaded."""


@dataclass(frozen=True, slots=True)
class Detection:
    """One detected ball in pixel coordinates of the input frame."""

    label: str
    confidence: float
    xyxy: tuple[int, int, int, int]


class BallDetector:
    """Load YOLO once and detect only ball-related classes in BGR frames."""

    def __init__(
        self,
        model_path: Path,
        confidence: float = CONFIDENCE_THRESHOLD,
        iou: float = IOU_THRESHOLD,
        image_size: int = INFERENCE_IMAGE_SIZE,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self._logger = logging.getLogger(__name__)
        self._model = self._load_model()
        self._ball_class_ids = self._resolve_ball_class_ids(self._model.names)
        self.last_inference_ms = 0.0

        if self._ball_class_ids:
            self._logger.info("Ball class IDs enabled: %s", self._ball_class_ids)
        else:
            self._logger.warning(
                "No class named %s was found. Inference will run without a class "
                "filter; detections are still filtered by label afterward.",
                ", ".join(sorted(BALL_CLASS_NAMES)),
            )

    def _load_model(self) -> YOLO:
        """Prepare official weights and instantiate the model once."""
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            if not self._has_valid_file_size():
                self._download_model()

            self._logger.info("Loading YOLO model from %s", self.model_path)
            try:
                return YOLO(str(self.model_path))
            except Exception as first_error:
                # Interrupted downloads can leave a non-empty yet unusable .pt
                # file. Replace it once and retry before reporting failure.
                self._logger.warning(
                    "Model checkpoint is invalid (%s). Downloading a fresh copy.",
                    first_error,
                )
                self._download_model(force=True)
                return YOLO(str(self.model_path))
        except Exception as error:
            raise ModelLoadError(
                f"Unable to load YOLO model '{self.model_path}': {error}"
            ) from error

    def _has_valid_file_size(self) -> bool:
        """Reject empty or implausibly small checkpoint files immediately."""
        return self.model_path.is_file() and self.model_path.stat().st_size > 1_000_000

    def _download_model(self, force: bool = False) -> None:
        """Download official weights, replacing an empty or corrupted local copy."""
        if self.model_path.exists() and (force or not self._has_valid_file_size()):
            self._logger.warning("Removing invalid model file: %s", self.model_path)
            self.model_path.unlink()

        self._logger.info("Downloading model weights: %s...", self.model_path.name)
        downloaded_path = Path(attempt_download_asset(self.model_path.name))
        if downloaded_path.resolve() != self.model_path.resolve():
            shutil.move(str(downloaded_path), str(self.model_path))

    @staticmethod
    def _resolve_ball_class_ids(
        names: Mapping[int, str] | list[str],
    ) -> list[int]:
        """Find compatible labels in standard COCO and custom-trained models."""
        items = names.items() if isinstance(names, Mapping) else enumerate(names)
        return [
            class_id
            for class_id, name in items
            if str(name).strip().lower() in BALL_CLASS_NAMES
        ]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run one inference pass and return balls in the source-frame space."""
        started_at = perf_counter()
        predict_options: dict[str, object] = {
            "source": frame,
            "conf": self.confidence,
            "iou": self.iou,
            "imgsz": self.image_size,
            "device": "cpu",
            "verbose": False,
            "max_det": 50,
        }
        if self._ball_class_ids:
            predict_options["classes"] = self._ball_class_ids

        results = self._model.predict(**predict_options)
        self.last_inference_ms = (perf_counter() - started_at) * 1000

        detections: list[Detection] = []
        result = results[0]
        if result.boxes is None:
            return detections

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            label = str(self._model.names[class_id])
            if label.strip().lower() not in BALL_CLASS_NAMES:
                continue

            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    label=label,
                    confidence=float(box.conf[0].item()),
                    xyxy=(x1, y1, x2, y2),
                )
            )
        return detections
