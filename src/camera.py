"""Webcam capture abstraction with explicit error handling."""

from __future__ import annotations

import logging

import cv2
import numpy as np


class CameraError(RuntimeError):
    """Raised when a webcam cannot be opened or provide a frame."""


class Camera:
    """Manage a single OpenCV webcam capture device."""

    def __init__(self, camera_id: int, width: int, height: int) -> None:
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self._capture: cv2.VideoCapture | None = None
        self._logger = logging.getLogger(__name__)

    def open(self) -> None:
        """Open and configure the camera, or raise a useful error."""
        if self.is_opened:
            return

        # DirectShow reduces startup latency on many Windows webcams. OpenCV's
        # default backend is retained elsewhere for cross-platform operation.
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
        self._capture = cv2.VideoCapture(self.camera_id, backend)
        if not self._capture.isOpened():
            self.release()
            raise CameraError(
                f"Could not open camera {self.camera_id}. Check its connection "
                "or select another camera index."
            )

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._logger.info(
            "Camera %s opened (requested %sx%s).",
            self.camera_id,
            self.width,
            self.height,
        )

    @property
    def is_opened(self) -> bool:
        """Return whether the capture device is currently available."""
        return self._capture is not None and self._capture.isOpened()

    def read(self) -> np.ndarray:
        """Return the newest BGR frame or raise when the camera disconnects."""
        if not self.is_opened or self._capture is None:
            raise CameraError("Camera is not open.")

        success, frame = self._capture.read()
        if not success or frame is None:
            raise CameraError(
                "Could not read a frame. The webcam may have been disconnected."
            )
        return frame

    def release(self) -> None:
        """Release the camera safely; this operation is idempotent."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            self._logger.info("Camera released.")

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
