"""OpenCV overlays for live ball-detection feedback."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from src.detector import Detection


BOX_COLOR = (0, 220, 0)
TEXT_COLOR = (255, 255, 255)
PANEL_COLOR = (28, 28, 28)


def draw_detections(frame: np.ndarray, detections: Sequence[Detection]) -> None:
    """Draw labeled bounding boxes directly onto a BGR frame."""
    frame_height, frame_width = frame.shape[:2]
    for detection in detections:
        x1, y1, x2, y2 = detection.xyxy
        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(0, min(x2, frame_width - 1))
        y2 = max(0, min(y2, frame_height - 1))
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

        caption = f"{detection.label}: {detection.confidence:.0%}"
        (caption_width, caption_height), baseline = cv2.getTextSize(
            caption, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        caption_y = max(caption_height + baseline + 4, y1)
        cv2.rectangle(
            frame,
            (x1, caption_y - caption_height - baseline - 4),
            (x1 + caption_width + 8, caption_y + 2),
            BOX_COLOR,
            thickness=-1,
        )
        cv2.putText(
            frame,
            caption,
            (x1 + 4, caption_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            PANEL_COLOR,
            2,
            cv2.LINE_AA,
        )


def draw_hud(
    frame: np.ndarray,
    detection_count: int,
    fps: float,
    inference_ms: float,
) -> None:
    """Draw count, stable FPS, and inference latency in the top-left corner."""
    lines = (
        f"Balls: {detection_count}",
        f"FPS: {fps:.1f}",
        f"Inference: {inference_ms:.1f} ms",
        "Q: quit",
    )
    panel_height = 30 * len(lines) + 12
    cv2.rectangle(frame, (8, 8), (245, panel_height), PANEL_COLOR, thickness=-1)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (16, 34 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )
