"""Entry point for HackTronix 2.0 Task 1 real-time ball detection."""

from __future__ import annotations

import argparse
import logging
import sys

import cv2

from src.camera import Camera, CameraError
from src.config import (
    CAMERA_ID,
    CONFIDENCE_THRESHOLD,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MODEL_PATH,
    WINDOW_NAME,
)
from src.detector import BallDetector, ModelLoadError
from src.draw import draw_detections, draw_hud
from src.fps import FPSCounter
from src.utils import configure_opencv, ensure_project_directories, setup_logging


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse optional launch-time overrides without complicating normal use."""
    parser = argparse.ArgumentParser(description="Real-time webcam ball detection")
    parser.add_argument(
        "--camera",
        type=int,
        default=CAMERA_ID,
        help=f"Webcam index (default: {CAMERA_ID})",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=f"Minimum detection confidence (default: {CONFIDENCE_THRESHOLD})",
    )
    return parser.parse_args()


def run() -> int:
    """Open the webcam and run detection until the user presses Q."""
    args = parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        LOGGER.error("Confidence must be between 0.0 and 1.0.")
        return 2

    ensure_project_directories()
    configure_opencv()
    LOGGER.info("Starting HackTronix 2.0 ball detector")
    LOGGER.info("Model: %s | confidence: %.2f", MODEL_PATH, args.confidence)
    LOGGER.info("Opening camera index %s", args.camera)

    try:
        detector = BallDetector(MODEL_PATH, confidence=args.confidence)
    except ModelLoadError as error:
        LOGGER.error("Model startup failed: %s", error)
        return 1

    fps_counter = FPSCounter()
    camera = Camera(args.camera, FRAME_WIDTH, FRAME_HEIGHT)
    try:
        camera.open()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        processed_frames = 0

        while True:
            frame = camera.read()
            detections = detector.detect(frame)
            fps = fps_counter.update()
            processed_frames += 1

            draw_detections(frame, detections)
            draw_hud(frame, len(detections), fps, detector.last_inference_ms)
            cv2.imshow(WINDOW_NAME, frame)
            # Reporting periodically avoids console I/O becoming an FPS bottleneck.
            if processed_frames % 60 == 0:
                LOGGER.info(
                    "Inference time: %.1f ms | live FPS: %.1f",
                    detector.last_inference_ms,
                    fps,
                )

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                LOGGER.info("Quit requested by user.")
                break
    except CameraError as error:
        LOGGER.error("Camera error: %s", error)
        return 1
    except KeyboardInterrupt:
        LOGGER.info("Stopped by keyboard interrupt.")
    except cv2.error as error:
        LOGGER.error("OpenCV display error: %s", error)
        return 1
    finally:
        camera.release()
        cv2.destroyAllWindows()
        LOGGER.info("Clean shutdown complete.")

    return 0


if __name__ == "__main__":
    setup_logging()
    sys.exit(run())
