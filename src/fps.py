"""High-performance, thread-safe FPS calculation module for real-time video processing.

This module is designed for real-time applications such as YOLO11n object detection
pipelines, where negligible CPU overhead and thread safety are critical.
"""

from dataclasses import dataclass
import threading
import time
from typing import Dict, Any, List


@dataclass(frozen=True)
class FPSStats:
    """Read-only snapshot of current FPS metrics.

    Attributes:
        current_fps: The most recently calculated FPS (updated once per second).
        average_fps: The moving average FPS over the last window size (e.g., 30 measurements).
        frame_count: Total frames processed.
        total_runtime: Total elapsed time in seconds.
    """
    current_fps: float
    average_fps: float
    frame_count: int
    total_runtime: float


class FPS:
    """Industrial-grade FPS calculator for high-speed video processing loops.

    Maintains a moving average of FPS over a sliding window and updates metrics
    periodically (default once per second) to minimize CPU overhead. Avoids
    unnecessary memory allocations on the hot path (update method). Thread-safe
    for multi-threaded camera acquisition and inference pipelines.
    """

    def __init__(self, window_size: int = 30, update_interval: float = 1.0) -> None:
        """Initializes the FPS counter.

        Args:
            window_size: Number of historical FPS measurements for moving average.
            update_interval: Interval in seconds to recalculate current FPS.
        """
        self._window_size = window_size
        self._update_interval = update_interval
        
        self._lock = threading.Lock()
        
        # Pre-allocated list for circular buffer to avoid allocations in update()
        self._history: List[float] = [0.0] * self._window_size
        self._history_idx: int = 0
        self._history_len: int = 0
        
        self._frame_count: int = 0
        self._interval_frames: int = 0
        self._current_fps: float = 0.0
        self._average_fps: float = 0.0
        
        self._start_time: float = time.perf_counter()
        self._last_update_time: float = self._start_time

    def update(self) -> None:
        """Increments the frame counter and updates FPS metrics if the interval elapsed.

        This method should be called exactly once per processed frame in the
        inference or capture loop. It is designed to run in sub-microsecond
        time with zero heap allocations on the hot path.
        """
        with self._lock:
            self._frame_count += 1
            self._interval_frames += 1
            
            now = time.perf_counter()
            elapsed = now - self._last_update_time
            
            if elapsed >= self._update_interval:
                # Calculate FPS for this interval
                self._current_fps = self._interval_frames / elapsed if elapsed > 0.0 else 0.0
                
                # Write to the circular buffer
                self._history[self._history_idx] = self._current_fps
                self._history_idx = (self._history_idx + 1) % self._window_size
                if self._history_len < self._window_size:
                    self._history_len += 1
                
                # Calculate the average without slicing (no memory allocation)
                self._average_fps = sum(self._history) / self._history_len
                
                # Reset interval counters
                self._interval_frames = 0
                self._last_update_time = now

    def reset(self) -> None:
        """Resets all metrics, timers, and history to their initial states.

        Useful when restarting a camera stream or re-initializing the detector.
        """
        with self._lock:
            self._history = [0.0] * self._window_size
            self._history_idx = 0
            self._history_len = 0
            self._frame_count = 0
            self._interval_frames = 0
            self._current_fps = 0.0
            self._average_fps = 0.0
            self._start_time = time.perf_counter()
            self._last_update_time = self._start_time

    @property
    def current_fps(self) -> float:
        """Gets the most recently calculated FPS value.

        If the first interval has not yet elapsed, falls back to estimating the
        running FPS of the current active interval.
        """
        with self._lock:
            if self._history_len > 0:
                return self._current_fps
            
            # Fallback estimation for the very first interval
            elapsed = time.perf_counter() - self._last_update_time
            if elapsed > 0.0:
                return self._interval_frames / elapsed
            return 0.0

    @property
    def average_fps(self) -> float:
        """Gets the moving average FPS over the last window size.

        If the first interval has not yet elapsed, falls back to estimating the
        running FPS since start/reset.
        """
        with self._lock:
            if self._history_len > 0:
                return self._average_fps
            
            # Fallback estimation for the very first interval
            elapsed = time.perf_counter() - self._start_time
            if elapsed > 0.0:
                return self._frame_count / elapsed
            return 0.0

    @property
    def frame_count(self) -> int:
        """Gets the total number of frames processed since initialization or reset."""
        with self._lock:
            return self._frame_count

    @property
    def total_runtime(self) -> float:
        """Gets the total elapsed runtime in seconds since start or reset."""
        with self._lock:
            return time.perf_counter() - self._start_time

    def get_fps(self) -> float:
        """Thread-safe accessor for the current FPS.

        Returns:
            The current frame rate in frames per second.
        """
        return self.current_fps

    def get_average_fps(self) -> float:
        """Thread-safe accessor for the moving average FPS.

        Returns:
            The average frame rate in frames per second.
        """
        return self.average_fps

    def get_stats(self) -> FPSStats:
        """Returns a snapshot of the current FPS metrics in a dataclass.

        Avoids deadlock by calculating values without nested lock acquisition.

        Returns:
            An FPSStats snapshot instance.
        """
        with self._lock:
            now = time.perf_counter()
            
            # Calculate current FPS fallback if no interval has completed
            if self._history_len > 0:
                current = self._current_fps
            else:
                elapsed = now - self._last_update_time
                current = self._interval_frames / elapsed if elapsed > 0.0 else 0.0
                
            # Calculate average FPS fallback if no interval has completed
            if self._history_len > 0:
                average = self._average_fps
            else:
                elapsed = now - self._start_time
                average = self._frame_count / elapsed if elapsed > 0.0 else 0.0
                
            return FPSStats(
                current_fps=current,
                average_fps=average,
                frame_count=self._frame_count,
                total_runtime=now - self._start_time
            )

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the current FPS metrics.

        Returns:
            A dictionary containing current_fps, average_fps, frame_count,
            and total_runtime.
        """
        stats = self.get_stats()
        return {
            "current_fps": stats.current_fps,
            "average_fps": stats.average_fps,
            "frame_count": stats.frame_count,
            "total_runtime": stats.total_runtime,
        }
