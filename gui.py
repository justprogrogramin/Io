#!/usr/bin/env python3
"""
Tkinter GUI for the People Counter
====================================
Provides a graphical interface to browse for a video file and watch the
people-counter run live.  Each tracked person is shown with:
  - a coloured bounding box and ID label
  - a centroid circle
  - a pastel trajectory line showing recent movement
  - a soft glowing blob trail on a transparent overlay

Usage
-----
    python gui.py
"""

import os
import queue
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Ensure the repo root is importable when running this file directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from people_counter import (
    CentroidTracker,
    build_hog_detector,
    build_yolo_detector,
    detect_people_hog,
    detect_people_yolo,
    draw_info_panel,
    draw_trajectory_lines,
    draw_trails,
    DEFAULT_MODELS_DIR,
)

# Maximum width (pixels) used when displaying video inside the canvas.
_DISPLAY_WIDTH = 800


class PeopleCounterApp(tk.Tk):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.title("People Counter")
        self.resizable(True, True)

        # Thread communication
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        # Callbacks that need to run on the main thread (set from background thread).
        # Using a queue instead of self.after() because tkinter._register /
        # createcommand must only be called from the main thread in Python 3.12+.
        self._ui_callback_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        # --- Top bar: file selector + buttons ---
        top = tk.Frame(self, padx=8, pady=6)
        top.pack(side=tk.TOP, fill=tk.X)

        tk.Label(top, text="Video file:").pack(side=tk.LEFT)
        self._path_var = tk.StringVar()
        tk.Entry(top, textvariable=self._path_var, width=50).pack(
            side=tk.LEFT, padx=4
        )
        tk.Button(top, text="Browse…", command=self._browse).pack(side=tk.LEFT)

        self._start_btn = tk.Button(
            top,
            text="▶  Start",
            width=10,
            bg="#4caf50",
            fg="white",
            command=self._start,
        )
        self._start_btn.pack(side=tk.LEFT, padx=8)

        self._stop_btn = tk.Button(
            top,
            text="■  Stop",
            width=10,
            bg="#f44336",
            fg="white",
            command=self._stop,
            state=tk.DISABLED,
        )
        self._stop_btn.pack(side=tk.LEFT)

        # --- Video canvas ---
        self._canvas = tk.Canvas(self, bg="black", width=640, height=360)
        self._canvas.pack(fill=tk.BOTH, expand=True)

        # --- Status bar ---
        self._status_var = tk.StringVar(
            value="Ready — select a video file and press Start."
        )
        tk.Label(
            self,
            textvariable=self._status_var,
            anchor=tk.W,
            relief=tk.SUNKEN,
            padx=6,
        ).pack(side=tk.BOTTOM, fill=tk.X)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._path_var.set(path)

    def _start(self) -> None:
        src = self._path_var.get().strip()
        if not src:
            messagebox.showwarning(
                "No file selected", "Please browse for a video file first."
            )
            return
        if not os.path.isfile(src):
            messagebox.showerror("File not found", f"Cannot open:\n{src}")
            return

        self._stop_event.clear()
        self._start_btn.config(state=tk.DISABLED)
        self._stop_btn.config(state=tk.NORMAL)
        self._status_var.set(f"Processing: {os.path.basename(src)}")

        self._thread = threading.Thread(
            target=self._process_video, args=(src,), daemon=True
        )
        self._thread.start()
        self._poll_queue()

    def _stop(self) -> None:
        self._stop_event.set()

    def on_close(self) -> None:
        self._stop_event.set()
        self.destroy()

    # ------------------------------------------------------------------
    # Processing (background thread)
    # ------------------------------------------------------------------
    def _process_video(self, src: str) -> None:
        """Run the people-counter loop in a background thread."""
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            self._ui_callback_queue.put(
                lambda: self._status_var.set(f"[ERROR] Cannot open: {src}")
            )
            return

        # Detectors
        models_dir = DEFAULT_MODELS_DIR
        yolo_info = build_yolo_detector(
            os.path.join(models_dir, "yolov3.cfg"),
            os.path.join(models_dir, "yolov3.weights"),
            os.path.join(models_dir, "coco.names"),
        )
        if yolo_info:
            detector_mode = "yolo"
        else:
            hog = build_hog_detector()
            detector_mode = "hog"

        # State
        tracker = CentroidTracker(max_disappeared=50, max_distance=80)
        trail_history: "dict[int, deque]" = {}
        last_rects: list = []
        frame_index = 0
        skip_frames = 20
        start_time = time.time()
        last_fps_time = start_time
        fps = 0.0

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_index += 1

                # Detection
                if frame_index % skip_frames == 0:
                    if detector_mode == "yolo":
                        last_rects = detect_people_yolo(yolo_info, frame)
                    else:
                        last_rects = detect_people_hog(hog, frame)

                objects = tracker.update(last_rects)

                # Update trails
                for obj_id, centroid in objects.items():
                    if obj_id not in trail_history:
                        trail_history[obj_id] = deque(maxlen=25)
                    trail_history[obj_id].append(
                        (int(centroid[0]), int(centroid[1]))
                    )
                for stale_id in set(trail_history) - set(objects):
                    del trail_history[stale_id]

                # --- Draw trajectory lines (connects positions with lines) ---
                draw_trajectory_lines(frame, trail_history)
                # --- Draw blob trails (soft glowing circles) -----------------
                draw_trails(frame, trail_history)

                # --- Bounding boxes, labels, centroid circles ----------------
                for obj_id, centroid in objects.items():
                    cx, cy = int(centroid[0]), int(centroid[1])
                    for x, y, w, h in last_rects:
                        if (
                            abs(x + w // 2 - cx) < 50
                            and abs(y + h // 2 - cy) < 50
                        ):
                            cv2.rectangle(
                                frame, (x, y), (x + w, y + h), (0, 200, 0), 2
                            )
                            cv2.putText(
                                frame,
                                f"ID {obj_id}",
                                (x, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                1,
                            )
                            break
                    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

                # FPS
                now = time.time()
                if now - last_fps_time >= 1.0:
                    fps = frame_index / (now - start_time)
                    last_fps_time = now

                draw_info_panel(frame, tracker.total_count, len(objects), fps)

                # Resize for display
                h, w = frame.shape[:2]
                if w > _DISPLAY_WIDTH:
                    scale = _DISPLAY_WIDTH / w
                    frame = cv2.resize(
                        frame, (_DISPLAY_WIDTH, int(h * scale))
                    )

                # Convert BGR → RGB → PIL
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                try:
                    self._frame_queue.put_nowait(img)
                except queue.Full:
                    pass

        finally:
            cap.release()
            summary = (
                f"Done — {tracker.total_count} unique person(s) counted "
                f"in {frame_index} frames."
            )

            def _finish(s=summary):
                self._status_var.set(s)
                self._start_btn.config(state=tk.NORMAL)
                self._stop_btn.config(state=tk.DISABLED)

            self._ui_callback_queue.put(_finish)

    # ------------------------------------------------------------------
    # Frame polling (main thread)
    # ------------------------------------------------------------------
    def _poll_queue(self) -> None:
        """Pull frames and UI callbacks from their queues on each tick.

        Reschedules itself while the processing thread is alive or either
        queue still has work pending.
        """
        # Drain any pending UI-update callbacks (posted by the background thread).
        while True:
            try:
                cb = self._ui_callback_queue.get_nowait()
                cb()
            except queue.Empty:
                break

        # Display the next available video frame.
        try:
            img = self._frame_queue.get_nowait()
            photo = ImageTk.PhotoImage(image=img)
            self._canvas.config(width=img.width, height=img.height)
            self._canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self._canvas.photo = photo  # prevent garbage collection
        except queue.Empty:
            pass

        # Keep polling while there is still work to do.
        thread_alive = self._thread is not None and self._thread.is_alive()
        if (
            thread_alive
            or not self._frame_queue.empty()
            or not self._ui_callback_queue.empty()
        ):
            self.after(15, self._poll_queue)


# ---------------------------------------------------------------------------
def main() -> None:
    app = PeopleCounterApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()


if __name__ == "__main__":
    main()
