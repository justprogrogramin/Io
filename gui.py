#!/usr/bin/env python3
"""
People Counter – Tkinter GUI
==============================
A graphical front-end for people_counter.py.

Usage
-----
    python gui.py

The GUI lets you:
  • Browse for a local video file or use the webcam.
  • Start / Pause / Stop the people-counter pipeline.
  • Watch the live video feed with bounding-box overlays.
  • Monitor real-time statistics (total count, in-frame count, FPS).
  • Toggle age/gender estimation and adjust the confidence threshold.
  • View the traffic heatmap after the session ends.
"""

import os
import sys
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Optional: PIL/Pillow for high-quality image display in the canvas.
# If not available we fall back to a pure-tkinter photo-image approach.
# ---------------------------------------------------------------------------
try:
    from PIL import Image, ImageTk
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# Ensure the repo root is importable when gui.py is executed from any CWD.
sys.path.insert(0, os.path.dirname(__file__))

from people_counter import (
    CentroidTracker,
    HeatmapAccumulator,
    build_hog_detector,
    build_yolo_detector,
    detect_people_hog,
    detect_people_yolo,
    load_age_gender_nets,
    load_face_detector,
    detect_faces,
    predict_age_gender,
    draw_info_panel,
    DEFAULT_MODELS_DIR,
    DEFAULT_HEATMAP_PATH,
)

# ---------------------------------------------------------------------------
# Processing thread
# ---------------------------------------------------------------------------

class ProcessingThread(threading.Thread):
    """Runs the people-counter pipeline in a background thread.

    Communication with the GUI happens through two queues:
      * ``frame_queue``  – latest annotated frame (only the newest is kept).
      * ``stats_queue``  – dict with keys: total, in_frame, fps, running.
    """

    def __init__(
        self,
        source,
        models_dir,
        use_age_gender,
        confidence,
        skip_frames,
        heatmap_path,
        frame_queue,
        stats_queue,
    ):
        super().__init__(daemon=True)
        self.source = source          # int (webcam index) or str (file path)
        self.models_dir = models_dir
        self.use_age_gender = use_age_gender
        self.confidence = confidence
        self.skip_frames = skip_frames
        self.heatmap_path = heatmap_path
        self.frame_queue = frame_queue
        self.stats_queue = stats_queue

        self._pause_event = threading.Event()
        self._pause_event.set()          # not paused initially
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    def pause(self):
        self._pause_event.clear()

    def resume(self):
        self._pause_event.set()

    def stop(self):
        self._stop_event.set()
        self._pause_event.set()  # unblock if paused

    # ------------------------------------------------------------------
    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.stats_queue.put({"error": f"Cannot open source: {self.source}"})
            return

        ret, first_frame = cap.read()
        if not ret:
            self.stats_queue.put({"error": "Failed to read from video source."})
            cap.release()
            return

        frame_h, frame_w = first_frame.shape[:2]

        # -- Detector --------------------------------------------------------
        yolo_cfg = os.path.join(self.models_dir, "yolov3.cfg")
        yolo_weights = os.path.join(self.models_dir, "yolov3.weights")
        yolo_names = os.path.join(self.models_dir, "coco.names")
        yolo_info = build_yolo_detector(yolo_cfg, yolo_weights, yolo_names)
        if yolo_info:
            detector_mode = "yolo"
        else:
            hog = build_hog_detector()
            detector_mode = "hog"

        # -- Age / gender ----------------------------------------------------
        age_net = gender_net = face_cascade = None
        if self.use_age_gender:
            age_proto = os.path.join(self.models_dir, "age_deploy.prototxt")
            age_model = os.path.join(self.models_dir, "age_net.caffemodel")
            gender_proto = os.path.join(self.models_dir, "gender_deploy.prototxt")
            gender_model = os.path.join(self.models_dir, "gender_net.caffemodel")
            age_net, gender_net = load_age_gender_nets(
                age_proto, age_model, gender_proto, gender_model
            )
            if age_net is not None:
                face_cascade = load_face_detector()

        # -- Tracker & heatmap -----------------------------------------------
        tracker = CentroidTracker(max_disappeared=50, max_distance=80)
        heatmap = HeatmapAccumulator(frame_shape=(frame_h, frame_w), sigma=40)

        last_rects: list = []
        frame_index = 0
        start_time = time.time()
        last_fps_time = start_time
        fps = 0.0
        background_frame = first_frame.copy()
        ag_cache: dict = {}

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while not self._stop_event.is_set():
            self._pause_event.wait()
            if self._stop_event.is_set():
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1

            if frame_index % self.skip_frames == 0:
                if detector_mode == "yolo":
                    last_rects = detect_people_yolo(
                        yolo_info, frame, confidence_threshold=self.confidence
                    )
                else:
                    last_rects = detect_people_hog(hog, frame)

            objects = tracker.update(last_rects)

            if ag_cache:
                stale = set(ag_cache) - set(objects)
                for sid in stale:
                    del ag_cache[sid]

            heatmap.update(objects.values())

            if age_net is not None and frame_index % self.skip_frames == 0:
                for obj_id, centroid in objects.items():
                    if obj_id in ag_cache:
                        continue
                    cx, cy = centroid
                    best_box = None
                    best_dist = float("inf")
                    for x, y, w, h in last_rects:
                        d = abs(x + w // 2 - cx) + abs(y + h // 2 - cy)
                        if d < best_dist:
                            best_dist = d
                            best_box = (x, y, w, h)
                    if best_box is not None:
                        crops = detect_faces(face_cascade, frame, best_box)
                        if crops:
                            age, gender = predict_age_gender(
                                crops[0], age_net, gender_net
                            )
                            if age and gender:
                                ag_cache[obj_id] = (age, gender)

            # Draw overlays
            for obj_id, centroid in objects.items():
                cx, cy = centroid
                label = f"ID {obj_id}"
                if obj_id in ag_cache:
                    age_lbl, gender_lbl = ag_cache[obj_id]
                    label += f" | {gender_lbl} {age_lbl}"
                for x, y, w, h in last_rects:
                    if abs(x + w // 2 - cx) < 50 and abs(y + h // 2 - cy) < 50:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
                        cv2.putText(
                            frame, label, (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                        )
                        break
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            now = time.time()
            if now - last_fps_time >= 1.0:
                fps = frame_index / (now - start_time)
                last_fps_time = now

            draw_info_panel(frame, tracker.total_count, len(objects), fps)

            # Push frame (drop old frames so GUI stays responsive)
            if not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)

            self.stats_queue.put(
                {
                    "total": tracker.total_count,
                    "in_frame": len(objects),
                    "fps": fps,
                    "running": True,
                }
            )

            background_frame = frame.copy()

        cap.release()

        # Save heatmap
        heatmap.save(self.heatmap_path, background_frame=background_frame)
        elapsed = time.time() - start_time
        self.stats_queue.put(
            {
                "total": tracker.total_count,
                "in_frame": 0,
                "fps": fps,
                "running": False,
                "elapsed": elapsed,
                "frames": frame_index,
                "heatmap": self.heatmap_path,
            }
        )


# ---------------------------------------------------------------------------
# GUI Application
# ---------------------------------------------------------------------------

class PeopleCounterApp(tk.Tk):
    """Main application window."""

    # Canvas display size (the video frame is scaled to fit)
    CANVAS_W = 640
    CANVAS_H = 480

    def __init__(self):
        super().__init__()
        self.title("People Counter")
        self.resizable(False, False)

        # State
        self._thread: ProcessingThread | None = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._stats_queue: queue.Queue = queue.Queue()
        self._running = False
        self._paused = False
        self._last_heatmap_path: str | None = None

        self._build_ui()
        self._poll_queues()

    # ------------------------------------------------------------------
    def _build_ui(self):
        """Create all widgets."""
        PAD = 8

        # ---- Top bar: source selection -----------------------------------
        src_frame = ttk.LabelFrame(self, text="Video Source", padding=PAD)
        src_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=PAD, pady=(PAD, 0))

        self._use_webcam = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            src_frame, text="Use Webcam", variable=self._use_webcam,
            command=self._on_webcam_toggle,
        ).grid(row=0, column=0, sticky="w")

        self._video_path = tk.StringVar()
        self._path_entry = ttk.Entry(src_frame, textvariable=self._video_path, width=52)
        self._path_entry.grid(row=0, column=1, padx=(PAD, 4))

        self._browse_btn = ttk.Button(src_frame, text="Browse…", command=self._browse)
        self._browse_btn.grid(row=0, column=2)

        # ---- Settings bar ------------------------------------------------
        cfg_frame = ttk.LabelFrame(self, text="Settings", padding=PAD)
        cfg_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=PAD, pady=(PAD, 0))

        self._age_gender = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            cfg_frame, text="Age / Gender Detection", variable=self._age_gender,
        ).grid(row=0, column=0, sticky="w")

        ttk.Label(cfg_frame, text="  Confidence:").grid(row=0, column=1, sticky="e")
        self._confidence = tk.DoubleVar(value=0.40)
        conf_spin = ttk.Spinbox(
            cfg_frame, from_=0.10, to=0.90, increment=0.05,
            textvariable=self._confidence, width=6, format="%.2f",
        )
        conf_spin.grid(row=0, column=2, padx=(2, PAD))

        ttk.Label(cfg_frame, text="  Skip Frames:").grid(row=0, column=3, sticky="e")
        self._skip_frames = tk.IntVar(value=20)
        ttk.Spinbox(
            cfg_frame, from_=1, to=60, increment=1,
            textvariable=self._skip_frames, width=5,
        ).grid(row=0, column=4, padx=(2, PAD))

        ttk.Label(cfg_frame, text="  Heatmap output:").grid(row=0, column=5, sticky="e")
        self._heatmap_path = tk.StringVar(value=DEFAULT_HEATMAP_PATH)
        ttk.Entry(cfg_frame, textvariable=self._heatmap_path, width=24).grid(
            row=0, column=6, padx=(2, 0)
        )

        # ---- Control buttons --------------------------------------------
        ctrl_frame = ttk.Frame(self)
        ctrl_frame.grid(row=2, column=0, columnspan=2, pady=(PAD, 0))

        self._start_btn = ttk.Button(
            ctrl_frame, text="▶  Start", command=self._start, width=14
        )
        self._start_btn.grid(row=0, column=0, padx=4)

        self._pause_btn = ttk.Button(
            ctrl_frame, text="⏸  Pause", command=self._pause, width=14,
            state=tk.DISABLED,
        )
        self._pause_btn.grid(row=0, column=1, padx=4)

        self._stop_btn = ttk.Button(
            ctrl_frame, text="⏹  Stop", command=self._stop, width=14,
            state=tk.DISABLED,
        )
        self._stop_btn.grid(row=0, column=2, padx=4)

        self._heatmap_btn = ttk.Button(
            ctrl_frame, text="View Heatmap", command=self._view_heatmap,
            width=18, state=tk.DISABLED,
        )
        self._heatmap_btn.grid(row=0, column=3, padx=4)

        # ---- Video canvas -----------------------------------------------
        self._canvas = tk.Canvas(
            self, width=self.CANVAS_W, height=self.CANVAS_H, bg="#1a1a2e"
        )
        self._canvas.grid(row=3, column=0, padx=PAD, pady=PAD)

        # Placeholder text on canvas
        self._canvas.create_text(
            self.CANVAS_W // 2, self.CANVAS_H // 2,
            text="No feed – press ▶ Start",
            fill="#888888", font=("Helvetica", 14),
            tags="placeholder",
        )

        # ---- Stats panel ------------------------------------------------
        stats_frame = ttk.LabelFrame(self, text="Live Statistics", padding=PAD)
        stats_frame.grid(row=3, column=1, sticky="n", padx=(0, PAD), pady=PAD)

        def _stat_row(label, row):
            ttk.Label(stats_frame, text=label, width=16, anchor="w").grid(
                row=row, column=0, sticky="w", pady=2
            )
            var = tk.StringVar(value="—")
            ttk.Label(
                stats_frame, textvariable=var,
                font=("Helvetica", 16, "bold"), foreground="#0078d7", width=8, anchor="e",
            ).grid(row=row, column=1, sticky="e", pady=2)
            return var

        self._total_var = _stat_row("Total Count", 0)
        self._inframe_var = _stat_row("In Frame", 1)
        self._fps_var = _stat_row("FPS", 2)
        self._status_var = tk.StringVar(value="Idle")
        ttk.Label(
            stats_frame, textvariable=self._status_var,
            font=("Helvetica", 10), foreground="#555555",
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(PAD, 0))

        # ---- Status bar -------------------------------------------------
        self._statusbar = ttk.Label(
            self, text="Ready", relief=tk.SUNKEN, anchor="w", padding=(4, 2)
        )
        self._statusbar.grid(
            row=4, column=0, columnspan=2, sticky="ew", padx=PAD, pady=(0, PAD)
        )

    # ------------------------------------------------------------------
    # Source helpers
    # ------------------------------------------------------------------
    def _on_webcam_toggle(self):
        if self._use_webcam.get():
            self._path_entry.configure(state=tk.DISABLED)
            self._browse_btn.configure(state=tk.DISABLED)
        else:
            self._path_entry.configure(state=tk.NORMAL)
            self._browse_btn.configure(state=tk.NORMAL)

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._video_path.set(path)

    # ------------------------------------------------------------------
    # Playback controls
    # ------------------------------------------------------------------
    def _get_source(self):
        """Return the video source (int for webcam, str for file)."""
        if self._use_webcam.get():
            return 0
        path = self._video_path.get().strip()
        if not path:
            messagebox.showerror("No Source", "Please select a video file or enable webcam.")
            return None
        if not os.path.isfile(path):
            messagebox.showerror("File Not Found", f"Cannot find:\n{path}")
            return None
        return path

    def _start(self):
        if self._running and not self._paused:
            return
        if self._paused and self._thread is not None:
            # Resume
            self._thread.resume()
            self._paused = False
            self._pause_btn.configure(text="⏸  Pause")
            self._status_var.set("Running")
            self._statusbar.configure(text="Resumed")
            return

        source = self._get_source()
        if source is None:
            return

        # Clear old queues
        self._frame_queue = queue.Queue(maxsize=2)
        self._stats_queue = queue.Queue()

        self._thread = ProcessingThread(
            source=source,
            models_dir=DEFAULT_MODELS_DIR,
            use_age_gender=self._age_gender.get(),
            confidence=self._confidence.get(),
            skip_frames=self._skip_frames.get(),
            heatmap_path=self._heatmap_path.get(),
            frame_queue=self._frame_queue,
            stats_queue=self._stats_queue,
        )
        self._thread.start()

        self._running = True
        self._paused = False

        self._start_btn.configure(state=tk.DISABLED)
        self._pause_btn.configure(state=tk.NORMAL, text="⏸  Pause")
        self._stop_btn.configure(state=tk.NORMAL)
        self._heatmap_btn.configure(state=tk.DISABLED)
        self._status_var.set("Running")
        self._statusbar.configure(text="Processing started…")

        # Delete canvas placeholder
        self._canvas.delete("placeholder")

    def _pause(self):
        if not self._running or self._thread is None:
            return
        if self._paused:
            self._thread.resume()
            self._paused = False
            self._pause_btn.configure(text="⏸  Pause")
            self._status_var.set("Running")
            self._statusbar.configure(text="Resumed")
        else:
            self._thread.pause()
            self._paused = True
            self._pause_btn.configure(text="▶  Resume")
            self._status_var.set("Paused")
            self._statusbar.configure(text="Paused")

    def _stop(self):
        if self._thread is not None:
            self._thread.stop()
        self._running = False
        self._paused = False
        self._start_btn.configure(state=tk.NORMAL)
        self._pause_btn.configure(state=tk.DISABLED, text="⏸  Pause")
        self._stop_btn.configure(state=tk.DISABLED)
        self._status_var.set("Stopped")
        self._statusbar.configure(text="Stopped by user")

    def _view_heatmap(self):
        path = self._last_heatmap_path or self._heatmap_path.get()
        if not path or not os.path.isfile(path):
            messagebox.showinfo("Heatmap", "Heatmap file not found yet.\nRun the counter first.")
            return
        # Open heatmap in a separate top-level window
        hm_win = tk.Toplevel(self)
        hm_win.title("People Traffic Heatmap")
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            messagebox.showerror("Error", f"Cannot load heatmap:\n{path}")
            return
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        max_dim = 800
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)))
            h, w = img_rgb.shape[:2]

        photo = _to_photo_image(img_rgb)
        lbl = tk.Label(hm_win, image=photo)
        lbl.image = photo  # keep reference
        lbl.pack()

    # ------------------------------------------------------------------
    # Queue polling (called every 30 ms from the main thread)
    # ------------------------------------------------------------------
    def _poll_queues(self):
        # -- Drain stats queue -------------------------------------------
        latest_stats = None
        while True:
            try:
                latest_stats = self._stats_queue.get_nowait()
            except queue.Empty:
                break

        if latest_stats:
            if "error" in latest_stats:
                messagebox.showerror("Error", latest_stats["error"])
                self._stop()
            else:
                self._total_var.set(str(latest_stats.get("total", "—")))
                self._inframe_var.set(str(latest_stats.get("in_frame", "—")))
                fps_val = latest_stats.get("fps", None)
                self._fps_var.set(f"{fps_val:.1f}" if fps_val is not None else "—")

                if not latest_stats.get("running", True):
                    # Pipeline finished naturally
                    elapsed = latest_stats.get("elapsed", 0)
                    frames = latest_stats.get("frames", 0)
                    heatmap = latest_stats.get("heatmap", "")
                    if heatmap:
                        self._last_heatmap_path = heatmap
                    self._running = False
                    self._start_btn.configure(state=tk.NORMAL)
                    self._pause_btn.configure(state=tk.DISABLED, text="⏸  Pause")
                    self._stop_btn.configure(state=tk.DISABLED)
                    self._heatmap_btn.configure(state=tk.NORMAL)
                    self._status_var.set("Finished")
                    self._statusbar.configure(
                        text=f"Finished — {latest_stats.get('total', 0)} people, "
                             f"{frames} frames, {elapsed:.1f}s"
                    )

        # -- Draw latest frame -------------------------------------------
        try:
            frame = self._frame_queue.get_nowait()
        except queue.Empty:
            frame = None

        if frame is not None:
            self._render_frame(frame)

        # Schedule next poll
        self.after(30, self._poll_queues)

    # ------------------------------------------------------------------
    def _render_frame(self, frame_bgr):
        """Scale a BGR frame and paint it onto the canvas."""
        h, w = frame_bgr.shape[:2]
        scale = min(self.CANVAS_W / w, self.CANVAS_H / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame_bgr, (new_w, new_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        photo = _to_photo_image(rgb)
        # Anchor in the center of the canvas
        x0 = (self.CANVAS_W - new_w) // 2
        y0 = (self.CANVAS_H - new_h) // 2
        self._canvas.delete("frame")
        self._canvas.create_image(x0, y0, anchor="nw", image=photo, tags="frame")
        self._canvas.image = photo  # keep reference to prevent GC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_photo_image(rgb_array: np.ndarray):
    """Convert an RGB numpy array to a Tk PhotoImage."""
    if _PIL_AVAILABLE:
        pil_img = Image.fromarray(rgb_array)
        return ImageTk.PhotoImage(pil_img)
    # Fallback: encode to PPM in memory (supported by tkinter natively)
    h, w = rgb_array.shape[:2]
    header = f"P6\n{w} {h}\n255\n".encode()
    raw_bytes = rgb_array.tobytes()
    ppm_data = header + raw_bytes
    photo = tk.PhotoImage(data=ppm_data)
    return photo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = PeopleCounterApp()
    app.mainloop()


if __name__ == "__main__":
    main()
