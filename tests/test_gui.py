#!/usr/bin/env python3
"""
Unit tests for gui.py
======================
Tests cover the pure-logic and helper components of the GUI module
without requiring a real display, camera, or video file.
"""

import os
import sys
import queue
import threading
import unittest

import numpy as np

# Make sure the repo root is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── tkinter & display check ────────────────────────────────────────────────
# Skip every GUI test when no display / tkinter is available so that
# headless CI doesn't fail.
import importlib
_tkinter_spec = importlib.util.find_spec("tkinter")
_has_tkinter = _tkinter_spec is not None

_has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

_SKIP_GUI = not (_has_tkinter and _has_display)
_SKIP_REASON = "tkinter or display not available"

if not _SKIP_GUI:
    import tkinter as tk
    from gui import _to_photo_image, PeopleCounterApp, ProcessingThread


# ---------------------------------------------------------------------------
# _to_photo_image tests
# ---------------------------------------------------------------------------
@unittest.skipIf(_SKIP_GUI, _SKIP_REASON)
class TestToPhotoImage(unittest.TestCase):
    """Tests for the _to_photo_image helper."""

    @classmethod
    def setUpClass(cls):
        cls.root = tk.Tk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def test_returns_photo_image_for_rgb_array(self):
        """A valid RGB uint8 array should produce a PhotoImage."""
        arr = np.zeros((50, 80, 3), dtype=np.uint8)
        photo = _to_photo_image(arr)
        self.assertIsNotNone(photo)

    def test_non_zero_dimensions_preserved(self):
        """The resulting image should have the correct width/height."""
        arr = np.ones((60, 90, 3), dtype=np.uint8) * 128
        photo = _to_photo_image(arr)
        # PhotoImage exposes .width() / .height() methods
        self.assertEqual(photo.width(), 90)
        self.assertEqual(photo.height(), 60)

    def test_full_white_image(self):
        """An all-white array should not raise."""
        arr = np.full((40, 40, 3), 255, dtype=np.uint8)
        photo = _to_photo_image(arr)
        self.assertIsNotNone(photo)

    def test_full_black_image(self):
        """An all-black array should not raise."""
        arr = np.zeros((40, 40, 3), dtype=np.uint8)
        photo = _to_photo_image(arr)
        self.assertIsNotNone(photo)


# ---------------------------------------------------------------------------
# PeopleCounterApp widget tests
# ---------------------------------------------------------------------------
@unittest.skipIf(_SKIP_GUI, _SKIP_REASON)
class TestPeopleCounterAppWidgets(unittest.TestCase):
    """Smoke-tests that the app window and key widgets are created correctly."""

    @classmethod
    def setUpClass(cls):
        cls.app = PeopleCounterApp()
        cls.app.update_idletasks()

    @classmethod
    def tearDownClass(cls):
        cls.app.destroy()

    def test_window_title(self):
        self.assertEqual(self.app.title(), "People Counter")

    def test_canvas_exists(self):
        self.assertIsNotNone(self.app._canvas)
        self.assertEqual(int(self.app._canvas.cget("width")), PeopleCounterApp.CANVAS_W)

    def test_default_stats_labels(self):
        """Stats should show '—' before processing starts."""
        self.assertEqual(self.app._total_var.get(), "—")
        self.assertEqual(self.app._inframe_var.get(), "—")
        self.assertEqual(self.app._fps_var.get(), "—")

    def test_initial_button_states(self):
        """Start is enabled; Pause/Stop/Heatmap are disabled initially."""
        self.assertEqual(str(self.app._start_btn["state"]), "normal")
        self.assertEqual(str(self.app._pause_btn["state"]), "disabled")
        self.assertEqual(str(self.app._stop_btn["state"]), "disabled")
        self.assertEqual(str(self.app._heatmap_btn["state"]), "disabled")

    def test_default_confidence_value(self):
        self.assertAlmostEqual(self.app._confidence.get(), 0.40, places=2)

    def test_default_skip_frames_value(self):
        self.assertEqual(self.app._skip_frames.get(), 20)

    def test_webcam_toggle_disables_entry(self):
        """Enabling webcam mode should disable the path entry."""
        self.app._use_webcam.set(True)
        self.app._on_webcam_toggle()
        self.app.update_idletasks()
        self.assertEqual(str(self.app._path_entry["state"]), "disabled")
        # Restore
        self.app._use_webcam.set(False)
        self.app._on_webcam_toggle()

    def test_webcam_toggle_enables_entry(self):
        """Disabling webcam mode should re-enable the path entry."""
        self.app._use_webcam.set(True)
        self.app._on_webcam_toggle()
        self.app._use_webcam.set(False)
        self.app._on_webcam_toggle()
        self.app.update_idletasks()
        self.assertEqual(str(self.app._path_entry["state"]), "normal")

    def test_status_initial_value(self):
        self.assertEqual(self.app._status_var.get(), "Idle")


# ---------------------------------------------------------------------------
# ProcessingThread unit tests (no real video needed)
# ---------------------------------------------------------------------------
@unittest.skipIf(_SKIP_GUI, _SKIP_REASON)
class TestProcessingThread(unittest.TestCase):
    """Tests for the ProcessingThread helper class."""

    def _make_thread(self):
        fq = queue.Queue(maxsize=2)
        sq = queue.Queue()
        t = ProcessingThread(
            source="nonexistent_video.mp4",
            models_dir="/tmp/no_models",
            use_age_gender=False,
            confidence=0.4,
            skip_frames=20,
            heatmap_path="/tmp/test_heatmap.png",
            frame_queue=fq,
            stats_queue=sq,
        )
        return t, fq, sq

    def test_thread_is_daemon(self):
        t, _, _ = self._make_thread()
        self.assertTrue(t.daemon)

    def test_stop_before_start_does_not_raise(self):
        t, _, _ = self._make_thread()
        t.stop()  # should not raise even if thread not started

    def test_invalid_source_puts_error_in_queue(self):
        """A thread given a nonexistent file should report an error."""
        t, fq, sq = self._make_thread()
        t.start()
        # Wait up to 3 seconds for the error message
        import time
        deadline = time.time() + 3.0
        msg = None
        while time.time() < deadline:
            try:
                msg = sq.get(timeout=0.1)
                break
            except queue.Empty:
                pass
        self.assertIsNotNone(msg, "Expected an error or stats message in stats_queue")
        self.assertIn("error", msg)

    def test_pause_resume_flags(self):
        """Pause/resume should manipulate the internal event correctly."""
        t, _, _ = self._make_thread()
        # Initially not paused
        self.assertTrue(t._pause_event.is_set())
        t.pause()
        self.assertFalse(t._pause_event.is_set())
        t.resume()
        self.assertTrue(t._pause_event.is_set())

    def test_stop_sets_stop_event(self):
        t, _, _ = self._make_thread()
        self.assertFalse(t._stop_event.is_set())
        t.stop()
        self.assertTrue(t._stop_event.is_set())
        # stop() must also unblock pause so thread can exit cleanly
        self.assertTrue(t._pause_event.is_set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
