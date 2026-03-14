#!/usr/bin/env python3
"""
Unit tests for people_counter.py
==================================
Tests cover the pure-Python logic components (CentroidTracker,
HeatmapAccumulator, non_max_suppression) without requiring a real
camera, video file, or any downloaded model files.
"""

import os
import sys
import tempfile
import unittest

import numpy as np

# Ensure the repo root is on the path so we can import the module directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from collections import deque

from people_counter import (
    CentroidTracker,
    HeatmapAccumulator,
    draw_trails,
    get_trail_color,
    non_max_suppression,
)


# ---------------------------------------------------------------------------
# CentroidTracker tests
# ---------------------------------------------------------------------------
class TestCentroidTracker(unittest.TestCase):
    def test_registers_new_objects(self):
        """New detections should be registered with unique IDs."""
        tracker = CentroidTracker(max_disappeared=5, max_distance=50)
        rects = [(10, 10, 30, 60), (100, 10, 30, 60)]
        objects = tracker.update(rects)
        self.assertEqual(len(objects), 2)
        self.assertEqual(tracker.total_count, 2)

    def test_maintains_id_across_frames(self):
        """Objects that move slightly should keep the same ID."""
        tracker = CentroidTracker(max_disappeared=5, max_distance=50)
        objects1 = tracker.update([(10, 10, 30, 60)])
        first_id = list(objects1.keys())[0]

        # Slightly moved
        objects2 = tracker.update([(12, 12, 30, 60)])
        self.assertIn(first_id, objects2)
        # No new registrations
        self.assertEqual(tracker.total_count, 1)

    def test_deregisters_after_max_disappeared(self):
        """Objects not seen for max_disappeared frames should be dropped."""
        tracker = CentroidTracker(max_disappeared=3, max_distance=50)
        tracker.update([(10, 10, 30, 60)])
        self.assertEqual(len(tracker.objects), 1)

        # No detections for max_disappeared + 1 frames
        for _ in range(4):
            tracker.update([])

        self.assertEqual(len(tracker.objects), 0)
        # total_count should still record the one we saw
        self.assertEqual(tracker.total_count, 1)

    def test_counts_unique_individuals(self):
        """total_count should increment for each new unique object."""
        tracker = CentroidTracker(max_disappeared=2, max_distance=40)
        tracker.update([(0, 0, 20, 40)])      # person A
        tracker.update([])
        tracker.update([])
        tracker.update([])                     # A deregistered
        tracker.update([(200, 200, 20, 40)])  # person B (far away → new ID)
        self.assertEqual(tracker.total_count, 2)

    def test_empty_update_returns_empty_when_no_objects(self):
        tracker = CentroidTracker()
        objects = tracker.update([])
        self.assertEqual(objects, {})
        self.assertEqual(tracker.total_count, 0)

    def test_total_count_increments_only_on_new_objects(self):
        tracker = CentroidTracker(max_disappeared=10, max_distance=50)
        tracker.update([(10, 10, 20, 40)])
        tracker.update([(12, 12, 20, 40)])  # same person
        tracker.update([(14, 14, 20, 40)])  # same person
        self.assertEqual(tracker.total_count, 1)


# ---------------------------------------------------------------------------
# non_max_suppression tests
# ---------------------------------------------------------------------------
class TestNonMaxSuppression(unittest.TestCase):
    def test_returns_empty_for_empty_input(self):
        self.assertEqual(non_max_suppression([]), [])

    def test_single_box_returned_as_is(self):
        boxes = [(10, 10, 50, 100)]
        result = non_max_suppression(boxes)
        self.assertEqual(result, boxes)

    def test_removes_overlapping_boxes(self):
        # Two heavily overlapping boxes — only one should survive
        boxes = [(10, 10, 50, 100), (12, 12, 50, 100)]
        result = non_max_suppression(boxes, overlap_thresh=0.5)
        self.assertEqual(len(result), 1)

    def test_keeps_non_overlapping_boxes(self):
        # Two boxes far apart — both should survive
        boxes = [(0, 0, 30, 60), (200, 200, 30, 60)]
        result = non_max_suppression(boxes, overlap_thresh=0.65)
        self.assertEqual(len(result), 2)


# ---------------------------------------------------------------------------
# HeatmapAccumulator tests
# ---------------------------------------------------------------------------
class TestHeatmapAccumulator(unittest.TestCase):
    def test_update_increments_map(self):
        hm = HeatmapAccumulator((100, 100), sigma=5)
        hm.update([[50, 50]])
        self.assertGreater(hm._map[50, 50], 0.0)

    def test_update_clamps_out_of_bounds(self):
        """Centroids outside frame bounds should not raise."""
        hm = HeatmapAccumulator((100, 100), sigma=5)
        hm.update([[-10, -10], [200, 200]])  # should clamp, not crash

    def test_save_creates_file(self):
        hm = HeatmapAccumulator((200, 320), sigma=10)
        hm.update([[100, 80], [160, 120]])
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            hm.save(path)
            self.assertTrue(os.path.isfile(path))
            self.assertGreater(os.path.getsize(path), 0)
        finally:
            os.unlink(path)

    def test_save_with_background(self):
        hm = HeatmapAccumulator((200, 320), sigma=10)
        hm.update([[100, 80]])
        bg = np.zeros((200, 320, 3), dtype=np.uint8)
        bg[:, :] = (30, 30, 30)  # dark grey background
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            hm.save(path, background_frame=bg)
            self.assertTrue(os.path.isfile(path))
        finally:
            os.unlink(path)

    def test_save_empty_heatmap(self):
        """Saving with no recorded positions should not raise."""
        hm = HeatmapAccumulator((100, 100), sigma=5)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            hm.save(path)
            self.assertTrue(os.path.isfile(path))
        finally:
            os.unlink(path)




# ---------------------------------------------------------------------------
# Trail helper tests
# ---------------------------------------------------------------------------
class TestGetTrailColor(unittest.TestCase):
    def test_returns_bgr_tuple(self):
        color = get_trail_color(0)
        self.assertIsInstance(color, tuple)
        self.assertEqual(len(color), 3)

    def test_values_in_valid_range(self):
        for obj_id in range(20):
            b, g, r = get_trail_color(obj_id)
            self.assertGreaterEqual(b, 0)
            self.assertLessEqual(b, 255)
            self.assertGreaterEqual(g, 0)
            self.assertLessEqual(g, 255)
            self.assertGreaterEqual(r, 0)
            self.assertLessEqual(r, 255)

    def test_different_ids_produce_different_colors(self):
        colors = {get_trail_color(i) for i in range(10)}
        self.assertGreater(len(colors), 1)


class TestDrawTrails(unittest.TestCase):
    def _blank_frame(self):
        return np.zeros((200, 320, 3), dtype=np.uint8)

    def test_no_crash_empty_history(self):
        frame = self._blank_frame()
        draw_trails(frame, {})  # should not raise

    def test_no_crash_single_point_trail(self):
        """A single-point trail produces radius=0 so nothing is drawn."""
        frame = self._blank_frame()
        trail = deque(maxlen=25)
        trail.append((100, 100))
        draw_trails(frame, {0: trail})
        # radius = int(12 * (0/1)) = 0 → skipped; frame stays all zeros
        self.assertTrue(np.all(frame == 0))

    def test_draws_on_frame_with_two_points(self):
        """A two-point trail draws a circle at the head and modifies the frame."""
        frame = self._blank_frame()
        trail = deque(maxlen=25)
        trail.append((10, 10))
        trail.append((50, 50))
        draw_trails(frame, {1: trail})
        # Second point: radius = int(12 * (1/2)) = 6 → circle drawn
        self.assertFalse(np.all(frame == 0))

    def test_draws_multiple_trails(self):
        """Multiple person trails should all be drawn without error."""
        frame = self._blank_frame()
        histories = {}
        for pid in range(3):
            trail = deque(maxlen=25)
            for i in range(10):
                trail.append((pid * 30 + i, pid * 30 + i))
            histories[pid] = trail
        draw_trails(frame, histories)
        self.assertFalse(np.all(frame == 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
