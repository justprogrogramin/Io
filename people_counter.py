#!/usr/bin/env python3
"""
People Counter with Age/Gender Detection and Heatmap Generation
================================================================
Detects people in a video stream or file, counts unique individuals,
estimates approximate age and gender using pre-trained Caffe models,
and writes a heatmap image of high-traffic zones when the program exits.

Usage
-----
    # Webcam (default):
    python people_counter.py

    # Video file:
    python people_counter.py --input path/to/video.mp4

    # Skip age/gender (faster):
    python people_counter.py --no-age-gender

    # Adjust detection confidence:
    python people_counter.py --confidence 0.4

Download required model files first:
    python download_models.py
"""

video = "/home/user/Downloads/my_video.mp4"

import argparse
import os
import sys
import time
from collections import OrderedDict, deque

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend so heatmap always saves
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Age buckets are fixed by the Levi & Hassner (2015) Caffe model's training labels.
# They cannot be changed without retraining the network.
AGE_BUCKETS = [
    "(0-2)", "(4-6)", "(8-12)", "(15-20)",
    "(25-32)", "(38-43)", "(48-53)", "(60-100)",
]
GENDER_LIST = ["Male", "Female"]
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Default paths for model files (can be overridden via CLI)
DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
AGE_PROTO = os.path.join(DEFAULT_MODELS_DIR, "age_deploy.prototxt")
AGE_MODEL = os.path.join(DEFAULT_MODELS_DIR, "age_net.caffemodel")
GENDER_PROTO = os.path.join(DEFAULT_MODELS_DIR, "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(DEFAULT_MODELS_DIR, "gender_net.caffemodel")

# YOLO model paths (optional, improves detection accuracy)
YOLO_CFG = os.path.join(DEFAULT_MODELS_DIR, "yolov3.cfg")
YOLO_WEIGHTS = os.path.join(DEFAULT_MODELS_DIR, "yolov3.weights")
YOLO_NAMES = os.path.join(DEFAULT_MODELS_DIR, "coco.names")

# Heatmap output
DEFAULT_HEATMAP_PATH = "heatmap_output.png"


# ---------------------------------------------------------------------------
# Centroid Tracker
# ---------------------------------------------------------------------------
class CentroidTracker:
    """Tracks objects across frames using centroid proximity.

    Assigns a unique ID to each new object and re-uses IDs for existing
    objects.  When an object has not been seen for *max_disappeared* frames
    its ID is retired.  ``total_count`` accumulates every unique ID ever
    registered — i.e. the cumulative person count.
    """

    def __init__(self, max_disappeared: int = 50, max_distance: int = 80):
        self.next_id = 0
        self.objects: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self.disappeared: "OrderedDict[int, int]" = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.total_count = 0  # cumulative unique person count

    # ------------------------------------------------------------------
    def _register(self, centroid: np.ndarray) -> None:
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1
        self.total_count += 1

    def _deregister(self, obj_id: int) -> None:
        del self.objects[obj_id]
        del self.disappeared[obj_id]

    # ------------------------------------------------------------------
    def update(self, rects: list) -> "OrderedDict[int, np.ndarray]":
        """Update tracker with a list of bounding boxes (x, y, w, h).

        Returns the current mapping of object-ID → centroid.
        """
        if not rects:
            for obj_id in list(self.disappeared):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)
            return self.objects

        input_centroids = np.array(
            [[int(x + w / 2), int(y + h / 2)] for x, y, w, h in rects],
            dtype="int",
        )

        if not self.objects:
            for c in input_centroids:
                self._register(c)
            return self.objects

        obj_ids = list(self.objects)
        obj_centroids = np.array(list(self.objects.values()))

        # Build distance matrix (objects × detections)
        diff = obj_centroids[:, None, :] - input_centroids[None, :, :]  # (O, D, 2)
        dist = np.sqrt((diff ** 2).sum(axis=2))  # (O, D)

        # Greedy matching: sort by smallest distance
        rows = dist.min(axis=1).argsort()
        cols = dist.argmin(axis=1)[rows]

        used_rows: set = set()
        used_cols: set = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if dist[row, col] > self.max_distance:
                continue
            obj_id = obj_ids[row]
            self.objects[obj_id] = input_centroids[col]
            self.disappeared[obj_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(len(obj_ids))) - used_rows
        unused_cols = set(range(len(input_centroids))) - used_cols

        for row in unused_rows:
            obj_id = obj_ids[row]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self._deregister(obj_id)

        for col in unused_cols:
            self._register(input_centroids[col])

        return self.objects


# ---------------------------------------------------------------------------
# Detector helpers
# ---------------------------------------------------------------------------
def build_hog_detector():
    """Return the built-in HOG person detector (no model files required)."""
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog


def detect_people_hog(hog, frame, scale: float = 1.05, win_stride: int = 8):
    """Run HOG detection; returns list of (x, y, w, h) bounding boxes."""
    # Resize for speed
    h, w = frame.shape[:2]
    target_w = 640
    if w > target_w:
        scale_factor = target_w / w
        resized = cv2.resize(frame, (target_w, int(h * scale_factor)))
    else:
        resized = frame
        scale_factor = 1.0

    boxes, _ = hog.detectMultiScale(
        resized,
        winStride=(win_stride, win_stride),
        padding=(8, 8),
        scale=scale,
    )
    if len(boxes) == 0:
        return []

    # Scale boxes back to original frame size
    boxes = [
        (int(x / scale_factor), int(y / scale_factor),
         int(w2 / scale_factor), int(h2 / scale_factor))
        for x, y, w2, h2 in boxes
    ]
    # Non-maximum suppression
    boxes = non_max_suppression(boxes, overlap_thresh=0.65)
    return boxes


def build_yolo_detector(cfg: str, weights: str, names: str):
    """Load YOLOv3 network from disk.  Returns (net, output_layers, class_ids).

    Returns None if model files are missing.
    """
    if not (os.path.isfile(cfg) and os.path.isfile(weights) and os.path.isfile(names)):
        return None

    net = cv2.dnn.readNetFromDarknet(cfg, weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    layer_names = net.getLayerNames()
    output_layers = [
        layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()
    ]

    with open(names) as f:
        class_names = [line.strip() for line in f]

    return net, output_layers, class_names


def detect_people_yolo(
    net_info, frame, confidence_threshold: float = 0.4, nms_threshold: float = 0.4
):
    """Run YOLO detection; returns list of (x, y, w, h) bounding boxes."""
    net, output_layers, class_names = net_info
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    boxes, confidences = [], []
    person_id = class_names.index("person") if "person" in class_names else 0

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if class_id == person_id and confidence >= confidence_threshold:
                cx, cy = int(detection[0] * w), int(detection[1] * h)
                bw, bh = int(detection[2] * w), int(detection[3] * h)
                x = max(0, cx - bw // 2)
                y = max(0, cy - bh // 2)
                boxes.append([x, y, bw, bh])
                confidences.append(confidence)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    if len(indices) == 0:
        return []
    return [boxes[i] for i in indices.flatten()]


# ---------------------------------------------------------------------------
# Age / Gender helpers
# ---------------------------------------------------------------------------
def load_age_gender_nets(age_proto, age_model, gender_proto, gender_model):
    """Load the age and gender Caffe models.

    Returns (age_net, gender_net) or (None, None) if files are missing.
    """
    for path in (age_proto, age_model, gender_proto, gender_model):
        if not os.path.isfile(path):
            print(
                f"[WARNING] Age/gender model file not found: {path}\n"
                "          Run `python download_models.py` to fetch them.\n"
                "          Age/gender estimation will be disabled."
            )
            return None, None

    age_net = cv2.dnn.readNet(age_model, age_proto)
    gender_net = cv2.dnn.readNet(gender_model, gender_proto)
    return age_net, gender_net


def predict_age_gender(face_img, age_net, gender_net):
    """Predict age bucket and gender for a cropped face image.

    Returns (age_label, gender_label) strings, or ("", "") on failure.
    """
    if face_img is None or face_img.size == 0:
        return "", ""

    blob = cv2.dnn.blobFromImage(
        face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False
    )

    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]

    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_BUCKETS[age_preds[0].argmax()]

    return age, gender


# ---------------------------------------------------------------------------
# Non-maximum suppression
# ---------------------------------------------------------------------------
def non_max_suppression(boxes, overlap_thresh: float = 0.65):
    """Apply NMS to list of (x, y, w, h) boxes."""
    if not boxes:
        return []

    boxes_arr = np.array(boxes)
    x1 = boxes_arr[:, 0]
    y1 = boxes_arr[:, 1]
    x2 = boxes_arr[:, 0] + boxes_arr[:, 2]
    y2 = boxes_arr[:, 1] + boxes_arr[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    pick = []
    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)
        xx1 = np.maximum(x1[last], x1[idxs[:-1]])
        yy1 = np.maximum(y1[last], y1[idxs[:-1]])
        xx2 = np.minimum(x2[last], x2[idxs[:-1]])
        yy2 = np.minimum(y2[last], y2[idxs[:-1]])
        w_ = np.maximum(0, xx2 - xx1 + 1)
        h_ = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w_ * h_) / areas[idxs[:-1]]
        idxs = np.delete(idxs, np.concatenate(([len(idxs) - 1], np.where(overlap > overlap_thresh)[0])))

    return [boxes[i] for i in pick]


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------
class HeatmapAccumulator:
    """Accumulates person centroid positions to build a density heatmap."""

    def __init__(self, frame_shape: tuple, sigma: int = 30):
        """
        Parameters
        ----------
        frame_shape : (height, width[, channels])
        sigma       : Gaussian blur radius in pixels applied at each centroid
        """
        self.height = frame_shape[0]
        self.width = frame_shape[1]
        self.sigma = sigma
        self._map = np.zeros((self.height, self.width), dtype=np.float32)

    def update(self, centroids):
        """Add centroids (iterable of [cx, cy]) to the accumulator."""
        for cx, cy in centroids:
            cx = int(np.clip(cx, 0, self.width - 1))
            cy = int(np.clip(cy, 0, self.height - 1))
            self._map[cy, cx] += 1.0

    def save(self, path: str, background_frame=None) -> None:
        """Blur, normalise, colour-map and save the heatmap to *path*.

        If *background_frame* is provided it is used as the base image with
        the heatmap overlaid semi-transparently.
        """
        from scipy.ndimage import gaussian_filter

        blurred = gaussian_filter(self._map, sigma=self.sigma)

        if blurred.max() == 0:
            print("[INFO] No movement recorded — heatmap will be blank.")
            blurred = np.zeros_like(blurred)
        else:
            blurred = blurred / blurred.max()

        colored = plt.cm.jet(blurred)  # RGBA array in [0,1]
        colored_uint8 = (colored[:, :, :3] * 255).astype(np.uint8)

        if background_frame is not None:
            bg = cv2.resize(background_frame, (self.width, self.height))
            bg_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
            alpha = 0.55
            overlay = (alpha * colored_uint8 + (1 - alpha) * bg_rgb).astype(np.uint8)
        else:
            overlay = colored_uint8

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.imshow(overlay)
        ax.set_title("People Traffic Heatmap", fontsize=16, fontweight="bold")
        ax.axis("off")

        # Colour-bar legend
        sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Relative Density", rotation=270, labelpad=15)

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Heatmap saved → {path}")


# ---------------------------------------------------------------------------
# Face detection (used to crop face for age/gender)
# ---------------------------------------------------------------------------
def load_face_detector():
    """Return a Haar cascade face detector (always available in OpenCV)."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


def detect_faces(face_cascade, frame, person_box):
    """Detect faces inside a person bounding box.

    Returns a list of face crops (BGR numpy arrays).
    """
    x, y, w, h = person_box
    roi = frame[y: y + h, x: x + w]
    if roi.size == 0:
        return []
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_roi, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    crops = []
    for fx, fy, fw, fh in faces:
        crop = roi[fy: fy + fh, fx: fx + fw]
        if crop.size > 0:
            crops.append(crop)
    return crops


# ---------------------------------------------------------------------------
# Movement trails
# ---------------------------------------------------------------------------
def get_trail_color(obj_id: int) -> tuple:
    """Return a unique soft pastel BGR color for the given object ID.

    Uses the ID to pick an evenly-spread hue through HSV with low saturation
    and high value so every person gets a distinct pastel shade.
    """
    hue = int((obj_id * 37) % 180)
    hsv = np.uint8([[[hue, 80, 220]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def draw_trails(frame: np.ndarray, trail_history: dict) -> None:
    """Draw a soft blob-shaped movement trail for each tracked person.

    A separate overlay layer is built with filled circles whose radius grows
    toward the most-recent (head) position, then blended onto the frame for a
    glowing pastel effect.

    Parameters
    ----------
    frame         : BGR frame to draw on (modified in-place).
    trail_history : dict mapping object-ID → deque of (cx, cy) tuples.
    """
    if not trail_history:
        return

    overlay = np.zeros_like(frame)

    for obj_id, trail in trail_history.items():
        pts = list(trail)
        n = len(pts)
        if n == 0:
            continue

        color = get_trail_color(obj_id)

        for index, (cx, cy) in enumerate(pts):
            radius = int(12 * (index / n))
            if radius <= 0:
                continue
            cv2.circle(overlay, (cx, cy), radius, color, -1)

    cv2.addWeighted(overlay, 0.6, frame, 1.0, 0, frame)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="People counter with age/gender detection and heatmap output."
    )
    ap.add_argument(
        "-i", "--input",
        default=None,
        help="Path to input video file. Omit to use the webcam (device 0).",
    )
    ap.add_argument(
        "-o", "--output",
        default=DEFAULT_HEATMAP_PATH,
        help=f"Output path for the heatmap image (default: {DEFAULT_HEATMAP_PATH}).",
    )
    ap.add_argument(
        "-c", "--confidence",
        type=float,
        default=0.4,
        help="Detection confidence threshold for YOLO (default: 0.4).",
    )
    ap.add_argument(
        "--no-age-gender",
        action="store_true",
        help="Disable age/gender estimation (faster processing).",
    )
    ap.add_argument(
        "--skip-frames",
        type=int,
        default=20,
        help="Run full detection every N frames; track in between (default: 20).",
    )
    ap.add_argument(
        "--models-dir",
        default=DEFAULT_MODELS_DIR,
        help="Directory containing model files (default: ./models/).",
    )
    ap.add_argument(
        "--display",
        action="store_true",
        help="Show live preview window (requires a display).",
    )
    return ap.parse_args()


def draw_info_panel(frame, total_count, current_count, fps):
    """Draw a semi-transparent info panel on the frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, f"Total Count : {total_count}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"In Frame    : {current_count}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(frame, f"FPS         : {fps:.1f}", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


def main():
    args = parse_args()

    # -- Open video source ---------------------------------------------------
    src = video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video source: {src}")

    ret, first_frame = cap.read()
    if not ret:
        sys.exit("[ERROR] Failed to read from video source.")

    frame_h, frame_w = first_frame.shape[:2]
    print(f"[INFO] Video source opened — resolution: {frame_w}×{frame_h}")

    # -- Initialise detectors ------------------------------------------------
    models_dir = args.models_dir
    yolo_cfg = os.path.join(models_dir, "yolov3.cfg")
    yolo_weights = os.path.join(models_dir, "yolov3.weights")
    yolo_names = os.path.join(models_dir, "coco.names")

    yolo_info = build_yolo_detector(yolo_cfg, yolo_weights, yolo_names)
    if yolo_info:
        print("[INFO] YOLOv3 detector loaded.")
        detector_mode = "yolo"
    else:
        print("[INFO] YOLO model files not found — using built-in HOG detector.")
        hog = build_hog_detector()
        detector_mode = "hog"

    # -- Age / gender --------------------------------------------------------
    age_net, gender_net = None, None
    face_cascade = None
    if not args.no_age_gender:
        age_proto = os.path.join(models_dir, "age_deploy.prototxt")
        age_model = os.path.join(models_dir, "age_net.caffemodel")
        gender_proto = os.path.join(models_dir, "gender_deploy.prototxt")
        gender_model = os.path.join(models_dir, "gender_net.caffemodel")
        age_net, gender_net = load_age_gender_nets(
            age_proto, age_model, gender_proto, gender_model
        )
        if age_net is not None:
            face_cascade = load_face_detector()
            print("[INFO] Age/gender estimation enabled.")
        else:
            print("[INFO] Age/gender estimation disabled (models not found).")

    # -- Tracker & heatmap ---------------------------------------------------
    tracker = CentroidTracker(max_disappeared=50, max_distance=80)
    heatmap = HeatmapAccumulator(
        frame_shape=(frame_h, frame_w), sigma=40
    )
    trail_history: "dict[int, deque]" = {}  # obj_id → deque of (cx, cy)

    # -- State ---------------------------------------------------------------
    last_rects: list = []          # last detected bounding boxes
    frame_index = 0
    start_time = time.time()
    last_fps_time = start_time
    fps = 0.0
    background_frame = first_frame.copy()  # saved for heatmap backdrop

    # Per-object age/gender cache to avoid re-running every frame
    ag_cache: "dict[int, tuple[str, str]]" = {}

    print("[INFO] Processing started.  Press 'q' to quit.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind to include first_frame

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of stream.")
                break

            frame_index += 1

            # -- Detection every N frames ------------------------------------
            if frame_index % args.skip_frames == 0:
                if detector_mode == "yolo":
                    last_rects = detect_people_yolo(
                        yolo_info, frame, confidence_threshold=args.confidence
                    )
                else:
                    last_rects = detect_people_hog(hog, frame)

            # -- Update tracker ----------------------------------------------
            objects = tracker.update(last_rects)

            # -- Evict deregistered IDs from the age/gender cache ------------
            if ag_cache:
                stale = set(ag_cache) - set(objects)
                for stale_id in stale:
                    del ag_cache[stale_id]

            # -- Update movement trails --------------------------------------
            for obj_id, centroid in objects.items():
                if obj_id not in trail_history:
                    trail_history[obj_id] = deque(maxlen=25)
                trail_history[obj_id].append((int(centroid[0]), int(centroid[1])))
            # Evict trails for deregistered IDs
            stale_trails = set(trail_history) - set(objects)
            for stale_id in stale_trails:
                del trail_history[stale_id]

            # -- Heatmap update ----------------------------------------------
            heatmap.update(objects.values())

            # -- Age/gender per object (throttled: only on detection frames) --
            if age_net is not None and frame_index % args.skip_frames == 0:
                for obj_id, centroid in objects.items():
                    if obj_id in ag_cache:
                        continue  # already estimated
                    # Find the bounding box closest to this centroid
                    cx, cy = centroid
                    best_box = None
                    best_dist = float("inf")
                    for x, y, w, h in last_rects:
                        box_cx = x + w // 2
                        box_cy = y + h // 2
                        d = abs(box_cx - cx) + abs(box_cy - cy)
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

            # -- Draw movement trails ----------------------------------------
            draw_trails(frame, trail_history)

            # -- Draw bounding boxes & labels --------------------------------
            for obj_id, centroid in objects.items():
                cx, cy = centroid
                label = f"ID {obj_id}"
                if obj_id in ag_cache:
                    age_lbl, gender_lbl = ag_cache[obj_id]
                    label += f" | {gender_lbl} {age_lbl}"

                # Find matching bounding box to draw rectangle
                for x, y, w, h in last_rects:
                    box_cx = x + w // 2
                    box_cy = y + h // 2
                    if abs(box_cx - cx) < 50 and abs(box_cy - cy) < 50:
                        cv2.rectangle(frame, (x, y), (x + w, y + h),
                                      (0, 200, 0), 2)
                        cv2.putText(frame, label, (x, y - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0), 1)
                        break
                # Centroid dot
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # -- FPS computation ---------------------------------------------
            now = time.time()
            if now - last_fps_time >= 1.0:
                fps = frame_index / (now - start_time)
                last_fps_time = now

            # -- Info panel --------------------------------------------------
            draw_info_panel(frame, tracker.total_count, len(objects), fps)

            # -- Display (optional) ------------------------------------------
            if args.display:
                cv2.imshow("People Counter", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Quit key pressed.")
                    break

            # Keep last valid background frame
            background_frame = frame.copy()

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        cap.release()
        if args.display:
            cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        print(
            f"\n{'='*50}\n"
            f"  Session Summary\n"
            f"{'='*50}\n"
            f"  Total unique people counted : {tracker.total_count}\n"
            f"  Frames processed            : {frame_index}\n"
            f"  Elapsed time                : {elapsed:.1f}s\n"
            f"  Average FPS                 : {frame_index / max(elapsed, 0.001):.1f}\n"
            f"{'='*50}"
        )

        heatmap.save(args.output, background_frame=background_frame)


if __name__ == "__main__":
    main()
