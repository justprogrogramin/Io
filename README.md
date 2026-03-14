# Io — People Counter

A real-time people counting application built with **OpenCV** that:

1. **Detects people** in any video source (webcam, CCTV footage, video file) using
   a built-in HOG detector or optional YOLOv3 for improved accuracy.
2. **Counts unique individuals** cumulatively from start to finish using a
   centroid-based tracker.
3. **Estimates approximate age & gender** for each tracked person using
   pre-trained Caffe neural networks (Levi & Hassner, 2015).
4. **Generates a traffic heatmap** of the scene and saves it as a PNG when the
   program exits — visually highlighting high-traffic zones.

---

## Requirements

- Python ≥ 3.9
- See `requirements.txt` for Python package dependencies

```bash
pip install -r requirements.txt
```

---

## Quick Start

### 1 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2 — Download model files (optional but recommended)

```bash
# Age & gender models only (~15 MB):
python download_models.py

# Also download YOLOv3 weights for better people detection (~240 MB):
python download_models.py --yolo
```

> **Without model files** the app falls back to the built-in OpenCV HOG
> person detector and skips age/gender estimation automatically.

### 3 — Run

```bash
# Webcam (default device 0):
python people_counter.py

# Video file:
python people_counter.py --input path/to/video.mp4

# Show live preview window:
python people_counter.py --display

# Disable age/gender for faster processing:
python people_counter.py --no-age-gender

# Custom heatmap output path:
python people_counter.py --output my_heatmap.png
```

Press **`q`** (or `Ctrl-C`) to stop.

---

## Command-Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `-i / --input` | webcam | Path to video file |
| `-o / --output` | `heatmap_output.png` | Heatmap save path |
| `-c / --confidence` | `0.4` | YOLO detection confidence threshold |
| `--no-age-gender` | — | Disable age/gender estimation |
| `--skip-frames` | `20` | Full detection every N frames |
| `--models-dir` | `./models/` | Directory with model files |
| `--display` | — | Show live preview window |

---

## Output

When the program ends (video finished, `q` pressed, or `Ctrl-C`) it prints a
session summary and saves a heatmap:

```
==================================================
  Session Summary
==================================================
  Total unique people counted : 42
  Frames processed            : 3600
  Elapsed time                : 120.3s
  Average FPS                 : 29.9
==================================================
[INFO] Heatmap saved → heatmap_output.png
```

The heatmap PNG uses a **jet colour scale** (blue → green → yellow → red) to
show where people spent the most time.  If a background frame is available it
is blended with the heatmap for context.

---

## Project Structure

```
.
├── people_counter.py    # Main application
├── download_models.py   # Model download helper
├── requirements.txt     # Python dependencies
├── models/              # Model files (git-ignored)
└── tests/
    └── test_people_counter.py
```

---

## Running the Tests

```bash
python -m pytest tests/ -v
# or
python -m unittest discover tests/
```

---

## Models & Credits

| Model | Source |
|-------|--------|
| Age & Gender estimation | G. Levi & T. Hassner — *Age and Gender Classification Using Convolutional Neural Networks* (CVPR 2015) |
| YOLOv3 object detection | J. Redmon & A. Farhadi — *YOLOv3: An Incremental Improvement* (2018) |
| HOG person detector | N. Dalal & B. Triggs — *Histograms of Oriented Gradients for Human Detection* (CVPR 2005), bundled with OpenCV |
