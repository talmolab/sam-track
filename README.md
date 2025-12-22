# sam-track

A `uv`-native CLI for tracking objects in videos using [SAM3](https://github.com/facebookresearch/sam3) (Segment Anything Model 3).

## Features

- **Three prompt modes**: Track by text description, ROI polygons, or SLEAP pose keypoints
- **Three output formats**: Bounding boxes (JSON), segmentation masks (HDF5), tracked poses (SLP)
- **Memory-efficient**: Streaming mode processes videos frame-by-frame
- **SLEAP integration**: Link untracked pose predictions to consistent identities

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

### As a uv tool (recommended)

```bash
# Linux/Windows with NVIDIA GPU (CUDA 13.0)
uv tool install sam-track --python 3.13 \
  --index https://download.pytorch.org/whl/cu130 \
  --index https://pypi.org/simple

# macOS with Apple Silicon (MPS)
uv tool install sam-track --python 3.13
```

After installation, `sam-track` is available globally.

### Ad-hoc with uvx

```bash
# Linux/Windows with NVIDIA GPU
uvx --python 3.13 \
  --index https://download.pytorch.org/whl/cu130 \
  --index https://pypi.org/simple \
  sam-track --help

# macOS with Apple Silicon
uvx --python 3.13 sam-track --help
```

### From source

```bash
git clone https://github.com/talmolab/sam-track && cd sam-track
uv sync
uv run sam-track --help
```

### GPU Requirements

| Platform | Requirement |
|----------|-------------|
| Linux | NVIDIA driver 580.65.06+ (CUDA 13.0) |
| Windows | NVIDIA driver 580.65+ (CUDA 13.0) |
| macOS | Apple Silicon (MPS, no driver needed) |

Check your setup:

```bash
sam-track system
```

## First-time Setup

SAM3 is a gated model requiring HuggingFace authentication.

**1. Check status:**

```bash
sam-track auth
```

**2. If not authenticated**, create a token:

1. Go to https://huggingface.co/settings/tokens
2. Click **Create new token**
3. Name it `sam-track`, select **Read** permission
4. Login:

```bash
sam-track auth --token hf_...
```

**3. If no model access**, request it:

1. Go to https://huggingface.co/facebook/sam3
2. Fill out the access request form
3. Run `sam-track auth` again to verify

## Quick Start

```bash
# Track a mouse by text description, output bounding boxes
sam-track track video.mp4 --text "mouse" --bbox

# Track from ROI annotations, output masks
sam-track track video.mp4 --roi annotations.yml --seg

# Track from SLEAP poses, output tracked SLP
sam-track track video.mp4 --pose labels.slp --slp
```

---

## Prompting Modes

sam-track supports three ways to specify what to track:

### Text Prompts (`--text`)

Track objects by natural language description. SAM3 detects matching objects in the first frame and tracks them through the video.

```bash
# Track a single object type
sam-track track video.mp4 --text "mouse" --bbox

# Track with description
sam-track track video.mp4 --text "black mouse" --bbox --seg

# Output to custom paths
sam-track track video.mp4 --text "fly" \
  --bbox-output fly_tracks.json \
  --seg-output fly_masks.h5
```

### ROI Prompts (`--roi`)

Track from polygon regions defined in a [labelroi](https://github.com/talmolab/labelroi) YAML file. Polygons are converted to binary masks for SAM3.

```bash
# Track from ROI annotations
sam-track track video.mp4 --roi rois.yml --bbox

# Output both formats
sam-track track video.mp4 --roi rois.yml --bbox --seg
```

**ROI YAML format:**

```yaml
video: video.mp4
frame_idx: 0
resolution: [1920, 1080]
rois:
  - id: 0
    name: mouse1
    polygon: [[100, 200], [150, 200], [150, 250], [100, 250]]
  - id: 1
    name: mouse2
    polygon: [[300, 400], [350, 400], [350, 450], [300, 450]]
```

### Pose Prompts (`--pose`)

Track from [SLEAP](https://sleap.ai) pose annotations. Keypoints from labeled frames are used as point prompts for SAM3.

```bash
# Track from poses, output tracked SLP
sam-track track video.mp4 --pose labels.slp --slp

# Output all formats
sam-track track video.mp4 --pose labels.slp --bbox --seg --slp

# Exclude body parts from matching
sam-track track video.mp4 --pose labels.slp --slp \
  --exclude-nodes "tail_tip,left_ear,right_ear"

# Only keep poses that matched a SAM3 mask
sam-track track video.mp4 --pose labels.slp --slp --remove-unmatched

# Only output masks/boxes that matched a pose
sam-track track video.mp4 --pose labels.slp --bbox --seg --filter-by-pose
```

**Pose mode features:**

- Uses keypoints as point prompts (visible keypoints only)
- Matches poses to SAM3 masks using Hungarian algorithm
- Propagates GT track names (e.g., "mouse1") to all outputs
- Supports multi-frame labeled SLPs (uses nearest GT frame for matching)
- Preserves PredictedInstance types and confidence scores

---

## Output Formats

### Bounding Boxes (`--bbox`)

JSON format with track metadata, per-frame detections, and statistics.

**Default path:** `<video>.bbox.json`

```json
{
  "metadata": {
    "version": "1.0",
    "video_source": "video.mp4",
    "width": 1920,
    "height": 1080,
    "fps": 30.0,
    "total_frames": 1000,
    "tracking_model": "facebook/sam3",
    "prompt_type": "text",
    "prompt_info": {"text": "mouse"},
    "created_at": "2025-12-21T12:00:00"
  },
  "tracks": [
    {
      "track_id": 0,
      "name": "mouse1",
      "first_frame": 0,
      "last_frame": 999,
      "avg_confidence": 0.95,
      "detections": [
        {
          "frame_idx": 0,
          "x_min": 100.0,
          "y_min": 200.0,
          "x_max": 300.0,
          "y_max": 400.0,
          "confidence": 0.98,
          "width": 200.0,
          "height": 200.0,
          "area": 40000.0
        }
      ]
    }
  ],
  "statistics": {
    "total_tracks": 2,
    "total_detections": 1998,
    "frames_with_detections": 1000,
    "avg_confidence": 0.94
  }
}
```

### Segmentation Masks (`--seg`)

HDF5 format with compressed binary masks and per-track metadata.

**Default path:** `<video>.seg.h5`

```
/masks              - uint8 (T, N, H, W) binary masks, GZIP compressed
/frame_indices      - int32 (T,) frame indices
/track_ids          - int32 (T, N) track ID per mask
/confidences        - float32 (T, N) detection confidence
/num_objects        - int32 (T,) objects per frame
/metadata/
  version           - "1.0"
  video_source      - "video.mp4"
  width, height     - frame dimensions
  fps               - video frame rate
  total_frames      - frames processed
  compression       - "gzip"
  compression_level - 1
/tracks/
  track_0/
    name            - "mouse1"
    first_frame     - 0
    last_frame      - 999
    avg_confidence  - 0.95
  track_1/
    ...
```

**Reading masks in Python:**

```python
import h5py

with h5py.File("video.seg.h5", "r") as f:
    masks = f["masks"][:]          # (T, N, H, W) uint8
    frame_indices = f["frame_indices"][:]
    track_ids = f["track_ids"][:]

    # Get mask for frame 100, track 0
    frame_mask = masks[100, 0]     # (H, W) binary mask
```

### Tracked Poses (`--slp`)

SLEAP SLP format with SAM3-assigned track identities. Only available with `--pose`.

**Default path:** `<pose>.sam-tracked.slp`

The output SLP contains:
- All instances from the input with SAM3-assigned tracks
- Track names propagated from GT labels (e.g., "mouse1", "mouse2")
- `tracking_score` field with pose-mask matching confidence
- Preserved instance types (Instance vs PredictedInstance)

**Loading in Python:**

```python
import sleap_io as sio

labels = sio.load_slp("labels.sam-tracked.slp")
for lf in labels:
    for inst in lf.instances:
        print(f"Frame {lf.frame_idx}: {inst.track.name}")
```

---

## CLI Reference

### Main Command

```bash
sam-track track VIDEO [OPTIONS]
```

### Prompt Options (exactly one required)

| Option | Description |
|--------|-------------|
| `--text`, `-t` | Text description of object to track |
| `--roi`, `-r` | Path to labelroi YAML file |
| `--pose`, `-p` | Path to SLEAP SLP file |

### Output Options (at least one required)

| Option | Description |
|--------|-------------|
| `--bbox`, `-b` | Enable bounding box output |
| `--bbox-output`, `-B` | Custom bbox output path (implies `--bbox`) |
| `--seg`, `-s` | Enable segmentation mask output |
| `--seg-output`, `-S` | Custom seg output path (implies `--seg`) |
| `--slp` | Output path for tracked SLP (pose mode only) |

### Pose Mode Options

| Option | Description |
|--------|-------------|
| `--remove-unmatched` | Remove poses without SAM3 mask matches |
| `--exclude-nodes` | Comma-separated nodes to exclude from matching |
| `--filter-by-pose` | Only output masks/boxes that matched a pose |

### Processing Options

| Option | Description |
|--------|-------------|
| `--device`, `-d` | Device for inference (cuda, cuda:0, mps, cpu) |
| `--start-frame` | Frame index to start from (0-indexed, default: 0) |
| `--stop-frame` | Frame index to stop at (exclusive) |
| `--max-frames`, `-n` | Maximum frames to process from start |
| `--preload` | Load all frames upfront (uses more memory) |
| `--quiet`, `-q` | Suppress progress output |

### Other Commands

```bash
sam-track auth [--token TOKEN]  # Check/set HuggingFace auth
sam-track system                # Display GPU/system info
sam-track --version             # Show version
```

---

## Examples

### Track mice in a behavioral video

```bash
# Simple text tracking
sam-track track experiment.mp4 --text "mouse" --bbox --seg

# Process only frames 1000-2000
sam-track track experiment.mp4 --text "mouse" --bbox \
  --start-frame 1000 --stop-frame 2000

# Process 500 frames starting from frame 1000
sam-track track experiment.mp4 --text "mouse" --bbox \
  --start-frame 1000 --max-frames 500
```

### Track from SLEAP predictions

```bash
# Add track identities to untracked predictions
sam-track track video.mp4 --pose predictions.slp --slp

# Get all outputs with consistent track names
sam-track track video.mp4 --pose predictions.slp \
  --bbox --seg --slp

# Exclude tail from matching (often occluded)
sam-track track video.mp4 --pose predictions.slp --slp \
  --exclude-nodes "tail_tip,tail_mid"
```

### Use specific GPU

```bash
# Use second GPU
sam-track track video.mp4 --text "fly" --bbox --device cuda:1

# Force CPU (slow but works without GPU)
sam-track track video.mp4 --text "fly" --bbox --device cpu
```

---

## Troubleshooting

### CUDA out of memory

Try these in order:

1. Use streaming mode (default) - don't use `--preload`
2. Process fewer frames: `--max-frames 100`
3. Use a smaller portion: `--start-frame 0 --stop-frame 500`
4. Close other GPU applications

### Authentication errors

```bash
# Check current status
sam-track auth

# Re-login with new token
sam-track auth --token hf_xxxxx
```

### Driver too old

SAM3 requires CUDA 13.0. Check your driver version:

```bash
sam-track system
nvidia-smi
```

Minimum drivers: Linux 580.65.06, Windows 580.65

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

BSD-3-Clause
