# sam-track

Track objects in videos using [SAM3](https://github.com/facebookresearch/sam3) (Segment Anything Model 3).

## Features

- **Text prompts**: Track objects by description (e.g., "mouse", "fly")
- **ROI prompts**: Track from [labelroi](https://github.com/talmolab/labelroi) annotations
- **Pose prompts**: Track from [SLEAP](https://sleap.ai) pose labels
- **Multiple outputs**: Bounding box tracks (JSON) and segmentation masks (HDF5)

## Quick Start

```bash
git clone https://github.com/talmolab/sam-track && cd sam-track
uv run sam-track system
```

Expected output (Linux with CUDA):
```
               System Information
┌───────────────────┬─────────────────────────┐
│ sam-track version │ 0.1.0                   │
│ Python version    │ 3.12.x                  │
│ Platform          │ Linux-...               │
│ PyTorch version   │ 2.9.1+cu130             │
│ CUDA available    │ True                    │
│ Driver version    │ 580.xx.xx               │
│ CUDA version      │ 13.0                    │
│ ...               │                         │
└───────────────────┴─────────────────────────┘

✓ CUDA tensor operations working
```

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/talmolab/sam-track
cd sam-track
uv sync
```

### GPU Requirements

sam-track uses PyTorch with CUDA 13.0. Minimum driver versions:

| Platform | Minimum Driver |
|----------|----------------|
| Linux    | 580.65.06      |
| Windows  | 580.65         |

macOS with Apple Silicon uses MPS (no driver required).

Run `uv run sam-track system` to check your setup.

### First-time Setup

SAM3 is a gated model requiring a HuggingFace account and access approval.

**Check your auth status:**
```bash
uv run sam-track auth
```

**If not authenticated:**

1. Go to https://huggingface.co/settings/tokens
2. Click **Create new token**
3. Name it `sam-track` and select **Read** permission (top tab, not fine-grained)
4. Login with your token:
   ```bash
   uv run sam-track auth --token hf_...
   ```

**If no model access:**

1. Go to https://huggingface.co/facebook/sam3
2. Fill out the access request form and accept the license
3. Run `uv run sam-track auth` again to verify

## Usage

```bash
# Track by text prompt, output bounding boxes
uv run sam-track video.mp4 --text "mouse" --bbox

# Track from ROI file, output both formats
uv run sam-track video.mp4 --roi rois.yaml --bbox --seg

# Track from pose labels, custom output paths
uv run sam-track video.mp4 --pose labels.slp --bbox tracks.json --seg masks.h5

# Output both bounding boxes and segmentation masks
uv run sam-track video.mp4 --text "fly" --bbox --seg
```

### Options

| Option | Description |
|--------|-------------|
| `--text`, `-t` | Text prompt for object detection |
| `--roi`, `-r` | Path to labelroi YAML file |
| `--pose`, `-p` | Path to SLEAP .slp file |
| `--bbox`, `-b` | Output bounding boxes (default: `video.bbox.json`) |
| `--seg`, `-s` | Output segmentation masks (default: `video.seg.h5`) |
| `--max-frames`, `-n` | Limit frames to process |
| `--device` | Device to use (e.g., `cuda:0`, `cpu`) |

## Output Formats

### Bounding Boxes (JSON)

```json
{
  "metadata": {"video_source": "video.mp4", "width": 1920, "height": 1080},
  "tracks": [
    {
      "track_id": 0,
      "bboxes": [
        {"frame_idx": 0, "x_min": 100, "y_min": 200, "x_max": 300, "y_max": 400, "confidence": 0.95}
      ]
    }
  ]
}
```

### Segmentation Masks (HDF5)

```
/metadata/          - Video info, compression settings
/frames/frame_N/    - Per-frame masks, track IDs, confidences
/tracks/track_N/    - Per-track metadata
```

## Development

See [CLAUDE.md](CLAUDE.md) for development practices and [ROADMAP.md](ROADMAP.md) for implementation status.

## License

BSD-3-Clause
