# sam-track

Track objects in videos using [SAM3](https://github.com/facebookresearch/sam3) (Segment Anything Model 3).

## Features

- **Text prompts**: Track objects by description (e.g., "mouse", "fly")
- **ROI prompts**: Track from [labelroi](https://github.com/talmolab/labelroi) annotations
- **Pose prompts**: Track from [SLEAP](https://sleap.ai) pose labels
- **Multiple outputs**: Bounding box tracks (JSON) and segmentation masks (HDF5)

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/talmolab/sam-track
cd sam-track
uv sync
```

### First-time Setup

SAM3 is a gated model requiring a HuggingFace account and access approval:

1. Create a HuggingFace account at https://huggingface.co/join
2. Go to https://huggingface.co/facebook/sam3 and fill out the access request form at the top of the page
3. Login (will prompt you to create/enter a token):
   ```bash
   uv run huggingface-cli login
   ```

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
