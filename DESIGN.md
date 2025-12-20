We're working on a python app called `sam-track`.

The app should be designed to be `uv` runnable as the primary way to execute it.

The `pyproject.toml` setup should figure out how to install `pytorch` with the appropriate system dependencies for GPU support (Linux/Windows with CUDA) or running on Macs (Apple Silicon). See relevant `uv` docs below.

Use SAM3 via the `transformers` library mostly via the `Sam3TrackerVideoModel` and `Sam3TrackerVideoProcessor` classes. This will require being logged in to HuggingFace Hub since this model requires auth. We should handle that gracefully, preferably by prompting the user to do the `uv run hf auth login` or whatever will work in this setting. See relevant SAM3 docs below.

Use `sleap-io[all]` for video I/O handling. See relevant `sleap-io` docs below.

The main usage should be via the CLI, like this:

```
# either --bbox or --seg must be specified, which determines the output modality
# --text specifies the text prompt for the object to be tracked
uv run sam-track video.mp4 --text "mouse" --bbox
# --roi specifies a yaml file with prespecified ROIs to use as the seed prompt
uv run sam-track video.mp4 --roi rois.yml --bbox
# --pose specifies a SLP file with prespecified poses to be treated as prompt points (will use the first LabeledFrame)
uv run sam-track video.mp4 --pose labels.slp --bbox
# both --bbox and --seg can be specified to output both modalities
uv run sam-track video.mp4 --roi rois.yml --bbox --seg
# bbox outputs to video.bbox.json unless overridden
# seg outputs to video.seg.h5 unless overridden
uv run sam-track video.mp4 --roi rois.yml --bbox "bbox_tracks.json" --seg "seg_tracks.h5"
```

The ROIs YAML input is in the format provided by `labelroi`. See `scratch/repos/labelroi` or `scratch/repos/vibes/labelroi`.
Sources:
- https://github.com/talmolab/labelroi
- https://github.com/talmolab/vibes

`uv` docs are available in: `scratch/docs/uv-pytorch.md`
Sources:
- https://raw.githubusercontent.com/astral-sh/uv/refs/heads/main/docs/guides/integration/pytorch.md

SAM3 docs are available in: `scratch/docs/sam3.md`
Sources:
- https://huggingface.co/facebook/sam3/raw/main/README.md
- https://huggingface.co/docs/transformers/model_doc/sam3_tracker_video.md

`sleap-io` docs are available in: `scratch/docs/sleap-io.md` and the repo code can be explored at `scratch/repos/sleap-io`
Sources:
- https://raw.githubusercontent.com/talmolab/sleap-io/refs/heads/main/docs/examples.md
- https://github.com/talmolab/sleap-io


Other dev notes:
- This is hosted at `talmolab/sam-track`
- Use `black` style formatting linted with `ruff`
- We will publish to PyPI using OIDC for auth, `uv` for packagin, and triggered via GitHub release.
- Use `click`, `typer` and `rich` for the CLI
- Use `pytest` for testing and CI