# sam-track Roadmap

Progress tracking for `sam-track` implementation.

## Status Summary

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | Complete | Project Setup & Dependencies |
| Phase 2 | Complete | Core Infrastructure |
| Phase 3 | Complete | Prompt Handlers |
| Phase 4 | Complete | SAM3 Integration |
| Phase 5 | Complete | Output Writers |
| Phase 6 | Complete | CLI Implementation |
| Phase 7 | Partial | Testing & CI/CD |
| Phase 8 | In Progress | Documentation & Publishing |

---

## Phase 1: Project Setup & Dependencies

- [x] 1.1.1 Create complete `pyproject.toml` with PyTorch platform markers
- [x] 1.1.2 Set up `.python-version` with Python 3.12
- [x] 1.1.3 Create `.gitignore` (Python, uv, HDF5, etc.)
- [x] 1.1.4 Create basic package structure in `src/sam_track/`
- [x] 1.1.5 Test `uv sync` on Linux with CUDA
- [x] 1.1.6 Test `uv sync` on macOS with Apple Silicon
- [x] 1.1.7 Test `uv sync` on Windows with CUDA
- [x] 1.1.8 Document first-time setup in README.md

## Phase 2: Core Infrastructure

- [x] 2.1.1 Implement `auth.py` with HF Hub authentication
- [x] 2.1.2 Add interactive login prompt with rich
- [x] 2.1.3 Test authentication flow with gated model
- ~~2.2.1 Implement `video.py` wrapper~~ (removed: use sleap-io Video directly)
- ~~2.2.2 Support all sleap-io video backends~~ (removed)
- ~~2.2.3 Handle video metadata (FPS, duration)~~ (removed)
- ~~2.3.1 Create `utils.py` with common helpers~~ (removed: add only if needed)

## Phase 3: Prompt Handlers

- [x] 3.1.1 Implement base prompt classes
- [x] 3.2.1 Implement text prompt handler
- [x] 3.3.1 Implement ROI YAML prompt handler (polygon-to-mask conversion)
- [x] 3.3.2 Test with labelroi output files
- [x] 3.4.1 Implement pose prompt handler
- [x] 3.4.2 Support track-based object IDs
- [x] 3.4.3 Support node filtering
- [x] 3.5.1 Add prompt validation

## Phase 4: SAM3 Integration

- [x] 4.1.1 Implement SAM3Tracker class
- [x] 4.1.2 Support text prompts via Sam3VideoModel
- [x] 4.1.3 Support visual prompts via Sam3TrackerVideoModel
- [x] 4.1.4 Handle multi-object tracking
- [x] 4.2.1 Add streaming inference mode
- [x] 4.2.2 Memory optimization for long videos
- [x] 4.2.3 Progress reporting with tqdm/rich

## Phase 5: Output Writers

- [x] 5.1.1 Implement BBoxWriter
- [x] 5.1.2 Add dataclass serialization
- [x] 5.2.1 Implement SegmentationWriter
- [x] 5.2.2 Add GZIP compression (level 1, single dataset)
- [x] 5.2.3 Test with large videos
- [x] 5.3.1 Implement SLPWriter for tracked poses
- [x] 5.3.2 Add pose-mask reconciliation (IDReconciler)
- [x] 5.3.3 Add track name resolution (TrackNameResolver)
- ~~5.4.1 Add combined output metadata file~~ (removed: metadata in each file)

## Phase 6: CLI Implementation

- [x] 6.1.1 Implement main CLI with typer
- [x] 6.1.2 Handle prompt option mutual exclusivity
- [x] 6.1.3 Handle output path defaults
- [x] 6.1.4 Add progress bar with rich
- [x] 6.2.1 Add --device option
- [x] 6.2.2 Add --max-frames option
- [x] 6.2.3 Add verbose/quiet mode
- [x] 6.2.4 Add --start-frame and --stop-frame options
- [x] 6.3.1 Handle CUDA OOM gracefully
- [x] 6.3.2 Add streaming mode as default (--preload for batch mode)
- [x] 6.3.3 Add graceful interrupt handling (Ctrl+C)
- [ ] 6.4.1 Add batch processing mode (multiple videos)

## Phase 7: Testing & CI/CD

- [x] 7.1.1 Create test fixtures
- [x] 7.1.2 Write unit tests for prompts
- [x] 7.1.3 Write unit tests for outputs
- [x] 7.1.4 Write CLI integration tests
- [x] 7.2.1 Set up GitHub Actions CI (lint + test)
- [x] 7.2.2 Add PyPI publish workflow with OIDC
- [ ] 7.3.1 Add GPU testing workflow
- [ ] 7.3.2 Add code coverage reporting

## Phase 8: Documentation & Publishing

- [x] 8.1.1 Write comprehensive README
- [x] 8.1.2 Add usage examples
- [x] 8.1.3 Create CONTRIBUTING.md
- [ ] 8.2.1 Set up mkdocs
- [x] 8.3.1 Configure PyPI publishing
- [x] 8.3.2 Add OIDC trusted publisher

---

## Design Decisions Made

- [x] Streaming mode for long videos: **streaming is default, --preload for batch**
- [x] ROI prompt type: **polygon-to-mask conversion** (best accuracy)
- [x] Track name propagation: **nearest-anchor flood fill** for sparse GT labels
- [x] Text prompt re-detection: **first frame only** (SAM3 propagates automatically)
- [x] Multi-object from text: **SAM3 handles internally** (returns multiple masks)
- [x] Graceful shutdown: **partial saves on interrupt** (Ctrl+C triggers save)

## Future Improvements

- [ ] Batch processing mode (multiple videos)
- [ ] GPU testing in CI
- [ ] Code coverage reporting
- [ ] mkdocs documentation site
- [ ] Video visualization output
- [ ] Re-detection at specified intervals
