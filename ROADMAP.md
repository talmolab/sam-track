# sam-track Roadmap

Progress tracking for `sam-track` implementation. See [Investigation Notes](scratch/2025-12-19-sam-track-roadmap/README.md) for detailed design and code examples.

## Phase 1: Project Setup & Dependencies

- [ ] 1.1.1 Create complete `pyproject.toml` with PyTorch platform markers
- [ ] 1.1.2 Set up `.python-version` with Python 3.11
- [ ] 1.1.3 Create `.gitignore` (Python, uv, HDF5, etc.)
- [ ] 1.1.4 Create basic package structure in `src/sam_track/`
- [ ] 1.1.5 Test `uv sync` on Linux with CUDA
- [ ] 1.1.6 Test `uv sync` on macOS with Apple Silicon
- [ ] 1.1.7 Test `uv sync` on Windows with CUDA
- [ ] 1.1.8 Document first-time setup in README.md

## Phase 2: Core Infrastructure

- [ ] 2.1.1 Implement `auth.py` with HF Hub authentication
- [ ] 2.1.2 Add interactive login prompt with rich
- [ ] 2.1.3 Test authentication flow with gated model
- [ ] 2.2.1 Implement `video.py` wrapper
- [ ] 2.2.2 Support all sleap-io video backends
- [ ] 2.2.3 Handle video metadata (FPS, duration)
- [ ] 2.3.1 Create `utils.py` with common helpers

## Phase 3: Prompt Handlers

- [ ] 3.1.1 Implement base prompt classes
- [ ] 3.2.1 Implement text prompt handler
- [ ] 3.3.1 Implement ROI YAML prompt handler
- [ ] 3.3.2 Test with labelroi output files
- [ ] 3.4.1 Implement pose prompt handler
- [ ] 3.4.2 Support track-based object IDs
- [ ] 3.4.3 Support node filtering
- [ ] 3.5.1 Add prompt validation

## Phase 4: SAM3 Integration

- [ ] 4.1.1 Implement SAM3Tracker class
- [ ] 4.1.2 Support text prompts via Sam3VideoModel
- [ ] 4.1.3 Support visual prompts via Sam3TrackerVideoModel
- [ ] 4.1.4 Handle multi-object tracking
- [ ] 4.2.1 Add streaming inference mode
- [ ] 4.2.2 Memory optimization for long videos
- [ ] 4.2.3 Progress reporting with tqdm/rich

## Phase 5: Output Writers

- [ ] 5.1.1 Implement BBoxWriter
- [ ] 5.1.2 Add dataclass serialization
- [ ] 5.2.1 Implement SegmentationWriter
- [ ] 5.2.2 Add GZIP compression
- [ ] 5.2.3 Test with large videos
- [ ] 5.3.1 Add combined output metadata file

## Phase 6: CLI Implementation

- [ ] 6.1.1 Implement main CLI with typer
- [ ] 6.1.2 Handle prompt option mutual exclusivity
- [ ] 6.1.3 Handle output path defaults
- [ ] 6.1.4 Add progress bar with rich
- [ ] 6.2.1 Add --device option
- [ ] 6.2.2 Add --max-frames option
- [ ] 6.2.3 Add verbose/quiet mode
- [ ] 6.3.1 Handle CUDA OOM gracefully
- [ ] 6.3.2 Add batch processing mode

## Phase 7: Testing & CI/CD

- [ ] 7.1.1 Create test fixtures
- [ ] 7.1.2 Write unit tests for prompts
- [ ] 7.1.3 Write unit tests for outputs
- [ ] 7.1.4 Write CLI integration tests
- [ ] 7.2.1 Set up GitHub Actions CI
- [ ] 7.2.2 Add GPU testing workflow
- [ ] 7.2.3 Add code coverage reporting

## Phase 8: Documentation & Publishing

- [ ] 8.1.1 Write comprehensive README
- [ ] 8.1.2 Add usage examples
- [ ] 8.1.3 Set up mkdocs
- [ ] 8.2.1 Configure PyPI publishing
- [ ] 8.2.2 Add OIDC authentication

---

## Open Design Questions

- [ ] Text prompt re-detection frequency (first frame only vs periodic)
- [ ] Multi-object handling from single text prompt
- [ ] Streaming mode for long videos
- [ ] ROI prompt type (bounding box vs centroid)
- [ ] Graceful shutdown with partial saves
