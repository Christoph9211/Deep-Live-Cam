# AGENT.md

This file outlines how to work effectively with the Deep-Live-Cam codebase (GUI and CLI for swapping or enhancing faces on images, videos, and live camera input).

## Project Info

- Version 1.9 (`modules/metadata.py`) tagged as GitHub Edition.
- Primary entry point `run.py` delegates to `modules/core.py` and respects Python 3.9+.
- Recommended conda env name `dlcam`; ensure it is active before installing Python packages.
- FFmpeg (ffmpeg and ffprobe) must be on PATH for all video and audio operations.
- Optional GPU acceleration via ONNX Runtime providers: `cpu`, `cuda`, `dml`, `rocm`, `coreml` (pick with `--execution-provider`).

## Setup

- Ensure `pip` is up to date and PyTorch is installed from the matching index for your GPU stack.
- Sample Windows or Linux install commands:
  
  ```bash
  pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --extra-index-url https://pypi.org/simple
  pip install -r requirements.txt --no-deps
  ```

- macOS arm64 users rely on MPS support (`torch` wheel from `cu126` index and `onnxruntime-silicon`).
- `requirements.txt` locks UI (CustomTkinter), media (OpenCV, Pillow), inference (onnxruntime-gpu, insightface, GFPGAN), NSFW filter, and MediaPipe dependencies.
- Pre-commit is configured; run `pre-commit install` then `pre-commit run --all-files` before pushing.

## Run Modes

- Launch GUI: `python run.py` (loads CustomTkinter interface defined in `modules/ui.py`).
- Headless image->image swap: `python run.py -s src.jpg -t dst.jpg -o out.jpg --frame-processor face_swapper`.
- Headless image->video swap: add `--keep-fps --keep-audio` to preserve timing and soundtrack.
- Enhancement only: `python run.py -t input.jpg -o enhanced.jpg --frame-processor face_enhancer`.
- Live usage: start GUI, choose camera, optional `--live-mirror` and `--live-resizable` flags; live pipeline flows through `modules/video_capture.py` and `modules/capturer.py`.
- Scripted runs accept `--execution-provider cuda` (or other providers) and respect saved UI toggles from `switch_states.json`.

## CLI Essentials

- Inputs and outputs: `-s/--source`, `-t/--target`, `-o/--output`, plus auto output naming via `modules/utilities.normalize_output_path`.
- Frame pipeline: `--frame-processor` accepts `face_swapper` and `face_enhancer`; UI checkboxes in `modules/ui.py` mirror these via `modules.globals.fp_ui`.
- Video handling: `--keep-fps`, `--keep-audio`, `--keep-frames`, `--video-encoder` (`libx264`, `libx265`, `libvpx-vp9`), `--video-quality` (0-51 CRF).
- Face selection: `--many-faces`, `--map-faces`, `--mouth-mask`, `--preserve-teeth`, `--preserve-hairline`, plus occlusion-aware compositing defaults in `modules/globals.py`.
- Segmentation and smoothing: `--segmenter-backend`, `--smoothing` with `--smoothing-stream-only`, `--smoothing-use-fps`, `--smoothing-fps`, `--smoothing-min-cutoff`, `--smoothing-beta`, `--smoothing-dcutoff`.
- Runtime control: `--execution-provider`, `--execution-threads`, `--max-memory`, `--nsfw-filter`, `--lang` (localization via `modules/gettext.py`), deprecated flags hidden from help for backwards compatibility.

## Architecture

- `modules/core.py` parses CLI, enforces environment checks, configures globals, orchestrates processing, and handles temp files, fps, audio, and live streaming.
- `modules/globals.py` stores runtime state (paths, toggles, smoothing, occlusion, preserved regions, color correction, threading limits).
- `modules/ui.py` builds the CustomTkinter GUI, persists switches in `switch_states.json`, manages live preview windows, camera discovery (`cv2_enumerate_cameras`), and exposes advanced settings panes.
- `modules/processors/frame/face_swapper.py` loads InsightFace InSwapper, supports multi-face mapping, semantic region preservation (mouth and hair masks), One-Euro landmark smoothing, and occlusion-aware compositing.
- `modules/processors/frame/face_enhancer.py` wraps GFPGAN with platform-sensitive device selection, enabling enhancement-only flows or chained processors.
- `modules/utilities.py` centralizes FFmpeg wrappers (detect fps, stream writer, audio restore), temp directory lifecycle, and model downloads.

## Support Modules

- `modules/face_analyser.py` uses InsightFace FaceAnalysis for detection, mapping, and MiniBatchKMeans clustering to build source-target maps.
- `modules/cluster_analysis.py` picks cluster counts via silhouette scoring, enabling `--map-faces` automation.
- `modules/segmenters/semantic.py` integrates MediaPipe Face Mesh for mouth masking and falls back to BiSeNet ONNX when available; `modules/segmenters/bisenet_onnx.py` loads optional segmentation models and caches masks.
- `modules/video_capture.py` and `modules/capturer.py` abstract camera access and optional RGB conversion when UI color correction is toggled.
- `modules/predicter.py` wires in the optional NSFW filter (OpenNSFW2) for stills and videos, respecting color correction.
- `modules/gettext.py` loads JSON locales (default English, `locales/zh.json` bundled) and provides translation helper `_()` to the UI.

## Models and Assets

- Place InsightFace ONNX models (`inswapper_128.onnx`, `inswapper_128_fp16.onnx`) and optional BiSeNet exports under `modules/models/`.
- GFPGAN weights (`GFPGANv1.4.pth`) auto-download into the same folder during first run of the enhancer.
- Set `DLC_BISENET_ONNX_PATH` or `DLC_BISENET_ONNX_URL` to use custom segmentation weights; otherwise the loader searches local files.
- Sample media and marketing assets live in `media/`; keep large binaries out of source control changes unless necessary.

## UI Toggles and Persistence

- UI switch states persist to `switch_states.json`; toggles include color correction, show FPS overlay, mouth mask preview, occlusion sensitivity, preserved regions, smoothing, and live window sizing.
- Color correction influences capture and NSFW prediction workflows (`modules/capturer.py`, `modules/predicter.py`).
- Occlusion-aware compositing defaults on to protect foreground props; tune sensitivity via advanced UI slider.
- Multi-face mapper panes let users assign source-target pairs with thumbnails generated by `modules/face_analyser.py` utilities.

## Packaging and Distribution

- `tools/create_self_extracting.py` builds a self-extracting archive; see `docs/self_extracting.md` for workflow and distribution tips.
- Generated artifacts land in `dist/` (`Deep-Live-Cam.sfx.py`, `.7z`, Windows installer, bundled 7-Zip SFX runtime under `dist/sfx/`).
- Batch helpers `run-cuda.bat` and `run-directml.bat` set default providers for Windows shortcuts.
- When shipping releases, validate that required models (excluding licensed or large files) are present or documented for download.

## Testing and QA

- Prefer `python -m unittest` or targeted script runs for regressions; there is no standalone test suite in this repository.
- Validate FFmpeg availability with `ffmpeg -version` and `ffprobe -version` before debugging video issues.
- Monitor GPU memory by toggling `--execution-threads` and `modules/globals.max_memory`; multi-threaded swaps are handled in `modules/processors/frame/core.py` via `ThreadPoolExecutor` batches.
- Keep translations in sync when adding UI strings; fallback keys render English text if no locale entry is found.

## Housekeeping

- Temporary frames store under `temp/<target_name>`; disable cleanup with `--keep-frames` for inspection.
- `modules/globals.WORKFLOW_DIR` reserved for future automation; avoid deleting until workflow features land.
- Large ONNX/PyTorch weights are tracked via `.gitignore`; ensure distribution packages either include them or reference download steps.
- Stay alert for upstream InsightFace, GFPGAN, or PyTorch updates that may require dependency pin changes or provider adjustments.
