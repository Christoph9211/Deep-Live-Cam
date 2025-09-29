# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Project: Deep-Live-Cam (GUI and CLI for face swap/enhance on images, videos, and live camera)

Prerequisites and setup

- Python 3.9+ (the app checks and requires >= 3.9)
- FFmpeg installed and on PATH (ffmpeg/ffprobe used for decoding/encoding)
- Conda environment (recommended). On this machine the environment is typically named dlcam. The user’s curated rule indicates a Linux path at /home/christoph92/miniconda/envs/dlcam; on Windows, just activate the dlcam environment by name.
- GPUs optional. ONNX Runtime providers supported include cpu, cuda (NVIDIA), dml (Windows DirectML), rocm (AMD), coreml (macOS). Choose with --execution-provider.

Setup commands (PowerShell)

- Create/activate env and install dependencies:
  - conda create -y -n dlcam python=3.10
  - conda activate dlcam
  - pip install -r requirements.txt
  - Optional (needed for face mapping features and dev tooling): pip install scikit-learn pre-commit

Common commands

- Run GUI (recommended for interactive use):
  - python run.py
- Run headless face swap (image -> image):
  - python run.py -s path\to\source.jpg -t path\to\target.jpg -o path\to\out.jpg --frame-processor face_swapper
- Run headless face swap (image -> video), keep original FPS and audio:
  - python run.py -s path\to\source.jpg -t path\to\clip.mp4 -o path\to\out.mp4 --frame-processor face_swapper --keep-fps --keep-audio
- Run only face enhancement (GFPGAN):
  - python run.py -t path\to\input.jpg -o path\to\enhanced.jpg --frame-processor face_enhancer
- Select hardware execution provider (examples):
  - CPU: python run.py ... --execution-provider cpu
  - NVIDIA CUDA: python run.py ... --execution-provider cuda
  - Windows DirectML (AMD/Intel/NVIDIA via D3D12): python run.py ... --execution-provider dml
- Useful flags you’ll commonly toggle:
  - --many-faces to process all faces instead of the leftmost
  - --map-faces to build source↔target mappings via clustering (requires scikit-learn)
  - --live-mirror, --live-resizable for live preview behavior (GUI)
  - --segmenter-backend auto|mediapipe|bisenet to enable semantic region preservation
  - --smoothing, --smoothing-fps, --smoothing-min-cutoff, --smoothing-beta, --smoothing-dcutoff to stabilize landmarks
  - --mouth-mask, --preserve-teeth, --preserve-hairline to selectively preserve regions during compositing

Linting

- Install and run pre-commit hooks:
  - pre-commit install
  - pre-commit run --all-files

High-level architecture

- Entry point: run.py → modules/core.py
  - Parses CLI, sets global runtime config (modules/globals.py), validates environment (Python ≥3.9, ffmpeg available), chooses ONNX Runtime providers, and orchestrates processing.
  - Headless mode runs directly; otherwise launches the CustomTkinter UI (modules/ui.py).
- UI: modules/ui.py
  - CustomTkinter GUI for selecting source/target/output, toggling options, showing previews, and starting a webcam Live preview (modules/video_capture.py). UI state persists in switch_states.json.
- Global state: modules/globals.py
  - Central configuration switches (provider, threads, fps/audio/frames toggles, region preservation, smoothing, etc.) read by all components.
- Processing pipeline
  - Frame processors are pluggable via modules/processors/frame/core.py, which dynamically loads modules implementing: pre_check, pre_start, process_frame, process_image, process_video, process_frame_stream.
  - Available processors in this repo:
    - face_swapper (modules/processors/frame/face_swapper.py):
      - Uses insightface InSwapper ONNX model (inswapper_128.onnx preferred; inswapper_128_fp16.onnx fallback). Applies optional occlusion-aware compositing and semantic region preservation.
    - face_enhancer (modules/processors/frame/face_enhancer.py):
      - Uses GFPGAN (GFPGANv1.4.pth) to enhance faces.
  - Segmenters (semantic region masks):
    - modules/segmenters/semantic.py (MediaPipe Face Mesh-based masks)
    - modules/segmenters/bisenet_onnx.py (optional BiSeNet ONNX backend via onnxruntime) for mouth/inner mouth/hair masks
  - Face analysis and mapping:
    - modules/face_analyser.py with insightface.app.FaceAnalysis for detection/landmarks/embeddings
    - modules/cluster_analysis.py (MiniBatchKMeans + silhouette scoring) to cluster target faces when --map-faces is used
  - Video/audio/IO utilities:
    - modules/utilities.py wraps ffmpeg/ffprobe, manages temp frames (temp/<target_name>), and writes output using a rawvideo pipe to ffmpeg.
  - NSFW filtering (optional):
    - modules/predicter.py using opennsfw2 and TensorFlow; enabled via --nsfw-filter.

Models and assets

- Models are expected under modules/models (auto-downloaded where possible):
  - GFPGANv1.4.pth is auto-downloaded on first run of face_enhancer.
  - InSwapper ONNX: inswapper_128.onnx (preferred) or inswapper_128_fp16.onnx. If neither exists, face swapper will error and print a clear status message. Place either file in modules/models.
  - Optional BiSeNet ONNX for semantic parsing: place a supported file (e.g., resnet34.onnx, face_parsing_bisenet_19.onnx) in modules/models or set env vars:
    - DLC_BISENET_ONNX_PATH: absolute path to a local ONNX model
    - DLC_BISENET_ONNX_URL: URL to download if not present

Operational notes

- Temporary work area: For videos, frames are extracted to temp/<target_basename>. Use --keep-frames to retain; otherwise they are removed after processing. Outputs are written next to the target when appropriate; if you pass a directory as -o/--output
- Performance: Control threading with --execution-threads. Provider selection heavily influences speed: prefer cuda on NVIDIA, dml on Windows for broad GPU support, cpu as fallback.
- Live camera: Start the GUI (python run.py), choose a camera from the dropdown, and click Live. Use UI toggles for mirror/resize and processor enabling (face enhancer switch).

What’s not present

- No build/package step (this is a Python application run directly from sources).
- No test suite exists in the repo at this time.
