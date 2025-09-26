# High-Impact, Low-Effort Face Swap Quality Improvements

This document collects implementation ideas that can noticeably improve swap quality without requiring large architectural changes.

## 1. Improve Face Selection and Tracking
- **Deterministic best-face selection:** Cache the face embeddings produced in `modules/face_analyser.py` and prefer the source face whose pose/lighting is closest to the current target frame, instead of always using the first embedding. This primarily requires minor changes in the analyser to expose pose metadata and a simple cosine-distance ranking.
- **Stabilize landmark jitter:** Enable the existing One-Euro smoothing parameters from `modules/core.py` by default for video pipelines, and expose presets in the UI so users can toggle them quickly. This is a configuration/UI update that can immediately reduce flicker.
- **Automatic face-ID re-association:** Track detected faces across frames using bounding box IoU + embedding similarity in `modules/processors/frame/core` so the same source face stays attached to the same target identity even when multiple faces are present.

## 2. Sharpen Masks and Blending
- **Feathered segmentation masks:** Leverage the existing semantic segmenters (`modules/segmenters`) to expand the swap mask to include forehead and blend it with a Gaussian feather. Implementing this only needs mask post-processing before blending.
- **Adaptive mask dilation near the mouth:** Use the `--preserve-teeth` flag logic to keep the inner mouth untouched unless confidence is high; this is a small conditional expansion around the current mask handling.
- **Optional Poisson blending path:** Integrate OpenCV's `seamlessClone` (already available via `cv2`) as a toggle to blend colors when lighting differences are large. This is a short code path guarded by a flag.

## 3. Automatic Color and Tone Matching
- **Histogram matching between source and target skin regions:** Sample the target skin using the segmentation mask and apply `cv2.createCLAHE` / histogram matching before compositing. This only requires a few lines in the blending stage and greatly reduces hue mismatches.
- **Dynamic gamma/white balance correction:** Compute average lab color difference between the source face and surrounding target area, then adjust via `cv2.cvtColor` + scale factors. This can be implemented as a utility function reused by processors.

## 4. Quality Assurance Hooks
- **Confidence-based frame skipping:** When face detection confidence in `modules/face_analyser.py` is below a threshold, keep the original frame to avoid obvious artifacts. It's a minor if-statement using the existing detector scores.
- **Debug visualization overlay:** Add a `--debug-overlay` flag that draws the swap mask and blend boundaries onto a copy of the frame, making it easier to tune mask parameters without major refactors.

## 5. Performance Tweaks That Unlock Quality Options
- **Auto-downscale for detection, full-res for blending:** Detect faces on a resized frame but perform warping on the original resolution by re-using the transformation matrices. This keeps quality high while keeping detection fast.
- **Batch processing for still-image targets:** When the target is an image sequence, stack faces detected in consecutive frames and process them in a batch to reuse GPU calls from the face swapper without changing its API.

Each idea above relies on modules already in the repository and focuses on small, incremental changes that can substantially improve the visual fidelity of swaps.
