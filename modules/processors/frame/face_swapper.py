import os  # <-- Added for os.path.exists
from typing import Any, List
import cv2
import insightface
import threading
import numpy as np

import modules.globals
import modules.processors.frame.core
# Ensure update_status is imported if not already globally accessible
# If it's part of modules.core, it might already be accessible via modules.core.update_status
from modules.core import update_status
from modules.face_analyser import get_one_face, get_many_faces, default_source_face
from modules.typing import Face, Frame
from modules.utilities import conditional_download, resolve_relative_path, is_image, is_video
from modules.cluster_analysis import find_closest_centroid

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'DLC.FACE-SWAPPER'


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    # Ensure both models are mentioned or downloaded if necessary
    # Conditional download might need adjustment if you want it to fetch FP32 too
    conditional_download(download_directory_path, ['https://huggingface.co/hacksider/deep-live-cam/blob/main/inswapper_128_fp16.onnx'])
    # Add a check or download for the FP32 model if you have a URL
    # conditional_download(download_directory_path, ['URL_TO_FP32_MODEL_HERE'])
    return True


def pre_start() -> bool:
    # --- No changes needed in pre_start ---
    if not modules.globals.map_faces and not is_image(modules.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False
    elif not modules.globals.map_faces and not get_one_face(cv2.imread(modules.globals.source_path)):
        update_status('No face in source path detected.', NAME)
        return False
    if not is_image(modules.globals.target_path) and not is_video(modules.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            # --- MODIFICATION START ---
            # Define paths for both FP32 and FP16 models
            model_dir = resolve_relative_path('../models')
            model_path_fp32 = os.path.join(model_dir, 'inswapper_128.onnx')
            model_path_fp16 = os.path.join(model_dir, 'inswapper_128_fp16.onnx')
            chosen_model_path = None

            # Prioritize FP32 model
            if os.path.exists(model_path_fp32):
                chosen_model_path = model_path_fp32
                update_status(f"Loading FP32 model: {os.path.basename(chosen_model_path)}", NAME)
            # Fallback to FP16 model
            elif os.path.exists(model_path_fp16):
                chosen_model_path = model_path_fp16
                update_status(f"FP32 model not found. Loading FP16 model: {os.path.basename(chosen_model_path)}", NAME)
            # Error if neither model is found
            else:
                error_message = f"Face Swapper model not found. Please ensure 'inswapper_128.onnx' (recommended) or 'inswapper_128_fp16.onnx' exists in the '{model_dir}' directory."
                update_status(error_message, NAME)
                raise FileNotFoundError(error_message)

            # Load the chosen model
            try:
                FACE_SWAPPER = insightface.model_zoo.get_model(chosen_model_path, providers=modules.globals.execution_providers)
            except Exception as e:
                update_status(f"Error loading Face Swapper model {os.path.basename(chosen_model_path)}: {e}", NAME)
                # Optionally, re-raise the exception or handle it more gracefully
                raise e
            # --- MODIFICATION END ---
    return FACE_SWAPPER


def _apply_mouth_mask(original_frame: Frame, swapped_frame: Frame, target_face: Face) -> Frame:
    """Blend the original mouth region back onto the swapped frame.

    Uses 5-point landmarks if available; otherwise falls back to a bbox heuristic.
    Respects globals: mouth_mask, show_mouth_mask_box, mask_feather_ratio, mask_down_size, mask_size.
    """
    try:
        h, w = swapped_frame.shape[:2]

        down = float(getattr(modules.globals, 'mask_down_size', 0.5) or 0.5)
        size_scale = float(getattr(modules.globals, 'mask_size', 1.0) or 1.0)
        feather_ratio = float(getattr(modules.globals, 'mask_feather_ratio', 8) or 8)

        # Prepare a working mask canvas (possibly downscaled for performance)
        ds_w = max(1, int(w * down))
        ds_h = max(1, int(h * down))
        mask_small = np.zeros((ds_h, ds_w), dtype=np.float32)

        # Determine mouth geometry
        has_kps = hasattr(target_face, 'kps') and target_face.kps is not None
        if has_kps:
            kps = np.array(target_face.kps, dtype=np.float32)  # shape (5,2)
            # Landmark indices: [left_eye, right_eye, nose, left_mouth, right_mouth]
            lm_left = kps[3]
            lm_right = kps[4]
            mouth_center = (lm_left + lm_right) / 2.0
            mouth_width = float(np.linalg.norm(lm_right - lm_left))
            # Heuristic for mouth height
            mouth_height = mouth_width * 0.6

            cx = float(mouth_center[0])
            cy = float(mouth_center[1])
            ax = max(1.0, (mouth_width * 0.5) * size_scale)
            ay = max(1.0, (mouth_height * 0.5) * size_scale)
        else:
            # Fallback: Use lower part of face bbox
            if hasattr(target_face, 'bbox') and target_face.bbox is not None:
                x1, y1, x2, y2 = map(float, target_face.bbox)
            else:
                x1, y1, x2, y2 = 0.0, 0.0, float(w), float(h)
            cx = (x1 + x2) / 2.0
            cy = y1 + (y2 - y1) * 0.72
            ax = max(1.0, (x2 - x1) * 0.20 * size_scale)
            ay = max(1.0, (y2 - y1) * 0.14 * size_scale)

        # Draw ellipse on downscaled canvas
        cx_ds = int(round(cx * down))
        cy_ds = int(round(cy * down))
        ax_ds = max(1, int(round(ax * down)))
        ay_ds = max(1, int(round(ay * down)))
        cv2.ellipse(mask_small, (cx_ds, cy_ds), (ax_ds, ay_ds), 0, 0, 360, 1.0, -1)

        # Feather edges for seamless blend
        feather = max(1, int(max(ax_ds, ay_ds) / max(feather_ratio, 1.0)))
        if feather % 2 == 0:
            feather += 1
        if feather >= 3:
            mask_small = cv2.GaussianBlur(mask_small, (feather, feather), 0)

        # Upscale mask to full size
        mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_3 = np.repeat(mask[:, :, None], 3, axis=2)

        # Composite: keep original mouth region
        composed = (original_frame.astype(np.float32) * mask_3 +
                    swapped_frame.astype(np.float32) * (1.0 - mask_3))
        composed = np.clip(composed, 0, 255).astype(np.uint8)

        # Optional: visualize mask boundary
        if getattr(modules.globals, 'show_mouth_mask_box', False):
            # Draw ellipse outline on result frame (full-res coords)
            cv2.ellipse(
                composed,
                (int(round(cx)), int(round(cy))),
                (int(round(ax)), int(round(ay))),
                0,
                0,
                360,
                (0, 255, 0),
                2,
            )
        return composed
    except Exception as e:
        update_status(f"Mouth mask failed: {e}", NAME)
        return swapped_frame


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:  # pyright: ignore[reportInvalidTypeForm]
    # --- No changes needed in swap_face ---
    """
    Replace the face in the temp_frame with the target_face, using the source_face as reference.

    Args:
        source_face (Face): The face to be replaced in temp_frame.
        target_face (Face): The face to replace the source_face with.
        temp_frame (Frame): The frame in which the replacement will take place.

    Returns:
        Frame: The frame with the replaced face.
    """
    swapper = get_face_swapper()
    if swapper is None:
        # Handle case where model failed to load
        update_status("Face swapper model not loaded, skipping swap.", NAME)
        return temp_frame

    original_frame = temp_frame.copy()
    swapped = swapper.get(temp_frame, target_face, source_face, paste_back=True)

    # Apply mouth mask if enabled
    if getattr(modules.globals, 'mouth_mask', False):
        swapped = _apply_mouth_mask(original_frame, swapped, target_face)
    return swapped


def process_frame(source_face: Face, temp_frame: Frame) -> Frame: # pyright: ignore[reportInvalidTypeForm]
    """
    Replace the face in the temp_frame with the target_face, using the source_face as reference.

    Args:
        source_face (Face): The face to be replaced in temp_frame.
        temp_frame (Frame): The frame in which the replacement will take place.

    Returns:
        Frame: The frame with the replaced face.
    """
    if modules.globals.many_faces:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
        target_face = get_one_face(temp_frame)
        if target_face:
            temp_frame = swap_face(source_face, target_face, temp_frame)
    return temp_frame


def process_frame_v2(temp_frame: Frame, temp_frame_path: str = "") -> Frame:
    # --- No changes needed in process_frame_v2 ---
    # (Assuming swap_face handles the potential None return from get_face_swapper)
    """
    Process a frame (image or video) by replacing the detected face(s) with the target face(s) as specified in the source-target map.
    
    Args:
        temp_frame (Frame): The frame in which the replacement will take place.
        temp_frame_path (str): The path of the frame being processed, required for video processing.
    
    Returns:
        Frame: The frame with the replaced face(s).
    """
    if is_image(modules.globals.target_path):
        if modules.globals.many_faces:
            source_face = default_source_face()
            for map_entry in modules.globals.source_target_map: # Renamed 'map' to 'map_entry'
                target_face = map_entry['target']['face']
                temp_frame = swap_face(source_face, target_face, temp_frame)

        elif not modules.globals.many_faces:
            for map_entry in modules.globals.source_target_map: # Renamed 'map' to 'map_entry'
                if "source" in map_entry:
                    source_face = map_entry['source']['face']
                    target_face = map_entry['target']['face']
                    temp_frame = swap_face(source_face, target_face, temp_frame)

    elif is_video(modules.globals.target_path):
        if modules.globals.many_faces:
            source_face = default_source_face()
            for map_entry in modules.globals.source_target_map: # Renamed 'map' to 'map_entry'
                target_frame = [f for f in map_entry['target_faces_in_frame'] if f['location'] == temp_frame_path]

                for frame in target_frame:
                    for target_face in frame['faces']:
                        temp_frame = swap_face(source_face, target_face, temp_frame)

        elif not modules.globals.many_faces:
            for map_entry in modules.globals.source_target_map: # Renamed 'map' to 'map_entry'
                if "source" in map_entry:
                    target_frame = [f for f in map_entry['target_faces_in_frame'] if f['location'] == temp_frame_path]
                    source_face = map_entry['source']['face']

                    for frame in target_frame:
                        for target_face in frame['faces']:
                            temp_frame = swap_face(source_face, target_face, temp_frame)
    else: # Fallback for neither image nor video (e.g., live feed?)
        detected_faces = get_many_faces(temp_frame)
        if modules.globals.many_faces:
            if detected_faces:
                source_face = default_source_face()
                for target_face in detected_faces:
                    temp_frame = swap_face(source_face, target_face, temp_frame)

        elif not modules.globals.many_faces:
            if detected_faces and hasattr(modules.globals, 'simple_map') and modules.globals.simple_map: # Check simple_map exists
                if len(detected_faces) <= len(modules.globals.simple_map['target_embeddings']):
                    for detected_face in detected_faces:
                        closest_centroid_index, _ = find_closest_centroid(modules.globals.simple_map['target_embeddings'], detected_face.normed_embedding)
                        temp_frame = swap_face(modules.globals.simple_map['source_faces'][closest_centroid_index], detected_face, temp_frame)
                else:
                    detected_faces_centroids = [face.normed_embedding for face in detected_faces]
                    i = 0
                    for target_embedding in modules.globals.simple_map['target_embeddings']:
                        closest_centroid_index, _ = find_closest_centroid(detected_faces_centroids, target_embedding)
                        # Ensure index is valid before accessing detected_faces
                        if closest_centroid_index < len(detected_faces):
                            temp_frame = swap_face(modules.globals.simple_map['source_faces'][i], detected_faces[closest_centroid_index], temp_frame)
                        i += 1
    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], progress: Any = None) -> None:
    # --- No changes needed in process_frames ---
    # Note: Ensure get_one_face is called only once if possible for efficiency if !map_faces
    """
    Process frames by replacing the face(s) in each frame with the target face(s) as specified in the source-target map.

    Args:
        source_path (str): The path of the source image or video.
        temp_frame_paths (List[str]): A list of paths to the frames to be processed.
        progress (Any): Optional progress object to track the progress of the processing.

    Returns:
        None
    """

    source_face = None
    if not modules.globals.map_faces:
        source_img = cv2.imread(source_path)
        if source_img is not None:
            source_face = get_one_face(source_img)
        if source_face is None:
             update_status(f"Could not find face in source image: {source_path}, skipping swap.", NAME)
             # If no source face, maybe skip processing? Or handle differently.
             # For now, it will proceed but swap_face might fail later.

    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        if temp_frame is None:
            update_status(f"Warning: Could not read frame {temp_frame_path}", NAME)
            if progress: progress.update(1) # Still update progress even if frame fails
            continue # Skip to next frame

        try:
            if not modules.globals.map_faces:
                if source_face: # Only process if source face was found
                    result = process_frame(source_face, temp_frame)
                else:
                    result = temp_frame # No source face, return original frame
            else:
                 result = process_frame_v2(temp_frame, temp_frame_path)

            cv2.imwrite(temp_frame_path, result)
        except Exception as exception:
            update_status(f"Error processing frame {os.path.basename(temp_frame_path)}: {exception}", NAME)
            # Decide whether to 'pass' (continue processing other frames) or raise
            pass # Continue processing other frames
        finally:
            if progress:
                progress.update(1)


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    # --- No changes needed in process_image ---
    # Note: Added checks for successful image reads and face detection
    """
    Processes an image by enhancing or swapping faces.

    This function attempts to read the target image from the specified path.
    If face mapping is disabled, it reads the source image and attempts to detect
    a face within it. If successful, it processes the target image using the detected
    face from the source image. If face mapping is enabled, the function processes
    the target image directly.

    Args:
        source_path (str): The file path to the source image.
        target_path (str): The file path to the target image.
        output_path (str): The file path where the processed image will be saved.

    Returns:
        None: The processed image is saved to the specified output path.

    Raises:
        None: Does not raise exceptions but logs errors if reading or processing fails.
    """

    target_frame = cv2.imread(target_path) # Read original target for processing
    if target_frame is None:
        update_status(f"Error: Could not read target image: {target_path}", NAME)
        return

    if not modules.globals.map_faces:
        source_img = cv2.imread(source_path)
        if source_img is None:
             update_status(f"Error: Could not read source image: {source_path}", NAME)
             return
        source_face = get_one_face(source_img)
        if source_face is None:
            update_status(f"Error: No face found in source image: {source_path}", NAME)
            return

        result = process_frame(source_face, target_frame)
    else:
        if modules.globals.many_faces:
            update_status('Many faces enabled. Using first source image (if applicable in v2). Processing...', NAME)
        # For process_frame_v2 on single image, it reads the 'output_path' which should be a copy
        # Let's process the 'target_frame' we read instead.
        result = process_frame_v2(target_frame) # Process the frame directly

    # Write the final result to the output path
    success = cv2.imwrite(output_path, result)
    if not success:
        update_status(f"Error: Failed to write output image to: {output_path}", NAME)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    # --- No changes needed in process_video ---
    """
    Process a video frame by frame.

    Args:
        source_path (str): Path to the source video.
        temp_frame_paths (List[str]): Paths to the temporary frames of the video.

    Returns:
        None: The processed video is saved back to the original source path.
    """
    if modules.globals.map_faces and modules.globals.many_faces:
        update_status('Many faces enabled. Using first source image (if applicable in v2). Processing...', NAME)
    # The core processing logic is delegated, which is good.
    modules.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)


STREAM_SOURCE_FACE = None


def process_frame_stream(source_path: str, frame: Frame) -> Frame:
    """
    Process a frame from a video stream.

    This function is intended to be used as a callback for video stream processing.
    It will read the source image if it has not already been read, and then use it to
    process the frame. If the source image is not provided, or could not be read, it
    will return the original frame.

    Args:
        source_path (str): Path to the source image.
        frame (Frame): The frame to be processed.

    Returns:
        Frame: The processed frame.
    """
    global STREAM_SOURCE_FACE
    if not modules.globals.map_faces:
        if STREAM_SOURCE_FACE is None:
            source_img = cv2.imread(source_path)
            if source_img is not None:
                STREAM_SOURCE_FACE = get_one_face(source_img)
        if STREAM_SOURCE_FACE is not None:
            return process_frame(STREAM_SOURCE_FACE, frame)
        return frame
    else:
        return process_frame_v2(frame)
