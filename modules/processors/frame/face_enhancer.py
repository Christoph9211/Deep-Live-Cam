from typing import Any, List, Optional
import cv2
import threading
import gfpgan
import os

import modules.globals
import modules.processors.frame.core
from modules.core import update_status
from modules.face_analyser import get_one_face
from modules.typing import Frame, Face
import platform
import torch
from modules.utilities import (
    conditional_download,
    is_image,
    is_video,
)

FACE_ENHANCER = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
NAME = "DLC.FACE-ENHANCER"

abs_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(abs_dir))), "models"
)


def pre_check() -> bool:
    download_directory_path = models_dir
    conditional_download(
        download_directory_path,
        [
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
        ],
    )
    return True


def pre_start() -> bool:
    """
    Check if the target path is an image or video. If not, update status to inform the user to select an image or video for the target path.
    Returns:
        bool: True if the target path is an image or video, False otherwise.
    """
    if not is_image(modules.globals.target_path) and not is_video(
        modules.globals.target_path
    ):
        update_status("Select an image or video for target path.", NAME)
        return False
    return True


def get_face_enhancer() -> Any:
    """
    Returns a singleton instance of the face enhancer model.

    The face enhancer model is lazily loaded when this function is first called.
    It is loaded from the models directory, which is downloaded from the official
    GFPGAN repository if it does not already exist.

    The face enhancer model is an instance of the GFPGANer class from the
    gfpgan module.

    Returns:
        Any: A singleton instance of the face enhancer model.
    """
    global FACE_ENHANCER

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = os.path.join(models_dir, "GFPGANv1.4.pth")

            match platform.system():
                case "Darwin":  # Mac OS
                    if torch.backends.mps.is_available():
                        mps_device = torch.device("mps")
                        FACE_ENHANCER = gfpgan.GFPGANer(model_path=model_path, upscale=1, device=mps_device)  # type: ignore[attr-defined]
                    else:
                        FACE_ENHANCER = gfpgan.GFPGANer(model_path=model_path, upscale=1)  # type: ignore[attr-defined]
                case _:  # Other OS
                    FACE_ENHANCER = gfpgan.GFPGANer(model_path=model_path, upscale=1)  # type: ignore[attr-defined]

    return FACE_ENHANCER


def enhance_face(temp_frame: Frame) -> Frame:
    """Enhance a face in a frame using GFPGAN.

    Args:
        temp_frame (Frame): The frame to enhance.

    Returns:
        Frame: The enhanced frame.
    """
    with THREAD_SEMAPHORE:
        _, _, temp_frame = get_face_enhancer().enhance(temp_frame, paste_back=True)
    return temp_frame


def process_frame(source_face: Face, temp_frame: Frame) -> Frame:
    """Enhance a face in a frame using GFPGAN.

    Args:
        source_face (Face): The source face to enhance.
        temp_frame (Frame): The frame to enhance.

    Returns:
        Frame: The enhanced frame.
    """
    target_face = get_one_face(temp_frame)
    if target_face is not None:
        temp_frame = enhance_face(temp_frame)
    return temp_frame


def process_frames(
    temp_frame_paths: List[str], progress: Any = None
) -> None:
    """
    Enhance faces in a list of frames using GFPGAN.

    Args:
        source_path (str): The source path of the video or image.
        temp_frame_paths (List[str]): A list of paths to the frames to be processed.
        progress (Any, optional): A progress object to update progress during
            processing. Defaults to None.

    Returns:
        None
    """

    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(None, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if progress:
            progress.update(1)


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """
    Enhance a face in a target image using GFPGAN and save the result.

    Args:
        source_path (str): The source path of the image.
        target_path (str): The path to the target image to process.
        output_path (str): The path where the enhanced image will be saved.

    Returns:
        None
    """

    target_frame = cv2.imread(target_path)
    result = process_frame(None, target_frame)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """
    Process a video by enhancing faces in its frames using GFPGAN.

    Args:
        source_path (str): The source path of the video.
        temp_frame_paths (List[str]): A list of paths to the frames to be processed.

    Returns:
        None: The processed video frames are enhanced and saved back to the original paths.
    """

    modules.processors.frame.core.process_video(None, temp_frame_paths, process_frames)


def process_frame_v2(temp_frame: Frame) -> Frame:
    """
    Enhance a face in a given frame using GFPGAN.

    Args:
        temp_frame (Frame): The frame containing the face to enhance.

    Returns:
        Frame: The frame with the enhanced face.
    """
    target_face = get_one_face(temp_frame)
    if target_face:
        temp_frame = enhance_face(temp_frame)
    return temp_frame


def process_frame_stream(source_path: str, frame: Frame) -> Frame:
    """
    Process a frame from a video stream by enhancing a face in it using GFPGAN.

    Args:
        source_path (str): The path of the source image or video.
        frame (Frame): The frame to be processed.

    Returns:
        Frame: The frame with the enhanced face.
    """
    return process_frame(None, frame)
