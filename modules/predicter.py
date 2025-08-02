import numpy
import opennsfw2
from PIL import Image
import cv2  # Add OpenCV import
import modules.globals  # Import globals to access the color correction toggle

from modules.typing import Frame

MAX_PROBABILITY = 1.0

# Preload the model once for efficiency
model = None

def predict_frame(target_frame: Frame) -> bool:
    # Convert the frame to RGB before processing if color correction is enabled
    """
    Predict whether a given frame contains sensitive content using an NSFW model.

    Args:
        target_frame (Frame): The frame to be evaluated.

    Returns:
        bool: True if the frame's probability of containing sensitive content 
              exceeds the defined threshold, False otherwise.
    """

    if modules.globals.color_correction:
        target_frame = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)
        
    image = Image.fromarray(target_frame)
    image = opennsfw2.preprocess_image(image, opennsfw2.Preprocessing.YAHOO)
    global model
    if model is None: 
        model = opennsfw2.make_open_nsfw_model()
        
    views = numpy.expand_dims(image, axis=0)
    _, probability = model.predict(views)[0]
    return probability > MAX_PROBABILITY


def predict_image(target_path: str) -> bool:
    """
    Predict whether an image at the given path contains sensitive content using an NSFW model.

    Args:
        target_path (str): The path to the image file to be evaluated.

    Returns:
        bool: True if the image's probability of containing sensitive content 
              exceeds the defined threshold, False otherwise.
    """

    return opennsfw2.predict_image(target_path) > MAX_PROBABILITY


def predict_video(target_path: str) -> bool:
    """
    Predict whether a video at the given path contains sensitive content using an NSFW model.

    Args:
        target_path (str): The path to the video file to be evaluated.

    Returns:
        bool: True if any frame in the video has a probability of containing sensitive content 
              that exceeds the defined threshold, False otherwise.
    """

    _, probabilities = opennsfw2.predict_video_frames(video_path=target_path, frame_interval=100)
    return any(probability > MAX_PROBABILITY for probability in probabilities)
