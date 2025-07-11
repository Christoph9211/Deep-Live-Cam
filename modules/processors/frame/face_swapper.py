import os  # <-- Added for os.path.exists
from typing import Any, List
from pathlib import Path
import cv2
import numpy as np
import insightface
import threading
import mediapipe as mp

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

SEGMENTER = None


from typing import Tuple


from typing import Optional

def get_combined_source_face(source_path: str) -> Optional[Face]: # type: ignore
    """Load the source face and optionally merge with an open-mouth still."""
    source_img = cv2.imread(source_path)
    if source_img is None:
        return None
    source_face = get_one_face(source_img)
    open_path = modules.globals.open_mouth_source_path
    if source_face is not None and open_path and is_image(open_path):
        open_img = cv2.imread(open_path)
        if open_img is not None:
            open_face = get_one_face(open_img)
            if open_face is not None and hasattr(source_face, "embedding") and hasattr(open_face, "embedding"):
                combined = source_face.embedding + open_face.embedding
                norm = np.linalg.norm(combined)
                if norm != 0:
                    combined = combined / norm
                source_face.embedding = combined
    return source_face

# def create_lower_mouth_mask(
#     face: Face, frame: Frame # type: ignore
# ) -> Tuple[np.ndarray, np.ndarray, tuple, np.ndarray]:
#     """Create a mask tightly around the mouth area, including only the upper lip (not the nose)."""

#     # Create a zero-filled mask with the same height and width as the frame
#     mask = np.zeros(frame.shape[:2], dtype=np.uint8)
#     # Initialize the mouth cutout, which will be populated if the face has landmarks
#     mouth_cutout = None
#     # Get the 106-point landmark array from the face
#     landmarks = face.landmark_2d_106

#     if landmarks is not None:
#         # Restrict indices to outer lip and immediate upper lip, avoiding nose
#         mouth_indices =  [
#             65,
#             66,
#             62,
#             70,
#             69,
#             18,
#             19,
#             20,
#             21,
#             22,
#             23,
#             24,
#             0,
#             8,
#             7,
#             6,
#             5,
#             4,
#             3,
#             2,
#             65,
#         ]
#         # mouth_indices = list(range(52, 72))
#         # mouth_indices = [
#         #     65, 66, 62, 70, 69, # upper lip corners
#         #     60, 61, 63, 64, 71, 72, 73, 74, 75, # just above upper lip
#         #     8, 7, 6, 5, 4, 3, 2, # lower lip
#         #     0 # left mouth corner
#         # ]
#         # Only use points that form the upper lip and immediate area above
#         mouth_points = landmarks[mouth_indices].astype(np.int32)
#         # Compute the convex hull of the mouth points
#         hull = cv2.convexHull(mouth_points)
#         # Get the bounding box of the convex hull
#         x, y, w, h = cv2.boundingRect(hull)

#         # Clip coordinates to ensure we do not index outside of the frame
#         frame_height, frame_width = frame.shape[:2]
#         min_x, min_y = x, y
#         max_x, max_y = x + w, y + h
#         clip_min_x = max(min_x, 0)
#         clip_min_y = max(min_y, 0)
#         clip_max_x = min(max_x, frame_width)
#         clip_max_y = min(max_y, frame_height)

#         # Create a zero-filled mask ROI with the same shape as the bounding box
#         mask_roi_full = np.zeros((h, w), dtype=np.uint8)
#         # Shift the convex hull so that the origin is at the top-left of the bounding box
#         shifted_hull = hull - [x, y]
#         # Fill the convex hull in the mask ROI with white (255)
#         cv2.fillConvexPoly(mask_roi_full, shifted_hull, 255)

#         # Optional: very slight blur for feathering, but not enough to extend past mouth
#         mask_roi_full = cv2.GaussianBlur(mask_roi_full, (5, 5), 1)

#         # Crop ROI if clipping occurred
#         mask_roi = mask_roi_full[clip_min_y - y : clip_max_y - y, clip_min_x - x : clip_max_x - x]

#         # Copy the mask ROI into the output mask
#         mask[clip_min_y:clip_max_y, clip_min_x:clip_max_x] = mask_roi

#         # Extract the mouth cutout from the original frame
#         mouth_cutout = frame[clip_min_y:clip_max_y, clip_min_x:clip_max_x].copy()
#         # Save the lower lip polygon (the convex hull of the mouth points)
#         lower_lip_polygon = hull
#         # Return the mask, mouth cutout, bounding box coordinates, and lower lip polygon
#         return mask, mouth_cutout, (clip_min_x, clip_min_y, clip_max_x, clip_max_y), lower_lip_polygon

#     # If the face does not have landmarks, return an empty mask and None for the mouth cutout
#     return mask, mouth_cutout, (0, 0, 0, 0), None


# # def draw_mouth_mask_visualization(
# #     frame: Frame, face: Face, mouth_mask_data: tuple # type: ignore
# # ) -> Frame:
# #     """Visualize the expanded mouth mask and the lower lip polygon.

# #     This function takes a frame and a face as input and returns a new frame with
# #     visualizations of the expanded mouth mask and the lower lip polygon.

# #     Args:
# #         frame: The original frame.
# #         face: The face object containing the landmarks.
# #         mouth_mask_data: A tuple containing the mouth mask, the mouth cutout, the
# #             bounding box coordinates, and the lower lip polygon.

# #     Returns:
# #         A new frame with visualizations of the expanded mouth mask and the lower
# #             lip polygon.
# #     """

# #     # Get the landmarks from the face object
# #     landmarks = face.landmark_2d_106

# #     # If the landmarks are not None, proceed with visualization
# #     if landmarks is not None and mouth_mask_data is not None:
# #         # Unpack the mouth mask data tuple
# #         mask, mouth_cutout, (min_x, min_y, max_x, max_y), lower_lip_polygon = mouth_mask_data

# #         # Create a copy of the original frame for visualization
# #         vis_frame = frame.copy()

# #         # Get the height and width of the frame
# #         height, width = vis_frame.shape[:2]

# #         # Clip the coordinates to ensure they do not exceed the frame boundaries
# #         min_x, min_y = max(0, min_x), max(0, min_y)
# #         max_x, max_y = min(width, max_x), min(height, max_y)

# #         # --- Expanded region: include chin, lower cheeks, and more of the upper lip ---

# #         # Define the indices for the expanded mouth region
# #         # These indices include the chin, lower cheek area, and more of the upper lip
# #         chin_indices = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 0, 1, 2, 3]
# #         mouth_indices = [
# #             65, 66, 62, 70, 69, 18, 19, 20, 21, 22, 23, 24, 0, 8, 7, 6, 5, 4, 3, 2,
# #             60, 61, 63, 64, 71, 72, 73, 74, 75  # points above upper lip (adjust as needed)
# #         ]
# #         upper_lip_indices = [76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]

# #         # Concatenate the expanded points
# #         expanded_points = np.concatenate([
# #             landmarks[mouth_indices],
# #             landmarks[chin_indices][::-1],
# #             landmarks[upper_lip_indices][::-1]
# #         ]).astype(np.int32)

# #         # Compute the convex hull of the expanded points
# #         expanded_hull = cv2.convexHull(expanded_points)

# #         # Draw the expanded hull as a yellow polygon
# #         cv2.polylines(vis_frame, [expanded_hull], True, (0, 128, 255), 2)

# #         # Draw the lower lip polygon as a green polygon
# #         cv2.polylines(vis_frame, [lower_lip_polygon], True, (0, 255, 0), 2)

# #         # Draw a filled expanded mask (semi-transparent overlay)
# #         overlay = vis_frame.copy()
# #         cv2.fillConvexPoly(overlay, expanded_hull, (0, 128, 255))
# #         alpha = 0.25
# #         vis_frame = cv2.addWeighted(overlay, alpha, vis_frame, 1 - alpha, 0)

# #         # Feathered mask for the expanded region
# #         mask_region = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
# #         shifted_hull = expanded_hull - [min_x, min_y]
# #         cv2.fillConvexPoly(mask_region, shifted_hull, 255)
# #         feather_amount = max(
# #             5,
# #             min(40, (max_x - min_x) // 8, (max_y - min_y) // 8),
# #         )
# #         kernel_size = 2 * feather_amount + 1
# #         feathered_mask = cv2.GaussianBlur(mask_region.astype(float), (kernel_size, kernel_size), 0)
# #         feathered_mask = (feathered_mask / feathered_mask.max() * 255).astype(np.uint8)
# #         # Overlay feathered mask as heatmap
# #         color_mask = cv2.applyColorMap(feathered_mask, cv2.COLORMAP_JET)
# #         mask_area = vis_frame[min_y:max_y, min_x:max_x]
# #         vis_frame[min_y:max_y, min_x:max_x] = cv2.addWeighted(mask_area, 0.7, color_mask, 0.3, 0)

# #         # Add a text label to the visualization
# #         cv2.putText(
# #             vis_frame,
# #             "Expanded Mouth/Chin Mask",
# #             (min_x, min_y - 10),
# #             cv2.FONT_HERSHEY_SIMPLEX,
# #             0.5,
# #             (255, 255, 255),
# #             1,
# #         )

# #         # Return the visualization
# #         return vis_frame

# #     # If the landmarks are None, return the original frame
# #     return frame

# def draw_mouth_mask_visualization(
#     frame: Frame, face: Face, mouth_mask_data: tuple # type: ignore
# ) -> Frame:
#     landmarks = face.landmark_2d_106
#     if landmarks is not None and mouth_mask_data is not None:
#         mask, mouth_cutout, (min_x, min_y, max_x, max_y), lower_lip_polygon = (
#             mouth_mask_data
#         )

#         vis_frame = frame.copy()

#         # Ensure coordinates are within frame bounds
#         height, width = vis_frame.shape[:2]
#         min_x, min_y = max(0, min_x), max(0, min_y)
#         max_x, max_y = min(width, max_x), min(height, max_y)

#         # Adjust mask to match the region size
#         mask_region = mask[0 : max_y - min_y, 0 : max_x - min_x]

#         # vis_region = vis_frame[min_y:max_y, min_x:max_x]

#         # Draw the lower lip polygon
#         cv2.polylines(vis_frame, [lower_lip_polygon], True, (0, 255, 0), 2)

#         feather_amount = max(
#             1,
#             min(
#                 30,
#                 (max_x - min_x) // modules.globals.mask_feather_ratio,
#                 (max_y - min_y) // modules.globals.mask_feather_ratio,
#             ),
#         )
#         kernel_size = 2 * feather_amount + 1
#         feathered_mask = cv2.GaussianBlur(
#             mask_region.astype(float), (kernel_size, kernel_size), 0
#         )
#         feathered_mask = (feathered_mask / feathered_mask.max() * 255).astype(np.uint8)

#         cv2.putText(
#             vis_frame,
#             "Lower Mouth Mask",
#             (min_x, min_y - 10),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.5,
#             (255, 255, 255),
#             1,
#         )
#         cv2.putText(
#             vis_frame,
#             "Feathered Mask",
#             (min_x, max_y + 20),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.5,
#             (255, 255, 255),
#             1,
#         )

#         return vis_frame
#     return frame

# def apply_mouth_area(
#     frame: np.ndarray,
#     mouth_cutout: np.ndarray,
#     mouth_box: tuple,
#     face_mask: np.ndarray,
#     mouth_polygon: np.ndarray,
# ) -> np.ndarray:
#     min_x, min_y, max_x, max_y = mouth_box
#     box_width = max_x - min_x
#     box_height = max_y - min_y

#     if (
#         mouth_cutout is None
#         or box_width is None
#         or box_height is None
#         or face_mask is None
#         or mouth_polygon is None
#     ):
#         return frame

#     try:
#         resized_mouth_cutout = cv2.resize(mouth_cutout, (box_width, box_height))
#         roi = frame[min_y:max_y, min_x:max_x]

#         if roi.shape != resized_mouth_cutout.shape:
#             resized_mouth_cutout = cv2.resize(
#                 resized_mouth_cutout, (roi.shape[1], roi.shape[0])
#             )

#         color_corrected_mouth = apply_color_transfer(resized_mouth_cutout, roi)

#         polygon_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
#         adjusted_polygon = mouth_polygon - [min_x, min_y]
#         cv2.fillPoly(polygon_mask, [adjusted_polygon], 255)

#         feather_amount = min(
#             30,
#             box_width // modules.globals.mask_feather_ratio,
#             box_height // modules.globals.mask_feather_ratio,
#         )
#         feathered_mask = cv2.GaussianBlur(
#             polygon_mask.astype(float), (0, 0), feather_amount
#         )
#         feathered_mask = feathered_mask / feathered_mask.max()

#         face_mask_roi = face_mask[min_y:max_y, min_x:max_x]
#         combined_mask = feathered_mask * (face_mask_roi / 255.0)

#         combined_mask = combined_mask[:, :, np.newaxis]
#         blended = (
#             color_corrected_mouth * combined_mask + roi * (1 - combined_mask)
#         ).astype(np.uint8)

#         face_mask_3channel = (
#             np.repeat(face_mask_roi[:, :, np.newaxis], 3, axis=2) / 255.0
#         )
#         final_blend = blended * face_mask_3channel + roi * (1 - face_mask_3channel)

#         frame[min_y:max_y, min_x:max_x] = final_blend.astype(np.uint8)
#     except Exception as e:
#         pass

#     return frame

def create_lower_mouth_mask(
    face: Face, frame: Frame # type: ignore
) -> Tuple[np.ndarray, np.ndarray, tuple, np.ndarray]:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mouth_cutout = None
    landmarks = face.landmark_2d_106
    if landmarks is not None:
        #                  0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20
        lower_lip_order = [
            65,
            66,
            62,
            70,
            69,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            0,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            65,
        ]
        lower_lip_landmarks = landmarks[lower_lip_order].astype(
            np.float32
        )  # Use float for precise calculations

        # Calculate the center of the landmarks
        center = np.mean(lower_lip_landmarks, axis=0)

        # Expand the landmarks outward
        expansion_factor = (
            1 + modules.globals.mask_down_size
        )  # Adjust this for more or less expansion
        expanded_landmarks = (lower_lip_landmarks - center) * expansion_factor + center

        # Extend the top lip part
        toplip_indices = [
            20,
            0,
            1,
            2,
            3,
            4,
            5,
        ]  # Indices for landmarks 2, 65, 66, 62, 70, 69, 18
        toplip_extension = (
            modules.globals.mask_size * 0.5
        )  # Adjust this factor to control the extension
        for idx in toplip_indices:
            direction = expanded_landmarks[idx] - center
            direction = direction / np.linalg.norm(direction)
            expanded_landmarks[idx] += direction * toplip_extension

        # Extend the bottom part (chin area)
        chin_indices = [
            11,
            12,
            13,
            14,
            15,
            16,
        ]  # Indices for landmarks 21, 22, 23, 24, 0, 8
        chin_extension = 2 * 0.0  # Adjust this factor to control the extension
        for idx in chin_indices:
            expanded_landmarks[idx][1] += (
                expanded_landmarks[idx][1] - center[1]
            ) * chin_extension

        # Convert back to integer coordinates
        expanded_landmarks = expanded_landmarks.astype(np.int32)

        # Calculate bounding box for the expanded lower mouth
        min_x, min_y = np.min(expanded_landmarks, axis=0)
        max_x, max_y = np.max(expanded_landmarks, axis=0)

        # Add some padding to the bounding box
        padding = int((max_x - min_x) * 0.1)  # 10% padding
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(frame.shape[1], max_x + padding)
        max_y = min(frame.shape[0], max_y + padding)

        # Ensure the bounding box dimensions are valid
        if max_x <= min_x or max_y <= min_y:
            if (max_x - min_x) <= 1:
                max_x = min_x + 1
            if (max_y - min_y) <= 1:
                max_y = min_y + 1

        # Create the mask
        mask_roi = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
        cv2.fillPoly(mask_roi, [expanded_landmarks - [min_x, min_y]], 255)

        # Apply Gaussian blur to soften the mask edges
        mask_roi = cv2.GaussianBlur(mask_roi, (15, 15), 5)

        # Place the mask ROI in the full-sized mask
        mask[min_y:max_y, min_x:max_x] = mask_roi

        # Extract the masked area from the frame
        mouth_cutout = frame[min_y:max_y, min_x:max_x].copy()

        # Return the expanded lower lip polygon in original frame coordinates
        lower_lip_polygon = expanded_landmarks

    return mask, mouth_cutout, (min_x, min_y, max_x, max_y), lower_lip_polygon


def draw_mouth_mask_visualization(
    frame: Frame, face: Face, mouth_mask_data: tuple # type: ignore
) -> Frame:
    landmarks = face.landmark_2d_106
    if landmarks is not None and mouth_mask_data is not None:
        mask, mouth_cutout, (min_x, min_y, max_x, max_y), lower_lip_polygon = (
            mouth_mask_data
        )

        vis_frame = frame.copy()

        # Ensure coordinates are within frame bounds
        height, width = vis_frame.shape[:2]
        min_x, min_y = max(0, min_x), max(0, min_y)
        max_x, max_y = min(width, max_x), min(height, max_y)

        # Adjust mask to match the region size
        mask_region = mask[0 : max_y - min_y, 0 : max_x - min_x]

        # Remove the color mask overlay
        # color_mask = cv2.applyColorMap((mask_region * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Ensure shapes match before blending
        vis_region = vis_frame[min_y:max_y, min_x:max_x]
        # Remove blending with color_mask
        # if vis_region.shape[:2] == color_mask.shape[:2]:
        #     blended = cv2.addWeighted(vis_region, 0.7, color_mask, 0.3, 0)
        #     vis_frame[min_y:max_y, min_x:max_x] = blended

        # Draw the lower lip polygon
        cv2.polylines(vis_frame, [lower_lip_polygon], True, (0, 255, 0), 2)

        # Remove the red box
        # cv2.rectangle(vis_frame, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)

        # Visualize the feathered mask
        feather_amount = max(
            1,
            min(
                30,
                (max_x - min_x) // modules.globals.mask_feather_ratio,
                (max_y - min_y) // modules.globals.mask_feather_ratio,
            ),
        )
        # Ensure kernel size is odd
        kernel_size = 2 * feather_amount + 1
        feathered_mask = cv2.GaussianBlur(
            mask_region.astype(float), (kernel_size, kernel_size), 0
        )
        max_val = feathered_mask.max()
        if max_val > 0:
            feathered_mask = (feathered_mask / max_val * 255).astype(np.uint8)
        else:
            feathered_mask = np.zeros_like(feathered_mask, dtype=np.uint8)
        # Remove the feathered mask color overlay
        # color_feathered_mask = cv2.applyColorMap(feathered_mask, cv2.COLORMAP_VIRIDIS)

        # Ensure shapes match before blending feathered mask
        # if vis_region.shape == color_feathered_mask.shape:
        #     blended_feathered = cv2.addWeighted(vis_region, 0.7, color_feathered_mask, 0.3, 0)
        #     vis_frame[min_y:max_y, min_x:max_x] = blended_feathered

        # Add labels
        cv2.putText(
            vis_frame,
            "Lower Mouth Mask",
            (min_x, min_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            vis_frame,
            "Feathered Mask",
            (min_x, max_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        return vis_frame
    return frame


def apply_mouth_area(
    frame: np.ndarray,
    mouth_cutout: np.ndarray,
    mouth_box: tuple,
    face_mask: np.ndarray,
    mouth_polygon: np.ndarray,
) -> np.ndarray:
    min_x, min_y, max_x, max_y = mouth_box
    box_width = max_x - min_x
    box_height = max_y - min_y

    if (
        mouth_cutout is None
        or box_width is None
        or box_height is None
        or face_mask is None
        or mouth_polygon is None
    ):
        return frame

    try:
        resized_mouth_cutout = cv2.resize(mouth_cutout, (box_width, box_height))
        roi = frame[min_y:max_y, min_x:max_x]

        if roi.shape != resized_mouth_cutout.shape:
            resized_mouth_cutout = cv2.resize(
                resized_mouth_cutout, (roi.shape[1], roi.shape[0])
            )

        color_corrected_mouth = apply_color_transfer(resized_mouth_cutout, roi)

        # Use the provided mouth polygon to create the mask
        polygon_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        adjusted_polygon = mouth_polygon - [min_x, min_y]
        cv2.fillPoly(polygon_mask, [adjusted_polygon], 255)

        # Apply feathering to the polygon mask
        feather_amount = min(
            30,
            box_width // modules.globals.mask_feather_ratio,
            box_height // modules.globals.mask_feather_ratio,
        )
        feathered_mask = cv2.GaussianBlur(
            polygon_mask.astype(float), (0, 0), feather_amount
        )
        feathered_mask = feathered_mask / feathered_mask.max()

        face_mask_roi = face_mask[min_y:max_y, min_x:max_x]
        combined_mask = feathered_mask * (face_mask_roi / 255.0)

        combined_mask = combined_mask[:, :, np.newaxis]
        blended = (
            color_corrected_mouth * combined_mask + roi * (1 - combined_mask)
        ).astype(np.uint8)

        # Apply face mask to blended result
        face_mask_3channel = (
            np.repeat(face_mask_roi[:, :, np.newaxis], 3, axis=2) / 255.0
        )
        final_blend = blended * face_mask_3channel + roi * (1 - face_mask_3channel)

        frame[min_y:max_y, min_x:max_x] = final_blend.astype(np.uint8)
    except Exception as e:
        pass

    return frame

def create_face_mask(face: Face, frame: Frame) -> np.ndarray: # type: ignore
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    landmarks = face.landmark_2d_106
    if landmarks is not None:
        landmarks = landmarks.astype(np.int32)

        right_side_face = landmarks[0:16]
        left_side_face = landmarks[17:32]
        # right_eye = landmarks[33:42]
        right_eye_brow = landmarks[43:51]
        # left_eye = landmarks[87:96]
        left_eye_brow = landmarks[97:105]

        right_eyebrow_top = np.min(right_eye_brow[:, 1])
        left_eyebrow_top = np.min(left_eye_brow[:, 1])
        eyebrow_top = min(right_eyebrow_top, left_eyebrow_top)

        face_top = np.min([right_side_face[0, 1], left_side_face[-1, 1]])
        forehead_height = face_top - eyebrow_top
        extended_forehead_height = int(forehead_height * 5.0)

        forehead_left = right_side_face[0].copy()
        forehead_right = left_side_face[-1].copy()
        forehead_left[1] -= extended_forehead_height
        forehead_right[1] -= extended_forehead_height

        face_outline = np.vstack(
            [
                [forehead_left],
                right_side_face,
                left_side_face[::-1],
                [forehead_right],
            ]
        )

        padding = int(
            np.linalg.norm(right_side_face[0] - left_side_face[-1]) * 0.05
        )

        hull = cv2.convexHull(face_outline)
        hull_padded = []
        for point in hull:
            x, y = point[0]
            center = np.mean(face_outline, axis=0)
            direction = np.array([x, y]) - center
            direction = direction / np.linalg.norm(direction)
            padded_point = np.array([x, y]) + direction * padding
            hull_padded.append(padded_point)

        hull_padded = np.array(hull_padded, dtype=np.int32)

        cv2.fillConvexPoly(mask, hull_padded, 255)

        mask = cv2.GaussianBlur(mask, (5, 5), 3)

    return mask


def get_foreground_mask(frame: Frame) -> np.ndarray:
    """Return a binary mask for prominent foreground objects with improved sensitivity."""
    global SEGMENTER
    if SEGMENTER is None:
        SEGMENTER = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

    results = SEGMENTER.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.segmentation_mask is None:
        return np.zeros(frame.shape[:2], dtype=np.uint8)

    # Increase sensitivity by adjusting the threshold
    mask = (results.segmentation_mask > 0.3).astype(np.uint8) * 255
    return mask


def apply_color_transfer(source, target):
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    source_mean, source_std = cv2.meanStdDev(source)
    target_mean, target_std = cv2.meanStdDev(target)

    source_mean = source_mean.reshape(1, 1, 3)
    source_std = source_std.reshape(1, 1, 3)
    target_mean = target_mean.reshape(1, 1, 3)
    target_std = target_std.reshape(1, 1, 3)

    # Avoid division by zero by adding epsilon
    epsilon = 1e-6
    safe_source_std = np.where(source_std < epsilon, epsilon, source_std)
    source = (source - source_mean) * (target_std / safe_source_std) + target_mean

    # Remove NaNs/Infs before casting
    source = np.nan_to_num(source, nan=0.0, posinf=255.0, neginf=0.0)
    return cv2.cvtColor(np.clip(source, 0, 255).astype("uint8"), cv2.COLOR_LAB2BGR)


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    # Ensure both models are mentioned or downloaded if necessary
    # Conditional download might need adjustment if you want it to fetch FP32 too
    conditional_download(download_directory_path, ['https://huggingface.co/hacksider/deep-live-cam/blob/main/inswapper_128_fp16.onnx'])
    # Add a check or download for the FP32 model if you have a URL
    # conditional_download(download_directory_path, ['URL_TO_FP32_MODEL_HERE'])
    return True


def pre_start() -> bool:
    global STREAM_SOURCE_FACE, STREAM_FRAME_IDX
    STREAM_SOURCE_FACE = None
    STREAM_FRAME_IDX = 0
    # Reset function-static cache used by ``process_frame_stream`` so that
    # frame indexes start from ``0`` for each new run. Without this the
    # index could carry over from a previous preview or run which results
    # in mismatched frame ids when mapping faces. This caused the mapped
    # faces logic to never find a matching frame and therefore no faces
    # were swapped in the final output.
    if hasattr(process_frame_stream, "cache"):
        process_frame_stream.cache = {
            "source_face": None,
            "frame_idx": 0,
        }
    if not modules.globals.map_faces and not is_image(modules.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False
    elif not modules.globals.map_faces and get_combined_source_face(modules.globals.source_path) is None:
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


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame: # type: ignore
    swapper = get_face_swapper()
    if swapper is None:
        update_status("Face swapper model not loaded, skipping swap.", NAME)
        return temp_frame

    swapped_frame = swapper.get(temp_frame, target_face, source_face, paste_back=True)

    apply_mouth = modules.globals.mouth_mask
    if modules.globals.mouth_mask_segments:
        fps = modules.globals.fps or 30.0
        current_time = modules.globals.current_frame_idx / fps
        apply_mouth = any(start <= current_time <= end for start, end in modules.globals.mouth_mask_segments)

    if apply_mouth:
        face_mask = create_face_mask(target_face, temp_frame)
        mouth_mask, mouth_cutout, mouth_box, lower_lip_polygon = create_lower_mouth_mask(
            target_face, temp_frame
        )

        swapped_frame = apply_mouth_area(
            swapped_frame, mouth_cutout, mouth_box, face_mask, lower_lip_polygon
        )

        if modules.globals.show_mouth_mask_box:
            mouth_mask_data = (mouth_mask, mouth_cutout, mouth_box, lower_lip_polygon)
            swapped_frame = draw_mouth_mask_visualization(
                swapped_frame, target_face, mouth_mask_data
            )

    if modules.globals.foreground_protection:
        # Protect non-face foreground elements using a segmentation mask
        fg_mask = get_foreground_mask(temp_frame)
        face_mask = create_face_mask(target_face, temp_frame)
        occlusion_mask = cv2.bitwise_and(fg_mask, cv2.bitwise_not(face_mask))
        occlusion_mask_3c = cv2.merge([occlusion_mask] * 3) / 255.0
        swapped_frame = (
            swapped_frame * (1 - occlusion_mask_3c) + temp_frame * occlusion_mask_3c
        ).astype(np.uint8)

    return swapped_frame


def process_frame(source_face: Face, temp_frame: Frame) -> Frame:  # type: ignore
    # --- No changes needed in process_frame ---
    # Ensure the frame is in RGB format if color correction is enabled
    # Note: InsightFace swapper often expects BGR by default. Double-check if color issues appear.
    # If color correction is needed *before* swapping and insightface needs BGR:
    # original_was_bgr = True # Assume input is BGR
    # if modules.globals.color_correction:
    #     temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
    #     original_was_bgr = False # Now it's RGB

    if modules.globals.many_faces:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
        target_face = get_one_face(temp_frame)
        if target_face:
            temp_frame = swap_face(source_face, target_face, temp_frame)

    # Convert back if necessary (example, might not be needed depending on workflow)
    # if modules.globals.color_correction and not original_was_bgr:
    #      temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_RGB2BGR)

    return temp_frame


def process_frame_v2(temp_frame: Frame, frame_id: Any = "") -> Frame:
    """Swap faces according to mapping rules.

    When ``map_faces`` is enabled it now takes priority over ``many_faces`` so
    that each source/target pair is honoured even if ``many_faces`` is also
    active.
    """
    try:
        if isinstance(frame_id, str):
            modules.globals.current_frame_idx = int(Path(frame_id).stem)
        else:
            modules.globals.current_frame_idx = int(frame_id)
    except Exception:
        pass

    # process image targets -------------------------------------------------
    if is_image(modules.globals.target_path):
        if modules.globals.map_faces:
            for map_entry in modules.globals.source_target_map:
                if "source" in map_entry:
                    source_face = map_entry['source']['face']
                    target_face = map_entry['target']['face']
                    temp_frame = swap_face(source_face, target_face, temp_frame)
        elif modules.globals.many_faces:
            source_face = default_source_face()
            for map_entry in modules.globals.source_target_map:
                target_face = map_entry['target']['face']
                temp_frame = swap_face(source_face, target_face, temp_frame)

    # process video targets --------------------------------------------------
    elif is_video(modules.globals.target_path):
        if modules.globals.map_faces:
            for map_entry in modules.globals.source_target_map:
                if "source" in map_entry:
                    target_frame = [
                        f for f in map_entry['target_faces_in_frame']
                        if (f.get('location') == frame_id or f.get('frame') == frame_id)
                    ]
                    source_face = map_entry['source']['face']
                    for frame in target_frame:
                        for target_face in frame['faces']:
                            temp_frame = swap_face(source_face, target_face, temp_frame)
        elif modules.globals.many_faces:
            source_face = default_source_face()
            for map_entry in modules.globals.source_target_map:
                target_frame = [
                    f for f in map_entry['target_faces_in_frame']
                    if (f.get('location') == frame_id or f.get('frame') == frame_id)
                ]
                for frame in target_frame:
                    for target_face in frame['faces']:
                        temp_frame = swap_face(source_face, target_face, temp_frame)

    # fallback for live feed -------------------------------------------------
    else:
        detected_faces = get_many_faces(temp_frame)
        if modules.globals.map_faces and detected_faces and hasattr(modules.globals, 'simple_map') and modules.globals.simple_map:
            if len(detected_faces) <= len(modules.globals.simple_map['target_embeddings']):
                for detected_face in detected_faces:
                    closest_centroid_index, _ = find_closest_centroid(
                        modules.globals.simple_map['target_embeddings'],
                        detected_face.normed_embedding,
                    )
                    temp_frame = swap_face(
                        modules.globals.simple_map['source_faces'][closest_centroid_index],
                        detected_face,
                        temp_frame,
                    )
            else:
                detected_faces_centroids = [face.normed_embedding for face in detected_faces]
                for i, target_embedding in enumerate(modules.globals.simple_map['target_embeddings']):
                    closest_centroid_index, _ = find_closest_centroid(
                        detected_faces_centroids, target_embedding
                    )
                    if closest_centroid_index < len(detected_faces):
                        temp_frame = swap_face(
                            modules.globals.simple_map['source_faces'][i],
                            detected_faces[closest_centroid_index],
                            temp_frame,
                        )
        elif modules.globals.many_faces and detected_faces:
            source_face = default_source_face()
            for target_face in detected_faces:
                temp_frame = swap_face(source_face, target_face, temp_frame)
        elif detected_faces:
            target_face = detected_faces[0]
            source_face = default_source_face()
            if source_face:
                temp_frame = swap_face(source_face, target_face, temp_frame)

    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], progress: Any = None) -> None:
    # --- No changes needed in process_frames ---
    # Note: Ensure get_one_face is called only once if possible for efficiency if !map_faces
    source_face = None
    if not modules.globals.map_faces:
        source_face = get_combined_source_face(source_path)
        if source_face is None:
             update_status(f"Could not find face in source image: {source_path}, skipping swap.", NAME)
             # If no source face, maybe skip processing? Or handle differently.
             # For now, it will proceed but swap_face might fail later.

    for frame_idx, temp_frame_path in enumerate(temp_frame_paths):
        temp_frame = cv2.imread(temp_frame_path)
        if temp_frame is None:
            update_status(f"Warning: Could not read frame {temp_frame_path}", NAME)
            if progress:
                progress.update(1)  # Still update progress even if frame fails
            continue  # Skip to next frame

        try:
            modules.globals.current_frame_idx = int(Path(temp_frame_path).stem)
        except Exception:
            pass

        try:
            if not modules.globals.map_faces:
                if source_face:  # Only process if source face was found
                    result = process_frame(source_face, temp_frame)
                else:
                    result = temp_frame  # No source face, return original frame
            else:
                result = process_frame_v2(temp_frame, frame_idx)

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
    target_frame = cv2.imread(target_path) # Read original target for processing
    if target_frame is None:
        update_status(f"Error: Could not read target image: {target_path}", NAME)
        return

    if not modules.globals.map_faces:
        source_face = get_combined_source_face(source_path)
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
    if modules.globals.map_faces and modules.globals.many_faces:
        update_status('Many faces enabled. Using first source image (if applicable in v2). Processing...', NAME)
    # The core processing logic is delegated, which is good.
    modules.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)

def process_frame_stream(source_path: str, frame: Frame) -> Frame:
    """
    Enhance faces in a live frame stream.

    Function-static storage is used to avoid using global variables.
    """
    # Function-static storage to avoid globals
    if not hasattr(process_frame_stream, "cache"):
        process_frame_stream.cache = {
            "source_face": None,  # The source face for non-mapped processing
            "frame_idx": 0,  # The current frame index
        }

    cache = process_frame_stream.cache

    if not modules.globals.map_faces:
        # If not mapping faces, use the first source face found
        if cache["source_face"] is None:
            img = cv2.imread(source_path)
            if img is not None:
                cache["source_face"] = get_one_face(img)

        if cache["source_face"] is not None:
            # Process the frame using the source face
            return process_frame(cache["source_face"], frame)
        # If no source face, return the original frame
        return frame
    else:
        # Process the frame using the frame index
        processed = process_frame_v2(frame, cache["frame_idx"])
        # Increment the frame index for the next iteration
        cache["frame_idx"] += 1
        return processed


# STREAM_SOURCE_FACE = None
# STREAM_FRAME_IDX = 0


# def process_frame_stream(source_path: str, frame: Frame) -> Frame:
#     global STREAM_SOURCE_FACE, STREAM_FRAME_IDX
#     if not modules.globals.map_faces:
#         if STREAM_SOURCE_FACE is None:
#             STREAM_SOURCE_FACE = get_combined_source_face(source_path)
#         if STREAM_SOURCE_FACE is not None:
#             modules.globals.current_frame_idx = STREAM_FRAME_IDX
#             STREAM_FRAME_IDX += 1
#             return process_frame(STREAM_SOURCE_FACE, frame)
#         return frame
#     else:
#         modules.globals.current_frame_idx = STREAM_FRAME_IDX
#         processed = process_frame_v2(frame, STREAM_FRAME_IDX)
#         STREAM_FRAME_IDX += 1
#         return processed
