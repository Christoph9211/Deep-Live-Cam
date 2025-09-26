from __future__ import annotations

"""
Semantic facial-region masking utilities.

Currently provides a mouth mask using MediaPipe Face Mesh when available.
Falls back to caller-provided heuristic if MediaPipe is not installed or
if landmark inference fails.

Contract:
    get_mouth_mask(frame, target_face) -> np.ndarray | None
        Returns an HxW float32 mask in [0, 1], where 1.0 means preserve
        original pixels and 0.0 means use swapped pixels. None on failure.
"""
import modules.globals
from mediapipe.python.solutions import face_mesh
from typing import Optional, Tuple
import numpy as np
import cv2
import modules

try:
    import mediapipe as mp  # type: ignore
    # Quiet MediaPipe/absl info-level logs that can include feedback manager notices
    try:
        from absl import logging as absl_logging  # type: ignore
        absl_logging.set_verbosity(absl_logging.ERROR)
    except Exception:
        pass
except Exception as _:
    mp = None  # type: ignore

try:
    # Optional BiSeNet ONNX backend
    from modules.segmenters import bisenet_onnx
except Exception:
    bisenet_onnx = None  # type: ignore


_FACE_MESH = None
_CACHE_KEY: Optional[tuple[int, int, int]] = None  # (ptr, w, h)
_CACHE_LANDMARKS = None


def _get_face_mesh():
    global _FACE_MESH
    if _FACE_MESH is None:
        if mp is None:
            raise RuntimeError("mediapipe not installed")
        else:
            _FACE_MESH = face_mesh.FaceMesh(
                static_image_mode=True,
                # Avoid calculators that require PROJECTION_MATRIX/IMAGE_DIMENSIONS paths
                # while still providing 468 landmarks sufficient for lips polygon.
                refine_landmarks=False,
                max_num_faces=5,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
    return _FACE_MESH


def _frame_cache_key(frame: np.ndarray) -> tuple[int, int, int]:
    """
    Returns a tuple of (data pointer, width, height) for a given frame.

    This is used as a lightweight identity within a frame pass, as the data pointer
    and shape of the frame are unlikely to change between calls.
    """
    h, w = frame.shape[:2]
    # Use data pointer + dims as a lightweight identity within a frame pass
    ptr = int(frame.__array_interface__['data'][0])
    return (ptr, w, h)


def _get_mediapipe_landmarks(frame: np.ndarray):
    """
    Returns Mediapipe landmarks for a given frame using FaceMesh.

    Uses a simple cache to avoid recomputation of landmarks for the same frame.

    Parameters
    ----------
    frame : np.ndarray
        Input frame (BGR)

    Returns
    -------
    list
        List of detected landmarks for each face (or None if no faces are detected)
    """

    global _CACHE_KEY, _CACHE_LANDMARKS
    key = _frame_cache_key(frame)
    if _CACHE_KEY == key and _CACHE_LANDMARKS is not None:
        return _CACHE_LANDMARKS
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_mesh = _get_face_mesh()
    results = face_mesh.process(img_rgb)
    _CACHE_KEY = key
    _CACHE_LANDMARKS = results.multi_face_landmarks if results else None
    return _CACHE_LANDMARKS


def _bbox_from_landmarks(landmarks, w: int, h: int) -> Tuple[int, int, int, int]:
    """
    Compute a bounding box from a list of landmarks.

    Parameters
    ----------
    landmarks : list
        List of detected landmarks for a face
    w : int
        Width of the frame
    h : int
        Height of the frame

    Returns
    -------
    tuple
        Bounding box coordinates (x1, y1, x2, y2)
    """
    xs = [int(l.x * w) for l in landmarks]
    ys = [int(l.y * h) for l in landmarks]
    x1, x2 = max(0, min(xs)), min(w - 1, max(xs))
    y1, y2 = max(0, min(ys)), min(h - 1, max(ys))
    return x1, y1, x2, y2


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """
    Compute the intersection over union (IoU) of two bounding boxes.

    Parameters
    ----------
    a : tuple
        First bounding box coordinates (x1, y1, x2, y2)
    b : tuple
        Second bounding box coordinates (x1, y1, x2, y2)

    Returns
    -------
    float
        IoU value in range [0.0, 1.0]
    """

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return (inter / denom) if denom > 0 else 0.0


def _select_face(result_faces, target_bbox: Tuple[float, float, float, float], w: int, h: int):
    """
    Select the face with the highest IoU from the list of result faces

    Parameters
    ----------
    result_faces : List[Face]
        List of faces to select from
    target_bbox : Tuple[float, float, float, float]
        Bounding box of the target face
    w : int
        Width of the frame
    h : int
        Height of the frame

    Returns
    -------
    int
        Index of the selected face, or -1 if no face is selected, or 0 if result_faces is empty
    """
    best_idx = -1
    best_iou = 0.0
    tbox = target_bbox
    for idx, fl in enumerate(result_faces):
        lms = fl.landmark
        bbox = _bbox_from_landmarks(lms, w, h)
        iou = _iou((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])), tbox)
        if iou > best_iou:
            best_iou = iou
            best_idx = idx
    return best_idx if best_idx >= 0 else (0 if result_faces else -1)


def _lip_indices() -> np.ndarray:
    """
    Returns a numpy array of indices of face mesh points that form lip connections.

    The returned array is empty if the "FACEMESH_LIPS" attribute is not present in the mediapipe FaceMesh module.

    The indices are sorted in ascending order.

    :return: A numpy array of dtype int32, containing the indices of face mesh points that form lip connections.
    :rtype: numpy.ndarray
    """
    lips = getattr(mp.solutions.face_mesh, "FACEMESH_LIPS", None)  # type: ignore[attr-defined]
    if lips is None:
        return np.array([], dtype=np.int32)
    idx = set()
    for conn in lips:  # conn is a pair of LandmarkIndex objects
        idx.add(conn[0])
        idx.add(conn[1])
    return np.array(sorted(int(i) for i in idx), dtype=np.int32)


def _get_mouth_mask_mediapipe(frame: np.ndarray, target_face) -> Optional[np.ndarray]:
    """
    Return a mouth mask using MediaPipe Face Mesh when available.

    This implementation selects a face from the input frame by finding the one with the highest IoU with the target face bounding box.
    It then uses the convex hull of the lips points to create a mask.

    Args:
        frame: Input frame as a numpy array.
        target_face: Target face object with an optional bbox attribute.

    Returns:
        A numpy array of shape (h, w) with dtype uint8, representing the mask in [0, 255].
        None if MediaPipe is not installed, or if the input frame is empty, or if the target face is not found.
    """
    if mp is None or frame is None or frame.size == 0:
        return None
    h, w = frame.shape[:2]
    results = _get_mediapipe_landmarks(frame)
    if not results:
        return None
    if hasattr(target_face, 'bbox') and target_face.bbox is not None:
        tx1, ty1, tx2, ty2 = map(float, target_face.bbox)
        target_bbox = (tx1, ty1, tx2, ty2)
    else:
        target_bbox = (0.0, 0.0, float(w), float(h))
    idx = _select_face(results, target_bbox, w, h)
    if idx < 0:
        return None
    face_lms = results[idx]
    lip_ids = _lip_indices()
    if lip_ids.size == 0:
        return None
    pts = []
    for i in lip_ids:
        if 0 <= i < len(face_lms):
            px = int(round(face_lms[i].x * w))
            py = int(round(face_lms[i].y * h))
            if 0 <= px < w and 0 <= py < h:
                pts.append([px, py])

    if len(pts) < 3:
        return None
    pts = np.array(pts, dtype=np.int32)
    hull = cv2.convexHull(pts)
    mask = np.ones((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, (0, 0, 0), lineType=cv2.LINE_AA)
    k = max(3, int(max(w, h) * 0.01))
    if k % 2 == 0:
        k += 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask


def get_mouth_mask(frame: np.ndarray, target_face) -> Optional[np.ndarray]:
    """Select backend and produce a soft mouth mask [H, W] in [0,1]."""
    backend = getattr(modules.globals, 'segmenter_backend', 'auto')

    # Try BiSeNet first if requested/auto and available
    if backend in ('auto', 'bisenet') and bisenet_onnx is not None and bisenet_onnx.available():
        try:
            masks = bisenet_onnx.get_region_masks(frame, target_face)
            if masks and 'inner_mouth' in masks:
                # Prefer inner_mouth, fallback to mouth union
                mm = masks['inner_mouth']
                if mm is not None and mm.max() > 0:
                    return mm.astype(np.float32)
            if masks and 'mouth' in masks and masks['mouth'] is not None:
                return masks['mouth'].astype(np.float32)
        except Exception:
            pass

    # Fallback to MediaPipe
    if backend in ('auto', 'mediapipe'):
        mp_mask = _get_mouth_mask_mediapipe(frame, target_face)
        if isinstance(mp_mask, np.ndarray):
            return mp_mask

    return None


def get_region_masks(frame: np.ndarray, target_face=None):
    """Unified region masks provider.

    Returns a dict with any of: 'mouth', 'inner_mouth', 'hair' -> float32 [H, W] in [0,1],
    or None if no backend available.
    """
    backend = getattr(modules.globals, 'segmenter_backend', 'auto')

    # Prefer BiSeNet when available/selected
    if backend in ('auto', 'bisenet') and bisenet_onnx is not None and bisenet_onnx.available():
        try:
            return bisenet_onnx.get_region_masks(frame, target_face)
        except Exception:
            pass

    # Fallback to MediaPipe mouth-only
    if backend in ('auto', 'mediapipe'):
        mm = _get_mouth_mask_mediapipe(frame, target_face)
        if isinstance(mm, np.ndarray):
            return {'mouth': mm}

    return None
