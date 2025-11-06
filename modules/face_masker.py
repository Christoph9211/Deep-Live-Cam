"""Utility functions for generating face swap masks.

This module implements box, occlusion, area, and semantic region masks
inspired by FaceFusion. Masks operate on warped face crops inside the
face swap pipeline and can be combined to form the final compositing
mask.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from onnxruntime import (  # type: ignore
        GraphOptimizationLevel,
        InferenceSession,
        SessionOptions,
        get_available_providers,
    )
except Exception:  # pragma: no cover - onnxruntime may be optional at import time
    GraphOptimizationLevel = None  # type: ignore
    InferenceSession = None  # type: ignore
    SessionOptions = None  # type: ignore
    get_available_providers = lambda: ["CPUExecutionProvider"]  # type: ignore

import modules.globals
from modules.utilities import conditional_download, resolve_relative_path

# Landmark index definitions reused from FaceFusion (68-point topology).
_FACE_MASK_AREA_POINTS: dict[str, Sequence[int]] = {
    "upper-face": [0, 1, 2, 31, 32, 33, 34, 35, 14, 15, 16, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17],
    "lower-face": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 35, 34, 33, 32, 31],
    "mouth": [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67],
}

# Semantic region labels (CelebAMask-HQ) consistent with FaceFusion.
_FACE_MASK_REGION_LABELS: dict[str, int] = {
    "skin": 1,
    "left-eyebrow": 2,
    "right-eyebrow": 3,
    "left-eye": 4,
    "right-eye": 5,
    "glasses": 6,
    "nose": 10,
    "mouth": 11,
    "upper-lip": 12,
    "lower-lip": 13,
}

_XSEG_MODELS: List[Tuple[str, Sequence[str]]] = [
    (
        "xseg_1.onnx",
        ("https://github.com/facefusion/facefusion-assets/releases/download/models-3.1.0/xseg_1.onnx",),
    ),
    (
        "xseg_2.onnx",
        ("https://github.com/facefusion/facefusion-assets/releases/download/models-3.1.0/xseg_2.onnx",),
    ),
    (
        "xseg_3.onnx",
        ("https://github.com/facefusion/facefusion-assets/releases/download/models-3.2.0/xseg_3.onnx",),
    ),
]

_FACE_PARSER_CANDIDATES: List[Tuple[str, Sequence[str]]] = [
    (
        "bisenet_resnet_34.onnx",
        ("https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/bisenet_resnet_34.onnx",),
    ),
    (
        "bisenet_resnet_18.onnx",
        ("https://github.com/facefusion/facefusion-assets/releases/download/models-3.1.0/bisenet_resnet_18.onnx",),
    ),
    # Accept common community exports if already provided by the user.
    ("face_parsing_bisenet_19.onnx", tuple()),
    ("bisenet_resnet18_19.onnx", tuple()),
    ("resnet34.onnx", tuple()),
]

_MODELS_DIR = resolve_relative_path("../models")
_FACE_OCCLUDER_SESSIONS: Optional[List[Tuple[InferenceSession, str, str, Tuple[int, int]]]] = None
_FACE_PARSER_SESSION: Optional[Tuple[InferenceSession, str, str, Tuple[int, int]]] = None


def _providers() -> List[str]:
    try:
        if modules.globals.execution_providers:
            return list(modules.globals.execution_providers)
    except Exception:
        pass
    try:
        return list(get_available_providers())  # type: ignore[misc]
    except Exception:
        return ["CPUExecutionProvider"]


def _session_options() -> Optional[SessionOptions]:
    if SessionOptions is None:
        return None
    so = SessionOptions()
    try:
        so.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        so.log_severity_level = 2  # WARNING
    except Exception:
        pass
    return so


def _ensure_model(filename: str, urls: Sequence[str]) -> Optional[str]:
    os.makedirs(_MODELS_DIR, exist_ok=True)
    path = os.path.join(_MODELS_DIR, filename)
    if os.path.exists(path):
        return path
    if urls:
        try:
            conditional_download(_MODELS_DIR, list(urls))
        except Exception:
            return path if os.path.exists(path) else None
    return path if os.path.exists(path) else None


def _load_face_occluder_sessions() -> List[Tuple[InferenceSession, str, str, Tuple[int, int]]]:
    global _FACE_OCCLUDER_SESSIONS
    if _FACE_OCCLUDER_SESSIONS is not None:
        return _FACE_OCCLUDER_SESSIONS
    sessions: List[Tuple[InferenceSession, str, str, Tuple[int, int]]] = []
    if InferenceSession is None:
        _FACE_OCCLUDER_SESSIONS = sessions
        return sessions
    providers = _providers()
    so = _session_options()
    for filename, urls in _XSEG_MODELS:
        path = _ensure_model(filename, urls)
        if path is None or not os.path.exists(path):
            continue
        try:
            session = InferenceSession(path, sess_options=so, providers=providers)  # type: ignore[call-arg]
        except Exception:
            try:
                session = InferenceSession(path, sess_options=so)  # type: ignore[call-arg]
            except Exception:
                continue
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if not inputs or not outputs:
            continue
        input_name = inputs[0].name
        output_name = outputs[0].name
        shape = inputs[0].shape
        if isinstance(shape, (list, tuple)) and len(shape) >= 4:
            h = int(shape[2]) if isinstance(shape[2], int) and shape[2] > 0 else 256
            w = int(shape[3]) if isinstance(shape[3], int) and shape[3] > 0 else 256
        else:
            w = h = 256
        sessions.append((session, input_name, output_name, (w, h)))
    _FACE_OCCLUDER_SESSIONS = sessions
    return sessions


def _load_face_parser_session() -> Optional[Tuple[InferenceSession, str, str, Tuple[int, int]]]:
    global _FACE_PARSER_SESSION
    if _FACE_PARSER_SESSION is not None:
        return _FACE_PARSER_SESSION
    if InferenceSession is None:
        _FACE_PARSER_SESSION = None
        return None
    providers = _providers()
    so = _session_options()
    for filename, urls in _FACE_PARSER_CANDIDATES:
        path = _ensure_model(filename, urls)
        if path is None or not os.path.exists(path):
            continue
        try:
            session = InferenceSession(path, sess_options=so, providers=providers)  # type: ignore[call-arg]
        except Exception:
            try:
                session = InferenceSession(path, sess_options=so)  # type: ignore[call-arg]
            except Exception:
                continue
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if not inputs or not outputs:
            continue
        input_name = inputs[0].name
        output_name = outputs[0].name
        shape = inputs[0].shape
        if isinstance(shape, (list, tuple)) and len(shape) >= 4:
            h = int(shape[2]) if isinstance(shape[2], int) and shape[2] > 0 else 512
            w = int(shape[3]) if isinstance(shape[3], int) and shape[3] > 0 else 512
        else:
            w = h = 512
        _FACE_PARSER_SESSION = (session, input_name, output_name, (w, h))
        return _FACE_PARSER_SESSION
    _FACE_PARSER_SESSION = None
    return None


def _gaussian_soften(mask: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    mask = mask.astype(np.float32)
    if sigma > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigma)
    return np.clip(mask, 0.0, 1.0, out=mask)


def create_box_mask(
    crop: np.ndarray,
    face_mask_blur: float,
    face_mask_padding: Sequence[float],
) -> np.ndarray:
    """Generate a padded/feathered box mask for a warped crop."""
    h, w = crop.shape[:2]
    blur_amount = max(0, int(w * 0.5 * float(face_mask_blur)))
    blur_area = max(1, blur_amount // 2)
    padding = list(face_mask_padding)[:4]
    if len(padding) < 4:
        padding.extend([0.0] * (4 - len(padding)))
    top, right, bottom, left = [max(0.0, float(p)) for p in padding]
    mask = np.ones((h, w), dtype=np.float32)
    if top > 0 or blur_amount > 0:
        mask[: max(blur_area, int(h * top / 100.0)), :] = 0.0
    if bottom > 0 or blur_amount > 0:
        mask[h - max(blur_area, int(h * bottom / 100.0)) :, :] = 0.0
    if left > 0 or blur_amount > 0:
        mask[:, : max(blur_area, int(w * left / 100.0))] = 0.0
    if right > 0 or blur_amount > 0:
        mask[:, w - max(blur_area, int(w * right / 100.0)) :] = 0.0
    if blur_amount > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), blur_amount * 0.25)
    return np.clip(mask, 0.0, 1.0)


def create_occlusion_mask(crop: np.ndarray) -> np.ndarray:
    """Infer occlusion mask from XSeg models (if available)."""
    sessions = _load_face_occluder_sessions()
    if not sessions:
        return np.ones(crop.shape[:2], dtype=np.float32)
    masks: List[np.ndarray] = []
    for session, input_name, output_name, (w, h) in sessions:
        resized = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
        inp = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0
        try:
            outputs = session.run([output_name], {input_name: inp})
        except Exception:
            continue
        if not outputs:
            continue
        mask = outputs[0]
        if mask.ndim == 4:
            mask = mask[0]
        if mask.ndim == 3:
            mask = mask[0]
        mask = mask.astype(np.float32)
        mask = cv2.resize(mask, crop.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
        masks.append(mask)
    if not masks:
        return np.ones(crop.shape[:2], dtype=np.float32)
    merged = np.minimum.reduce(masks)
    merged = _gaussian_soften(merged, 5.0)
    merged = (np.clip(merged, 0.5, 1.0) - 0.5) * 2.0
    return np.clip(merged, 0.0, 1.0)


def create_area_mask(
    crop: np.ndarray,
    landmark_68: Optional[np.ndarray],
    face_mask_areas: Sequence[str],
) -> np.ndarray:
    """Create a convex-hull mask from selected 68-point landmark areas."""
    if landmark_68 is None or len(face_mask_areas) == 0:
        return np.ones(crop.shape[:2], dtype=np.float32)
    points: List[int] = []
    for area in face_mask_areas:
        indices = _FACE_MASK_AREA_POINTS.get(str(area))
        if indices:
            points.extend(int(i) for i in indices if 0 <= int(i) < landmark_68.shape[0])
    if not points:
        return np.ones(crop.shape[:2], dtype=np.float32)
    hull_points = landmark_68[points, :2].astype(np.int32)
    if hull_points.ndim != 2 or hull_points.shape[0] < 3:
        return np.ones(crop.shape[:2], dtype=np.float32)
    convex_hull = cv2.convexHull(hull_points)
    mask = np.zeros(crop.shape[:2], dtype=np.float32)
    cv2.fillConvexPoly(mask, convex_hull, 1.0)  # type: ignore[arg-type]
    mask = _gaussian_soften(mask, 5.0)
    mask = (np.clip(mask, 0.5, 1.0) - 0.5) * 2.0
    return np.clip(mask, 0.0, 1.0)


def _preprocess_region_input(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    w, h = size
    img = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    img = img[:, :, ::-1].astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)


def create_region_mask(crop: np.ndarray, face_mask_regions: Sequence[str]) -> np.ndarray:
    """Create a semantic region mask via BiSeNet parsing (if available)."""
    if len(face_mask_regions) == 0:
        return np.ones(crop.shape[:2], dtype=np.float32)
    session_info = _load_face_parser_session()
    if session_info is None:
        return np.ones(crop.shape[:2], dtype=np.float32)
    session, input_name, output_name, (w, h) = session_info
    inp = _preprocess_region_input(crop, (w, h))
    try:
        outputs = session.run([output_name], {input_name: inp})
    except Exception:
        return np.ones(crop.shape[:2], dtype=np.float32)
    if not outputs:
        return np.ones(crop.shape[:2], dtype=np.float32)
    logits = outputs[0]
    if logits.ndim == 4:
        logits = logits[0]
    labelmap = logits.argmax(axis=0).astype(np.int32)
    valid_labels = [
        _FACE_MASK_REGION_LABELS[key]
        for key in face_mask_regions
        if key in _FACE_MASK_REGION_LABELS
    ]
    if not valid_labels:
        return np.ones(crop.shape[:2], dtype=np.float32)
    mask = np.isin(labelmap, valid_labels).astype(np.float32)
    mask = cv2.resize(mask, crop.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
    mask = _gaussian_soften(mask, 5.0)
    mask = (np.clip(mask, 0.5, 1.0) - 0.5) * 2.0
    return np.clip(mask, 0.0, 1.0)


@lru_cache(maxsize=1)
def available_area_names() -> Tuple[str, ...]:
    return tuple(_FACE_MASK_AREA_POINTS.keys())


@lru_cache(maxsize=1)
def available_region_names() -> Tuple[str, ...]:
    return tuple(_FACE_MASK_REGION_LABELS.keys())
