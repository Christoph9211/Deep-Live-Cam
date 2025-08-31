from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np

import modules.globals
from modules.utilities import resolve_relative_path, conditional_download

from onnxruntime import (
    InferenceSession,
    get_available_providers,
    SessionOptions,
    GraphOptimizationLevel,
)

_SESSION: Optional[InferenceSession] = None
_INPUT_NAME: Optional[str] = None
_INPUT_SHAPE = (512, 512)  # (H, W)


def _models_dir() -> str:
    return resolve_relative_path('../models')


def _preferred_model_names() -> list[str]:
    # Accept common filenames for BiSeNet face parsing exports
    return [
        'face_parsing_bisenet_19.onnx',
        'bisenet_resnet18_19.onnx',
        'resnet34.onnx',
    ]


def _existing_model_path() -> Optional[str]:
    # Explicit env path wins
    env_path = os.getenv('DLC_BISENET_ONNX_PATH')
    if env_path and os.path.exists(env_path):
        return env_path
    models_dir = _models_dir()
    for name in _preferred_model_names():
        path = os.path.join(models_dir, name)
        if os.path.exists(path):
            return path
    return None


def _model_path() -> str:
    # Default path used if downloading via URL
    return os.path.join(_models_dir(), 'resnet34.onnx')


def _candidate_urls() -> list[str]:
    # Optionally allow users to set a URL to download the model
    env_url = os.getenv('DLC_BISENET_ONNX_URL')
    urls: list[str] = []
    if env_url:
        urls.append(env_url)
    return urls


def ensure_model() -> bool:
    """Ensure an ONNX model exists locally (or is downloadable).

    Returns True when a model file is present.
    """
    existing = _existing_model_path()
    if existing is not None:
        return True
    urls = _candidate_urls()
    if urls:
        try:
            conditional_download(_models_dir(), urls)
        except Exception:
            pass
    return _existing_model_path() is not None


def _create_session() -> Optional[InferenceSession]:
    if not ensure_model():
        return None
    so = SessionOptions()
    so.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL  # type: ignore[attr-defined]
    try:
        # Suppress info-level logs from ORT for this session
        so.log_severity_level = 2  # WARNING
    except Exception:
        pass
    providers = modules.globals.execution_providers or get_available_providers()
    model_p = _existing_model_path() or _model_path()
    try:
        return InferenceSession(model_p, sess_options=so, providers=providers)  # type: ignore[call-arg]
    except Exception:
        try:
            return InferenceSession(model_p, sess_options=so)
        except Exception:
            return None


def available() -> bool:
    global _SESSION, _INPUT_NAME
    if _SESSION is None:
        _SESSION = _create_session()
        if _SESSION is not None:
            _INPUT_NAME = _SESSION.get_inputs()[0].name
    return _SESSION is not None


def _preprocess_bgr(frame: np.ndarray) -> np.ndarray:
    h, w = _INPUT_SHAPE
    img = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    img = img[:, :, ::-1]  # BGR->RGB
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))  # CHW
    img = np.expand_dims(img, 0)  # NCHW
    return img


def _postprocess_logits(logits: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    pred = logits[0].argmax(axis=0).astype(np.uint8)
    if pred.shape != (out_h, out_w):
        pred = cv2.resize(pred, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return pred


# CelebAMask-HQ labels
MOUTH_LABELS = {11, 12, 13, 14}
INNER_MOUTH_LABELS = {13, 11}
HAIR_LABELS = {15}


def _labels_to_masks(labelmap: np.ndarray) -> Dict[str, np.ndarray]:
    h, w = labelmap.shape
    mouth = np.isin(labelmap, list(MOUTH_LABELS)).astype(np.float32)
    inner_mouth = np.isin(labelmap, list(INNER_MOUTH_LABELS)).astype(np.float32)
    hair = np.isin(labelmap, list(HAIR_LABELS)).astype(np.float32)

    def blur(m: np.ndarray) -> np.ndarray:
        k = max(3, int(max(h, w) * 0.01))
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(m, (k, k), 0)

    return {
        'mouth': blur(mouth),
        'inner_mouth': blur(inner_mouth),
        'hair': blur(hair),
    }


def get_region_masks(frame: np.ndarray, target_face=None) -> Optional[Dict[str, np.ndarray]]:
    """Return region masks using BiSeNet ONNX if available.

    Keys: 'mouth', 'inner_mouth', 'hair' -> float32 [H, W] in [0,1].
    """
    if not available():
        return None
    global _SESSION, _INPUT_NAME, _INPUT_SHAPE
    assert _SESSION is not None and _INPUT_NAME is not None

    # Try to infer static input size from model input (N, C, H, W)
    try:
        in0 = _SESSION.get_inputs()[0]
        shape = getattr(in0, 'shape', None)
        if isinstance(shape, (list, tuple)) and len(shape) == 4:
            H, W = shape[2], shape[3]
            if isinstance(H, int) and isinstance(W, int) and H > 0 and W > 0:
                _INPUT_SHAPE = (H, W)
    except Exception:
        pass

    h, w = frame.shape[:2]
    inp = _preprocess_bgr(frame)
    output_names = [_SESSION.get_outputs()[0].name]
    outputs = _SESSION.run(output_names, {_INPUT_NAME: inp})
    logits = outputs[0]
    labelmap = _postprocess_logits(logits, h, w)
    return _labels_to_masks(labelmap)


# late import to avoid circular
import cv2  # noqa: E402
