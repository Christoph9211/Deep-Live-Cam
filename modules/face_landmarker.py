from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from onnxruntime import (  # type: ignore
        GraphOptimizationLevel,
        InferenceSession,
        SessionOptions,
        get_available_providers,
    )
except Exception:  # pragma: no cover - onnxruntime may be optional
    GraphOptimizationLevel = None  # type: ignore
    InferenceSession = None  # type: ignore
    SessionOptions = None  # type: ignore

    def get_available_providers() -> List[str]:  # type: ignore
        return ["CPUExecutionProvider"]

import modules.globals
from modules.utilities import conditional_download, resolve_relative_path


@dataclass(frozen=True)
class LandmarkModelDef:
    """Definition for a single ONNX landmark model."""

    set_name: str
    name: str
    filename: str
    urls: Tuple[str, ...]
    input_size: int
    heatmap_size: int
    num_landmarks: int = 68


@dataclass
class _LandmarkModelRuntime:
    definition: LandmarkModelDef
    session: InferenceSession
    input_name: str
    output_name: str


@lru_cache(maxsize=None)
def _models_dir() -> str:
    return resolve_relative_path("../models")


@lru_cache(maxsize=None)
def _landmark_model_sets() -> Dict[str, Tuple[LandmarkModelDef, ...]]:
    """Return available landmark model definitions grouped by set name."""

    return {
        "2dfan": (
            LandmarkModelDef(
                set_name="2dfan",
                name="2dfan4",
                filename="2dfan4.onnx",
                urls=(
                    "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/2dfan4.onnx",
                ),
                input_size=256,
                heatmap_size=64,
            ),
        ),
        "peppa_wutz": (
            LandmarkModelDef(
                set_name="peppa_wutz",
                name="peppa_wutz",
                filename="peppa_wutz.onnx",
                urls=(
                    "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/peppa_wutz.onnx",
                ),
                input_size=256,
                heatmap_size=64,
            ),
        ),
    }


def available_landmark_sets(include_virtual: bool = True) -> List[str]:
    """Return the list of supported landmark model sets.

    Parameters
    ----------
    include_virtual:
        When True, include helper entries like "auto" and "insightface".
    """

    concrete = sorted(_landmark_model_sets().keys())
    if include_virtual:
        return ["auto", "insightface", *concrete]
    return concrete


@lru_cache(maxsize=None)
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


@lru_cache(maxsize=None)
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


def _ensure_model_file(definition: LandmarkModelDef) -> Optional[str]:
    os.makedirs(_models_dir(), exist_ok=True)
    path = os.path.join(_models_dir(), definition.filename)
    if os.path.exists(path):
        return path
    if definition.urls:
        try:
            conditional_download(_models_dir(), list(definition.urls))
        except Exception:
            pass
    return path if os.path.exists(path) else None


@lru_cache(maxsize=None)
def _load_model_runtime(definition: LandmarkModelDef) -> Optional[_LandmarkModelRuntime]:
    if InferenceSession is None:
        return None
    path = _ensure_model_file(definition)
    if path is None:
        return None
    providers = _providers()
    so = _session_options()
    try:
        session = InferenceSession(path, sess_options=so, providers=providers)  # type: ignore[call-arg]
    except Exception:
        try:
            session = InferenceSession(path, sess_options=so)  # type: ignore[call-arg]
        except Exception:
            return None
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if not inputs or not outputs:
        return None
    return _LandmarkModelRuntime(
        definition=definition,
        session=session,
        input_name=inputs[0].name,
        output_name=outputs[0].name,
    )


def _normalize_angle(angle: float) -> float:
    try:
        value = float(angle)
    except Exception:
        return 0.0
    if math.isnan(value) or math.isinf(value):
        return 0.0
    # InsightFace may report roll in degrees but near zero for frontal faces.
    # If the absolute value is modest (<= 45) we treat it directly as degrees.
    # Otherwise, fall back to converting from radians when the value looks like radians.
    if abs(value) <= 45.0:
        return value
    if abs(value) <= math.pi * 1.5:
        return math.degrees(value)
    return value


def _crop_with_padding(image: np.ndarray, left: int, top: int, size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = image.shape[:2]
    right = left + size
    bottom = top + size
    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - w)
    pad_bottom = max(0, bottom - h)
    if pad_left or pad_top or pad_right or pad_bottom:
        image = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REFLECT_101,
        )
        left += pad_left
        top += pad_top
    crop = image[top : top + size, left : left + size]
    return crop, (left, top)


def _aligned_square_crop(
    frame: np.ndarray,
    bbox: Sequence[float],
    angle: float,
    scale: float = 1.35,
) -> Optional[Tuple[np.ndarray, Tuple[int, int], np.ndarray, int]]:
    if frame is None or frame.size == 0:
        return None
    if bbox is None:
        return None
    if len(bbox) < 4:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    side = max(x2 - x1, y2 - y1)
    if not math.isfinite(side) or side <= 0:
        return None
    side = int(max(32.0, round(side * scale)))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    rotation = cv2.getRotationMatrix2D((cx, cy), _normalize_angle(angle), 1.0)
    rotated = cv2.warpAffine(
        frame,
        rotation,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    crop, origin = _crop_with_padding(rotated, left, top, side)
    inverse = cv2.invertAffineTransform(rotation.astype(np.float32))
    return crop, origin, inverse, side


def _preprocess_fan(image: np.ndarray, input_size: int) -> np.ndarray:
    resized = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = (tensor - 0.5) / 0.5
    tensor = np.transpose(tensor, (2, 0, 1))  # CHW
    return np.expand_dims(tensor, axis=0)


def _decode_heatmap(heatmap: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if heatmap is None:
        return None
    arr = np.asarray(heatmap)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        return None
    num, height, width = arr.shape
    flat = arr.reshape(num, -1)
    indices = np.argmax(flat, axis=1)
    scores = flat[np.arange(num), indices]
    ys = indices // width
    xs = indices % width
    coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    return coords, scores.astype(np.float32)


def _project_points(
    coords: np.ndarray,
    origin: Tuple[int, int],
    inverse: np.ndarray,
    crop_size: int,
    input_size: int,
    heatmap_shape: Tuple[int, int],
) -> np.ndarray:
    scale_x = float(input_size) / float(max(1, heatmap_shape[1]))
    scale_y = float(input_size) / float(max(1, heatmap_shape[0]))
    scale_crop = float(crop_size) / float(input_size)
    pts = np.asarray(coords, dtype=np.float32)
    pts_input = pts.copy()
    pts_input[:, 0] = pts[:, 0] * scale_x
    pts_input[:, 1] = pts[:, 1] * scale_y
    pts_rotated = pts_input * scale_crop
    offset = np.asarray(origin, dtype=np.float32)
    pts_rotated += offset
    ones = np.ones((pts_rotated.shape[0], 1), dtype=np.float32)
    hom = np.concatenate([pts_rotated, ones], axis=1)
    projected = hom @ inverse.T
    return projected[:, :2]


@dataclass
class LandmarkResult:
    points: np.ndarray
    score: float
    set_name: str
    model_name: str


def _preferred_model_sets() -> Iterable[str]:
    preference = getattr(modules.globals, "face_landmark_set", "auto") or "auto"
    if preference == "insightface":
        return []
    if preference != "auto":
        if preference in _landmark_model_sets():
            return [preference]
        return []
    order = ["2dfan", "peppa_wutz"]
    existing = _landmark_model_sets().keys()
    return [name for name in order if name in existing]


def detect_face_landmarks(
    frame: np.ndarray,
    bbox: Sequence[float],
    angle: float,
) -> Optional[LandmarkResult]:
    candidate_sets = list(_preferred_model_sets())
    if not candidate_sets:
        return None
    crop_data = _aligned_square_crop(frame, bbox, angle)
    if crop_data is None:
        return None
    crop, origin, inverse, crop_size = crop_data
    if crop is None or crop.size == 0:
        return None
    best: Optional[LandmarkResult] = None
    for set_name in candidate_sets:
        definitions = _landmark_model_sets().get(set_name)
        if not definitions:
            continue
        for definition in definitions:
            runtime = _load_model_runtime(definition)
            if runtime is None:
                continue
            try:
                input_tensor = _preprocess_fan(crop, definition.input_size)
                output = runtime.session.run(
                    [runtime.output_name], {runtime.input_name: input_tensor}
                )[0]
            except Exception:
                continue
            decoded = _decode_heatmap(output)
            if decoded is None:
                continue
            coords, scores = decoded
            if coords.shape[0] < definition.num_landmarks:
                continue
            projected = _project_points(
                coords,
                origin,
                inverse,
                crop_size,
                definition.input_size,
                (output.shape[-2], output.shape[-1]),
            )
            points = projected[: definition.num_landmarks].astype(np.float32)
            score = float(np.mean(scores[: definition.num_landmarks]))
            result = LandmarkResult(points=points, score=score, set_name=set_name, model_name=definition.name)
            if best is None or result.score > best.score:
                best = result
        if best is not None:
            break
    return best


def get_cached_landmarks(face: object, spec: str = "68") -> Optional[np.ndarray]:
    try:
        data = getattr(face, "landmark_set", None)
        if isinstance(data, dict):
            value = data.get(spec)
            if value is not None:
                arr = np.asarray(value, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    return arr[:, :2]
    except Exception:
        pass
    return None


def extract_existing_landmarks(face: object, minimum_points: int = 68) -> Optional[np.ndarray]:
    candidates = [
        getattr(face, "landmark_2d_68", None),
        getattr(face, "landmark_3d_68", None),
        getattr(face, "landmark_2d_106", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        arr = np.asarray(candidate, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] >= 2:
            arr = arr[..., :2].reshape(arr.shape[0], 2)
        if arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] >= minimum_points:
            return arr[:minimum_points, :2]
    return None


def best_landmarks_68(face: object) -> Optional[np.ndarray]:
    cached = get_cached_landmarks(face, "68")
    if cached is not None:
        return cached
    existing = extract_existing_landmarks(face, 68)
    if existing is not None:
        return existing
    return None
