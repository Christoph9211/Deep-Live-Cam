import os  # <-- Added for os.path.exists
import time
import math
from typing import Any, Iterable, List, Optional, Sequence, Tuple
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
try:
    # Optional semantic segmenter (MediaPipe Face Mesh based)
    from modules.segmenters.semantic import get_mouth_mask as get_semantic_mouth_mask
except Exception:
    get_semantic_mouth_mask = None  # type: ignore

# Try to import unified region masks provider
try:
    from modules.segmenters.semantic import get_region_masks as get_semantic_region_masks  # type: ignore
except Exception:
    try:
        from modules.segmenters.bisenet_onnx import get_region_masks as get_semantic_region_masks  # type: ignore
    except Exception:
        get_semantic_region_masks = None  # type: ignore

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'DLC.FACE-SWAPPER'

# ---------------------------------------------------------------------------
# Smooth skin-mask builder (convex-hull SDF with logistic falloff)
# ---------------------------------------------------------------------------


def _select_skin_points(
    kps: np.ndarray,
    spec: str = "auto",
    available: Optional[Iterable[int]] = None,
) -> np.ndarray:
    """Return a safe subset of landmarks used to approximate facial skin."""

    n = len(kps)
    if n == 0:
        return kps

    if spec == "auto":
        spec = "468" if n >= 200 else "68"

    if spec == "68":
        idx: Sequence[int] = list(range(2, 15)) + [31, 35, 1, 15]
    elif spec == "468":
        idx = [
            93,
            132,
            58,
            172,
            136,
            150,
            149,
            176,
            148,
            152,
            323,
            361,
            288,
            397,
            365,
            379,
            378,
            400,
            377,
            383,
            127,
            234,
            454,
            227,
            447,
            205,
            50,
            280,
            101,
            330,
        ]
    else:
        idx = list(range(n))

    idx = [i for i in idx if 0 <= i < n]
    if not idx:
        idx = list(range(n))

    if available is not None:
        avail = set(int(i) for i in available)
        idx = [i for i in idx if i in avail]
        if not idx:
            idx = list(range(n))

    return kps[idx]


def build_skin_sdf_mask(
    roi_shape: Tuple[int, int],
    kps_xy: Sequence[Sequence[float]],
    offset_xy: Tuple[int, int] = (0, 0),
    landmark_spec: str = "auto",
    forehead_pad_frac: float = 0.10,
    edge_width_px: float = 18.0,
    inner_bias_px: float = 0.0,
    gamma: float = 1.0,
    min_hull_points: int = 3,
) -> np.ndarray:
    """Create a smooth facial skin mask from landmarks using an SDF profile."""

    H, W = roi_shape
    if H <= 0 or W <= 0:
        return np.zeros((max(H, 1), max(W, 1)), dtype=np.float32)

    kps = np.asarray(kps_xy, dtype=np.float32)
    if kps.ndim != 2 or kps.shape[1] < 2 or kps.shape[0] < min_hull_points:
        return np.zeros((H, W), dtype=np.float32)

    ox, oy = map(float, offset_xy)
    pts = kps[:, :2].copy()
    pts[:, 0] -= ox
    pts[:, 1] -= oy

    margin = 8.0
    in_roi = (
        (pts[:, 0] >= -margin)
        & (pts[:, 0] <= W + margin)
        & (pts[:, 1] >= -margin)
        & (pts[:, 1] <= H + margin)
    )
    pts = pts[in_roi]
    if len(pts) < min_hull_points:
        return np.zeros((H, W), dtype=np.float32)

    skin_pts = _select_skin_points(pts, spec=landmark_spec)
    if len(skin_pts) < min_hull_points:
        skin_pts = pts

    x1, y1 = np.min(skin_pts, axis=0)
    x2, y2 = np.max(skin_pts, axis=0)
    if forehead_pad_frac > 1e-6:
        pad = (y2 - y1) * float(forehead_pad_frac)
        span = (y2 - y1) if (y2 - y1) > 1e-6 else 1.0
        thresh = y1 + 0.35 * span
        top_mask = skin_pts[:, 1] <= thresh
        skin_pts[top_mask, 1] = np.maximum(0.0, skin_pts[top_mask, 1] - pad)

    skin_pts_i = np.round(skin_pts).astype(np.int32)
    if len(skin_pts_i) < min_hull_points:
        return np.zeros((H, W), dtype=np.float32)

    hull = cv2.convexHull(skin_pts_i)
    if hull is None or len(hull) < min_hull_points:
        return np.zeros((H, W), dtype=np.float32)

    inside = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(inside, hull, 1)

    dist_in = cv2.distanceTransform(inside, distanceType=cv2.DIST_L2, maskSize=3)
    dist_out = cv2.distanceTransform(1 - inside, distanceType=cv2.DIST_L2, maskSize=3)
    sdf = dist_in - dist_out

    width = max(1e-3, float(edge_width_px))
    mask = 1.0 / (1.0 + np.exp(-(sdf - float(inner_bias_px)) / width))

    if gamma != 1.0:
        mask = np.clip(mask, 0.0, 1.0) ** float(gamma)

    return mask.astype(np.float32)


def _extract_face_landmarks(face: Face) -> Optional[np.ndarray]:
    """Return the most detailed available set of 2D landmarks for *face*."""

    candidate_attrs = (
        "landmark_2d_106",
        "landmark_3d_68",
        "landmark_2d_68",
        "landmark_3d_5",
        "landmark_2d_5",
        "kps",
    )
    for attr in candidate_attrs:
        pts = getattr(face, attr, None)
        if pts is None:
            continue
        arr = np.asarray(pts, dtype=np.float32)
        if arr.ndim == 1 and arr.size % 2 == 0:
            arr = arr.reshape(-1, 2)
        if arr.ndim >= 2 and arr.shape[1] >= 2:
            return arr[:, :2].astype(np.float32)
    return None


def _compute_face_roi(face: Face, frame_shape: Tuple[int, int, int]) -> Optional[Tuple[int, int, int, int]]:
    """Return a padded, clipped ROI covering the target face."""

    h, w = frame_shape[:2]
    if hasattr(face, "bbox") and face.bbox is not None:
        x1, y1, x2, y2 = map(float, face.bbox)
    else:
        x1, y1, x2, y2 = 0.0, 0.0, float(w), float(h)

    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    pad_x = box_w * 0.12
    pad_y = box_h * 0.18

    x1 = max(0, int(math.floor(x1 - pad_x)))
    y1 = max(0, int(math.floor(y1 - pad_y)))
    x2 = min(w, int(math.ceil(x2 + pad_x)))
    y2 = min(h, int(math.ceil(y2 + pad_y)))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def _build_ellipse_mask(
    roi_shape: Tuple[int, int],
    face: Face,
    offset_xy: Tuple[int, int],
    size_scale: float,
    height_scale: float,
    feather_ratio: float,
) -> Optional[np.ndarray]:
    """Approximate a mouth/teeth preserve mask using an ellipse heuristic."""

    h, w = roi_shape
    if h <= 0 or w <= 0:
        return None

    down = float(getattr(modules.globals, "mask_down_size", 0.5) or 0.5)
    ds_w = max(1, int(round(w * down)))
    ds_h = max(1, int(round(h * down)))
    mask_small = np.zeros((ds_h, ds_w), dtype=np.float32)

    kps = getattr(face, "kps", None)
    has_kps = kps is not None
    if has_kps:
        pts = np.asarray(kps, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 2:
            has_kps = False
    if has_kps and pts.shape[0] >= 5:
        offset = np.array(offset_xy, dtype=np.float32)
        lm_left = pts[3, :2] - offset
        lm_right = pts[4, :2] - offset
        mouth_center = (lm_left + lm_right) / 2.0
        mouth_width = float(np.linalg.norm(lm_right - lm_left))
        mouth_height = mouth_width * float(height_scale)
        cx = float(mouth_center[0])
        cy = float(mouth_center[1])
    else:
        if hasattr(face, "bbox") and face.bbox is not None:
            x1, y1, x2, y2 = map(float, face.bbox)
        else:
            x1, y1, x2, y2 = float(offset_xy[0]), float(offset_xy[1]), float(offset_xy[0] + w), float(offset_xy[1] + h)
        x1 -= offset_xy[0]
        x2 -= offset_xy[0]
        y1 -= offset_xy[1]
        y2 -= offset_xy[1]
        cx = (x1 + x2) / 2.0
        cy = y1 + (y2 - y1) * 0.72
        mouth_width = (x2 - x1) * 0.40
        mouth_height = (y2 - y1) * 0.28

    ax = max(1.0, (mouth_width * 0.5) * float(size_scale))
    ay = max(1.0, (mouth_height * 0.5) * float(size_scale))

    cx_ds = int(round(cx * down))
    cy_ds = int(round(cy * down))
    ax_ds = max(1, int(round(ax * down)))
    ay_ds = max(1, int(round(ay * down)))

    cv2.ellipse(mask_small, (cx_ds, cy_ds), (ax_ds, ay_ds), 0, 0, 360, (255, 255, 255), -1)
    denom = float(mask_small.max()) if mask_small.size else 0.0
    mask_small = mask_small / max(denom, 1.0)

    feather = max(1, int(max(ax_ds, ay_ds) / max(float(feather_ratio), 1.0)))
    if feather % 2 == 0:
        feather += 1
    if feather >= 3:
        mask_small = cv2.GaussianBlur(mask_small, (feather, feather), 0)
        denom = float(mask_small.max()) if mask_small.size else 0.0
        mask_small = mask_small / max(denom, 1.0)

    mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    if mask.size:
        peak = float(mask.max())
        if peak > 0.0:
            mask = mask / peak
    return np.clip(mask, 0.0, 1.0)


def _build_hairline_mask(roi_shape: Tuple[int, int], base_mask: np.ndarray) -> np.ndarray:
    """Create a gentle falloff mask favouring the original hairline region."""

    h, w = roi_shape
    if h <= 0 or w <= 0:
        return np.zeros((h, w), dtype=np.float32)

    rows = np.linspace(0.0, 1.0, num=h, dtype=np.float32)
    top_emphasis = np.clip(1.0 - (rows / 0.35), 0.0, 1.0)
    hair = top_emphasis[:, None] * base_mask
    if hair.size:
        peak = float(hair.max())
        if peak > 0.0:
            hair = hair / peak
    return hair

# -----------------------------
# One-Euro filter for smoothing
# -----------------------------

class _LowPass:
    def __init__(self, alpha: float, init: Optional[float] = None):
        self.alpha = alpha
        self.s: Optional[float] = init

    def filter(self, x: float, alpha: Optional[float] = None) -> float:
        """
        Applies the One-Euro low-pass filter to a signal.

        The filter uses the following recurrence relation to smooth the signal:

        `s = alpha * x + (1 - alpha) * s`

        Where `alpha` is the smoothing factor, `x` is the current signal value, and `s` is the
        previous smoothed signal value.

        If `alpha` is not provided, the filter uses the `alpha` value set during initialization.

        Args:
            x (float): The current signal value.
            alpha (Optional[float), optional): The smoothing factor. Defaults to None.

        Returns:
            float: The smoothed signal value.
        """
        if alpha is None:
            alpha = self.alpha
        if self.s is None:
            self.s = x
        else:
            self.s = alpha * x + (1.0 - alpha) * self.s
        return self.s


def _alpha(cutoff: float, dt: float) -> float:
    """
    Calculates the alpha value for the One-Euro filter.

    The alpha value is calculated as `1.0 / (1.0 + tau / dt)`, where
    `tau` is the time constant of the filter and `dt` is the time step.

    If the cutoff frequency is less than or equal to zero, returns 1.0.

    Args:
        cutoff (float): The cutoff frequency for the One-Euro filter.
        dt (float): The time step in seconds.

    Returns:
        float: The alpha value for the One-Euro filter.
    """
    if cutoff <= 0.0:
        return 1.0
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


class _OneEuro:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, dcutoff: float = 1.0):
        """
        Initializes a _OneEuro object.

        Args:
            min_cutoff (float): The minimum cutoff frequency for the One-Euro filter.
            beta (float): The beta value for the One-Euro filter.
            dcutoff (float): The derivative cutoff frequency for the One-Euro filter.
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self.dx = _LowPass(alpha=1.0)
        self.x = _LowPass(alpha=1.0)
        self._prev: Optional[float] = None

    def filter(self, x: float, dt: float) -> float:
        """
        Applies the One-Euro filter to a signal.

        The filter is a second-order low-pass filter that is optimized for
        real-time signal processing. It is designed to remove high-frequency
        noise from a signal while preserving the signal's low-frequency
        components.

        The filter's cutoff frequency is dynamically adjusted based on the signal's
        derivative. The derivative is used to estimate the signal's high-frequency
        components, and the cutoff frequency is adjusted accordingly.

        The filter's output is a smoothed version of the input signal.
        """
        if self._prev is None:
            self._prev = x
        # Derivative of the signal
        dx = (x - self._prev) / max(dt, 1e-6)
        self._prev = x
        edx = self.dx.filter(dx, _alpha(self.dcutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self.x.filter(x, _alpha(cutoff, dt))


class _VectorEuro:
    def __init__(self, dim: int, min_cutoff: float, beta: float, dcutoff: float):
        """
        Initializes a _VectorEuro object.

        Args:
            dim (int): The number of dimensions to smooth.
            min_cutoff (float): The minimum cutoff frequency for the One-Euro filter.
            beta (float): The beta value for the One-Euro filter.
            dcutoff (float): The derivative cutoff frequency for the One-Euro filter.
        """
        self.filters = [_OneEuro(min_cutoff, beta, dcutoff) for _ in range(dim)]

    def filter(self, vec: List[float], dt: float) -> List[float]:

        """
        Applies the One-Euro filter to a list of values.

        Args:
            vec (List[float]): The list of values to smooth.
            dt (float): The time step in seconds.

        Returns:
            List[float]: The smoothed list of values.
        """
        return [self.filters[i].filter(float(vec[i]), dt) for i in range(len(vec))]


# Smoothing state
_SMOOTH_SINGLE: Optional[Tuple[_VectorEuro, _VectorEuro]] = None  # (kps10, bbox4)
_SMOOTH_LAST_TS: Optional[float] = None
_SMOOTH_IN_STREAM: bool = False


def _smoothing_dt() -> float:
    """
    Returns the time step for smoothing in seconds.

    If 'smoothing_use_fps' is True, the time step is calculated as 1.0 / fps,
    where fps is the value of 'smoothing_fps' or 30.0 if not set.

    Otherwise, the time step is the difference in seconds between the current time
    and the last time step, or 1e-3 if the difference is less than 1e-3.
    """
    if getattr(modules.globals, 'smoothing_use_fps', True):
        fps = float(getattr(modules.globals, 'smoothing_fps', 30.0) or 30.0)
        return 1.0 / max(1.0, fps)
    global _SMOOTH_LAST_TS
    now = time.perf_counter()
    if _SMOOTH_LAST_TS is None:
        _SMOOTH_LAST_TS = now
        return 1.0 / float(getattr(modules.globals, 'smoothing_fps', 30.0) or 30.0)
    dt = now - _SMOOTH_LAST_TS
    _SMOOTH_LAST_TS = now
    return max(1e-3, dt)


def _get_single_filters() -> Tuple[_VectorEuro, _VectorEuro]:
    """
    Returns a tuple of two _VectorEuro filters, for smoothing the keypoints
    and bounding box of the face in the stream. The filters are created once
    and stored in the _SMOOTH_SINGLE global variable to avoid repeated creation.
    The filters are created with the values of the smoothing_min_cutoff,
    smoothing_beta, and smoothing_dcutoff attributes of the modules.globals
    object, which can be set in the Advanced Settings popup.
    """

    global _SMOOTH_SINGLE
    if _SMOOTH_SINGLE is None:
        mc = float(getattr(modules.globals, 'smoothing_min_cutoff', 1.0))
        beta = float(getattr(modules.globals, 'smoothing_beta', 0.0))
        dc = float(getattr(modules.globals, 'smoothing_dcutoff', 1.0))
        _SMOOTH_SINGLE = (_VectorEuro(10, mc, beta, dc), _VectorEuro(4, mc, beta, dc))
    return _SMOOTH_SINGLE


def _auto_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """
    Automatically apply optimal lower and upper thresholds to the Canny edge detector.

    Args:
        gray (np.ndarray): Input grayscale image
        sigma (float, optional): Standard deviation of Gaussian distribution. Defaults to 0.33.

    Returns:
        np.ndarray: Output binary image containing detected edges
    """

    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)


def _expand_bbox(bbox: Tuple[float, float, float, float], w: int, h: int, pad: float = 0.12) -> Tuple[int, int, int, int]:
    """
    Expand a bounding box by a given padding factor.

    Args:
        bbox (Tuple[float, float, float, float]): Bounding box coordinates (x1, y1, x2, y2)
        w (int): Width of the image
        h (int): Height of the image
        pad (float): Padding factor (default = 0.12)

    Returns:
        Tuple[int, int, int, int]: Expanded bounding box coordinates (xi1, yi1, xi2, yi2)
    """
    x1, y1, x2, y2 = map(float, bbox)
    bw = x2 - x1
    bh = y2 - y1
    x1 -= bw * pad
    y1 -= bh * pad
    x2 += bw * pad
    y2 += bh * pad
    xi1 = max(0, int(math.floor(x1)))
    yi1 = max(0, int(math.floor(y1)))
    xi2 = min(w, int(math.ceil(x2)))
    yi2 = min(h, int(math.ceil(y2)))
    if xi2 <= xi1 or yi2 <= yi1:
        return 0, 0, w, h
    return xi1, yi1, xi2, yi2


def _apply_occlusion_preserve(original_frame: Frame, swapped_frame: Frame, target_face: Face) -> Frame: # type: ignore
    """
    Preserves foreground occluders (e.g., hands, props) by reinstating
    original pixels where strong edges present in the original are missing
    in the swapped output. Enabled by default.

    The detection process is based on edge differences between the original and swapped frames.
    The edges are detected using the Canny edge detector after applying a bilateral filter to reduce noise.
    The edges are then dilated to cover thin fingers/props, and the resulting mask is used to blend the original and swapped frames.
    The blending process is soft, meaning that the original and swapped frames are weighted by the edge mask and then added together.
    The final output is a frame that contains the original occluders, but with the swapped face.
    """
    try:
        h, w = swapped_frame.shape[:2]
        s = float(getattr(modules.globals, 'occlusion_sensitivity', 0.5) or 0.0)
        s = 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)
        sigma = max(0.1, min(0.5, 0.5 - 0.35 * s))
        k_factor = 0.012 + 0.028 * s
        b_factor = 0.012 + 0.028 * s
        pctl = 90.0 - 40.0 * s
        # Derive ROI from face bbox; fall back to full frame
        if hasattr(target_face, 'bbox') and target_face.bbox is not None:
            rx1, ry1, rx2, ry2 = _expand_bbox(tuple(target_face.bbox), w, h, pad=0.12)
        else:
            rx1, ry1, rx2, ry2 = 0, 0, w, h

        roi_o = original_frame[ry1:ry2, rx1:rx2]
        roi_s = swapped_frame[ry1:ry2, rx1:rx2]
        if roi_o.size == 0 or roi_s.size == 0:
            return swapped_frame

        # Edge-based occlusion detection: edges in original missing in swapped
        gray_o = cv2.cvtColor(roi_o, cv2.COLOR_BGR2GRAY)
        gray_s = cv2.cvtColor(roi_s, cv2.COLOR_BGR2GRAY)
        # Light denoise to stabilize edges
        gray_o = cv2.bilateralFilter(gray_o, 5, 75, 75)
        gray_s = cv2.bilateralFilter(gray_s, 5, 75, 75)

        e_o = _auto_canny(gray_o, sigma=sigma)
        e_s = _auto_canny(gray_s, sigma=sigma)
        missing = cv2.subtract(e_o, e_s)  # edges present in original but not in swapped

        # Expand to cover thin fingers/props; kernel scales with ROI size
        k = max(3, int(k_factor * max(roi_o.shape[0], roi_o.shape[1])))
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        occ = cv2.dilate(missing, kernel)

        # Optional gating by color difference to avoid over-preservation
        diff = cv2.absdiff(roi_o, roi_s)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        # Normalize and threshold softly
        thr = max(5, int(np.percentile(diff_gray, pctl)))
        diff_mask = (diff_gray > thr).astype(np.uint8) * 255
        occ = cv2.bitwise_and(occ, diff_mask)

        # Soften to alpha matte
        b = max(3, int(b_factor * max(roi_o.shape[0], roi_o.shape[1])))
        if b % 2 == 0:
            b += 1
        alpha = cv2.GaussianBlur(occ, (b, b), 0).astype(np.float32) / 255.0
        alpha = np.clip(alpha, 0.0, 1.0)
        alpha3 = np.repeat(alpha[:, :, None], 3, axis=2)

        blended = (roi_o.astype(np.float32) * alpha3 + roi_s.astype(np.float32) * (1.0 - alpha3))
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        out = swapped_frame.copy()
        out[ry1:ry2, rx1:rx2] = blended
        return out
    except Exception:
        return swapped_frame


def _smooth_face_inplace(face: Face, dt: float) -> None: # type: ignore
    """Smooths the face landmarks and bounding box of a face using a
    single-pole IIR filter.

    Args:
        face (Face): The face containing the landmarks and bounding box to smooth.
        dt (float): The time step for the filter.

    Returns:
        None
    """
    try:
        kps = getattr(face, 'kps', None)
        bbox = getattr(face, 'bbox', None)
        if kps is None and bbox is None:
            return
        kpsf, bboxf = _get_single_filters()
        if kps is not None:
            flat = [float(v) for v in kps.flatten().tolist()]
            out = kpsf.filter(flat, dt)
            face.kps = np.asarray(out, dtype=np.float32).reshape(-1, 2)  # type: ignore[attr-defined]
        if bbox is not None:
            bb = [float(b) for b in bbox]
            outb = bboxf.filter(bb, dt)
            face.bbox = np.asarray(outb, dtype=np.float32)  # type: ignore[attr-defined]
    except Exception:
        # Be robust: if smoothing fails, skip without breaking swap
        pass


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
    if not modules.globals.source_path or not modules.globals.target_path:
        update_status('Source and target paths must be set.', NAME)
        return False
    if not is_image(modules.globals.source_path) and not is_video(modules.globals.source_path):
        update_status('Source path must be an image or video.', NAME)
        return False
    if not is_image(modules.globals.target_path) and not is_video(modules.globals.target_path):
        update_status('Target path must be an image or video.', NAME)
        return False
    if not modules.globals.map_faces:
        source_img = cv2.imread(modules.globals.source_path)
        if source_img is None:
            update_status(f"Error: Could not read source image: {modules.globals.source_path}", NAME)
            return False
        source_face = get_one_face(source_img)
        if source_face is None:
            update_status(f"No face in source path detected: {modules.globals.source_path}", NAME)
            return False
    return True


def get_face_swapper() -> Any:
    """
    Returns the face swapper model instance.

    This function will load the face swapper model from disk if it has not been loaded before.
    It will first try to load the FP32 model, and if that fails, it will try to load the FP16 model.
    If neither model is found, it will raise a FileNotFoundError.

    The function is thread-safe and will block other threads until the model is loaded or an exception is thrown.
    """
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


def _apply_mouth_mask(original_frame: Frame, swapped_frame: Frame, target_face: Face) -> Frame:  # type: ignore
    """Blend original mouth region back using semantic parsing when available."""
    try:
        h, w = swapped_frame.shape[:2]

        def _normalize_mask(mask: np.ndarray) -> np.ndarray:
            """
            Normalize a mask to the range [0.0, 1.0].
            If the maximum value of the mask is greater than 1.0, it is divided by 255.0 to scale it down.
            Finally, the values are clipped to the range [0.0, 1.0] to ensure it is a valid mask.
            """

            m = mask.astype(np.float32, copy=False)
            if m.size and float(m.max()) > 1.0:
                m = m / 255.0
            return np.clip(m, 0.0, 1.0)

        # 1) Try unified region masks if available (BiSeNet preferred)
        if get_semantic_region_masks is not None:
            try:
                masks = get_semantic_region_masks(swapped_frame, target_face)  # type: ignore
            except Exception:
                masks = None
            if isinstance(masks, dict):
                composed = swapped_frame.astype(np.float32)
                applied = False
                # Teeth/inner mouth preservation
                if getattr(modules.globals, 'preserve_teeth', False) and 'inner_mouth' in masks and isinstance(masks['inner_mouth'], np.ndarray):
                    m = _normalize_mask(masks['inner_mouth'])
                    m3 = np.repeat(m[:, :, None], 3, axis=2)
                    composed = original_frame.astype(np.float32) * m3 + composed * (1.0 - m3)
                    applied = True
                # Mouth/lips preservation
                if getattr(modules.globals, 'mouth_mask', False) and 'mouth' in masks and isinstance(masks['mouth'], np.ndarray):
                    m = _normalize_mask(masks['mouth'])
                    m3 = np.repeat(m[:, :, None], 3, axis=2)
                    composed = original_frame.astype(np.float32) * m3 + composed * (1.0 - m3)
                    applied = True
                # Hairline preservation
                if getattr(modules.globals, 'preserve_hairline', False) and 'hair' in masks and isinstance(masks['hair'], np.ndarray):
                    m = _normalize_mask(masks['hair'])
                    m3 = np.repeat(m[:, :, None], 3, axis=2)
                    composed = original_frame.astype(np.float32) * m3 + composed * (1.0 - m3)
                    applied = True

                composed = np.clip(composed, 0, 255).astype(np.uint8)
                if applied:
                    if getattr(modules.globals, 'show_mouth_mask_box', False):
                        try:
                            to_draw = []
                            if getattr(modules.globals, 'mouth_mask', False) and 'mouth' in masks:
                                to_draw.append(masks['mouth'])
                            if getattr(modules.globals, 'preserve_teeth', False) and 'inner_mouth' in masks:
                                to_draw.append(masks['inner_mouth'])
                            if getattr(modules.globals, 'preserve_hairline', False) and 'hair' in masks:
                                to_draw.append(masks['hair'])
                            for ms in to_draw:
                                vis = (_normalize_mask(ms) * 255).astype(np.uint8)
                                contours, _ = cv2.findContours(vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                cv2.drawContours(composed, contours, -1, (0, 255, 0), 2)
                        except Exception:
                            pass
                    return composed

        # 1b) Try single semantic mouth mask (MediaPipe path)
        mask = None
        if get_semantic_mouth_mask is not None:
            try:
                mask = get_semantic_mouth_mask(swapped_frame, target_face)
            except Exception:
                mask = None
        if isinstance(mask, np.ndarray) and mask.shape[:2] == (h, w) and getattr(modules.globals, 'mouth_mask', False):
            mask = _normalize_mask(mask)
            feather = int(max(3, max(w, h) / max(float(getattr(modules.globals, 'mask_feather_ratio', 8) or 8), 1.0)))
            if feather % 2 == 0:
                feather += 1
            if feather >= 3:
                mask = cv2.GaussianBlur(mask, (feather, feather), 0)
                mask = _normalize_mask(mask)
            mask_3 = np.repeat(mask[:, :, None], 3, axis=2)
            composed = (original_frame.astype(np.float32) * mask_3 + swapped_frame.astype(np.float32) * (1.0 - mask_3))
            composed = np.clip(composed, 0, 255).astype(np.uint8)
            if getattr(modules.globals, 'show_mouth_mask_box', False):
                try:
                    vis = (mask * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(composed, contours, -1, (0, 255, 0), 2)
                except Exception:
                    pass
            return composed

        # 2) Fallback: signed-distance skin mask with logistic edge profile
        roi = _compute_face_roi(target_face, swapped_frame.shape)
        if roi is None:
            return swapped_frame

        rx1, ry1, rx2, ry2 = roi
        roi_w = rx2 - rx1
        roi_h = ry2 - ry1
        if roi_w <= 0 or roi_h <= 0:
            return swapped_frame

        original_roi = original_frame[ry1:ry2, rx1:rx2]
        swapped_roi = swapped_frame[ry1:ry2, rx1:rx2]

        landmarks = _extract_face_landmarks(target_face)
        if landmarks is None or len(landmarks) < 3:
            # Use the ROI corners as a minimal convex hull fallback
            landmarks = np.array(
                [
                    [rx1, ry1],
                    [rx2, ry1],
                    [rx2, ry2],
                    [rx1, ry2],
                ],
                dtype=np.float32,
            )

        edge_px = max(12.0, 0.08 * float(max(roi_w, roi_h)))
        mask = build_skin_sdf_mask(
            roi_shape=(roi_h, roi_w),
            kps_xy=landmarks,
            offset_xy=(rx1, ry1),
            landmark_spec="auto",
            forehead_pad_frac=0.10,
            edge_width_px=edge_px,
            inner_bias_px=1.5,
            gamma=1.05,
        )

        if not mask.size or float(mask.max()) <= 0.0:
            return swapped_frame

        swap_weight = np.clip(mask, 0.0, 1.0)

        size_scale = float(getattr(modules.globals, "mask_size", 1.0) or 1.0)
        feather_ratio = float(getattr(modules.globals, "mask_feather_ratio", 8) or 8)

        if getattr(modules.globals, "mouth_mask", False):
            mouth_mask = _build_ellipse_mask(
                (roi_h, roi_w),
                target_face,
                (rx1, ry1),
                size_scale=size_scale,
                height_scale=0.6,
                feather_ratio=feather_ratio,
            )
            if isinstance(mouth_mask, np.ndarray):
                swap_weight *= 1.0 - np.clip(mouth_mask, 0.0, 1.0)

        if getattr(modules.globals, "preserve_teeth", False):
            teeth_mask = _build_ellipse_mask(
                (roi_h, roi_w),
                target_face,
                (rx1, ry1),
                size_scale=size_scale * 0.55,
                height_scale=0.45,
                feather_ratio=max(3.0, feather_ratio * 0.75),
            )
            if isinstance(teeth_mask, np.ndarray):
                swap_weight *= 1.0 - np.clip(teeth_mask, 0.0, 1.0)

        if getattr(modules.globals, "preserve_hairline", False):
            hair_mask = _build_hairline_mask((roi_h, roi_w), mask)
            if isinstance(hair_mask, np.ndarray) and hair_mask.size:
                swap_weight *= 1.0 - np.clip(hair_mask, 0.0, 1.0)

        swap_weight = np.clip(swap_weight, 0.0, 1.0)
        mask_3 = np.repeat(swap_weight[:, :, None], 3, axis=2)
        composed_roi = (
            swapped_roi.astype(np.float32) * mask_3
            + original_roi.astype(np.float32) * (1.0 - mask_3)
        )
        composed_roi = np.clip(composed_roi, 0, 255).astype(np.uint8)

        composed = swapped_frame.copy()
        composed[ry1:ry2, rx1:rx2] = composed_roi

        if getattr(modules.globals, "show_mouth_mask_box", False):
            try:
                vis = (np.clip(1.0 - swap_weight, 0.0, 1.0) * 255).astype(np.uint8)
                contours, _ = cv2.findContours(vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(composed[ry1:ry2, rx1:rx2], contours, -1, (0, 255, 0), 2)
            except Exception:
                pass
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
    # Optional smoothing on target landmarks/bbox
    if getattr(modules.globals, 'smoothing_enabled', False) and not getattr(modules.globals, 'many_faces', False):
        if (not getattr(modules.globals, 'smoothing_stream_only', True)) or _SMOOTH_IN_STREAM:
            _smooth_face_inplace(target_face, _smoothing_dt())

    swapped = swapper.get(temp_frame, target_face, source_face, paste_back=True)

    # Apply region preservation if any toggle is enabled
    if (
        getattr(modules.globals, 'mouth_mask', False)
        or getattr(modules.globals, 'preserve_teeth', False)
        or getattr(modules.globals, 'preserve_hairline', False)
    ):
        swapped = _apply_mouth_mask(original_frame, swapped, target_face)
    # Preserve foreground occluders (hands/props) by reinstating
    # original pixels where strong edges were removed by swapping
    if getattr(modules.globals, 'occlusion_aware_compositing', True):
        swapped = _apply_occlusion_preserve(original_frame, swapped, target_face)
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
                if target_face is not None:
                    temp_frame = swap_face(source_face, target_face, temp_frame)

        elif not modules.globals.many_faces:
            for map_entry in modules.globals.source_target_map: # Renamed 'map' to 'map_entry'
                if "source" in map_entry:
                    source_face = map_entry['source']['face']
                    target_face = map_entry['target']['face']
                    if source_face is not None and target_face is not None:
                        temp_frame = swap_face(source_face, target_face, temp_frame)

    elif is_video(modules.globals.target_path):
        if modules.globals.many_faces:
            source_face = default_source_face()
            for map_entry in modules.globals.source_target_map: # Renamed 'map' to 'map_entry'
                target_frame = [f for f in map_entry['target_faces_in_frame'] if f['location'] == temp_frame_path]

                for frame in target_frame:
                    if frame is not None:
                        for target_face in frame['faces']:
                            if target_face is not None:
                                temp_frame = swap_face(source_face, target_face, temp_frame)

        elif not modules.globals.many_faces:
            for map_entry in modules.globals.source_target_map: # Renamed 'map' to 'map_entry'
                if "source" in map_entry:
                    target_frame = [f for f in map_entry['target_faces_in_frame'] if f['location'] == temp_frame_path]
                    source_face = map_entry['source']['face']

                    for frame in target_frame:
                        if frame is not None:
                            for target_face in frame['faces']:
                                if target_face is not None:
                                    temp_frame = swap_face(source_face, target_face, temp_frame)
    else: # Fallback for neither image nor video (e.g., live feed?)
        detected_faces = get_many_faces(temp_frame)
        if modules.globals.many_faces:
            if detected_faces is not None:
                source_face = default_source_face()
                for target_face in detected_faces:
                    if target_face is not None:
                        temp_frame = swap_face(source_face, target_face, temp_frame)

        elif not modules.globals.many_faces:
            if detected_faces is not None:
                if hasattr(modules.globals, 'simple_map') and modules.globals.simple_map: # Check simple_map exists
                    if len(detected_faces) <= len(modules.globals.simple_map['target_embeddings']):
                        for detected_face in detected_faces:
                            if detected_face is not None:
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
    global _SMOOTH_IN_STREAM
    if not modules.globals.map_faces:
        if STREAM_SOURCE_FACE is None:
            source_img = cv2.imread(source_path)
            if source_img is not None:
                STREAM_SOURCE_FACE = get_one_face(source_img)
        if STREAM_SOURCE_FACE is not None:
            # Mark smoothing context as streaming for this call only
            prev_flag = _SMOOTH_IN_STREAM
            _SMOOTH_IN_STREAM = True
            try:
                return process_frame(STREAM_SOURCE_FACE, frame)
            finally:
                _SMOOTH_IN_STREAM = prev_flag
        return frame
    else:
        # Streaming context also applies to v2 path
        prev_flag = _SMOOTH_IN_STREAM
        _SMOOTH_IN_STREAM = True
        try:
            return process_frame_v2(frame)
        finally:
            _SMOOTH_IN_STREAM = prev_flag
