import os
from typing import List, Dict, Any, Optional

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.join(ROOT_DIR, "workflow")

file_types = [
    ("Image", ("*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp")),
    ("Video", ("*.mp4", "*.mkv")),
]

source_target_map = []
simple_map = {}

source_path = None
target_path = None
output_path = None
frame_processors: List[str] = []
keep_fps = True
keep_audio = True
keep_frames = False
many_faces = False
map_faces = False
color_correction = False  # New global variable for color correction toggle
nsfw_filter = False
video_encoder = None
video_quality = None
live_mirror = False
live_resizable = True
max_memory = 48
execution_providers: List[str] = []
execution_threads = 16
headless = None
log_level = "error"
fp_ui: Dict[str, bool] = {"face_enhancer": False}
camera_input_combobox = None
webcam_preview_running = False
show_fps = False
mouth_mask = False
show_mouth_mask_box = False
mask_feather_ratio = 8
mask_down_size = 0.50
mask_size = 1

# Landmark smoothing (One-Euro filter)
smoothing_enabled = False
smoothing_stream_only = True
smoothing_use_fps = True
smoothing_fps = 30.0
smoothing_min_cutoff = 1.0
smoothing_beta = 0.0
smoothing_dcutoff = 1.0
smoothing_max_track_age = 30

# Region preservation toggles
preserve_teeth = False
preserve_hairline = False

# Semantic segmenter backend selection: 'auto' | 'mediapipe' | 'bisenet'
segmenter_backend: str = 'auto'

# Occlusion-aware compositing
# Preserves foreground occluders (e.g., hands, props) by reinstating
# original pixels where strong edges present in the original are missing
# in the swapped output. Enabled by default.
occlusion_aware_compositing = True
# Sensitivity 0.0 (conservative) .. 1.0 (aggressive)
occlusion_sensitivity = 0.5

# Optional override for reusing an existing directory of extracted frames.
temp_frame_input_directory: Optional[str] = None
# Indicates that we should avoid cleaning up temp resources because we are
# reusing a pre-existing directory of extracted frames.
reuse_temp_dir = False
