import os
from typing import List, Dict, Any

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.join(ROOT_DIR, "workflow")

file_types = [
    ("Image", ("*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp")),
    ("Video", ("*.mp4", "*.mkv")),
]

souce_target_map = []
simple_map = {}

source_path = None
target_path = None
output_path = None
frame_processors: List[str] = []
keep_fps = True
keep_frames = False
many_faces = False
map_faces = False
color_correction = False  # New global variable for color correction toggle
nsfw_filter = False
max_memory = None
execution_providers: List[str] = []
execution_threads = None
log_level = "error"
camera_input_combobox = None
webcam_preview_running = False
show_fps = False
mouth_mask = False
show_mouth_mask_box = False
mask_feather_ratio = 8
mask_down_size = 0.50
mask_size = 1

# live_stream_min_duration = 5.0
# live_stream_prev_time = 0.0

user_select_image_path = None
user_select_image = None
live_stream_state = "default"