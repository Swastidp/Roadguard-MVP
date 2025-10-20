"""
Configuration settings for RoadGuard application.
All paths and constants used across the application.
"""

from pathlib import Path
import os

# ==================== PATH CONFIGURATION ====================

# Base directory (project root)
BASE_DIR = Path(__file__).parent.parent

# Model paths
MODEL_PATH_PT = BASE_DIR / "models" / "best.pt"
MODEL_PATH_TFLITE = BASE_DIR / "models" / "best_int8.tflite"
PRIVACY_MODEL_PATH = "yolov8n.pt"  # Will auto-download if not found

# Database path
DB_PATH = BASE_DIR / "data" / "hazards.db"

# Data directories
DATA_DIR = BASE_DIR / "data"
SAMPLE_VIDEOS_DIR = DATA_DIR / "sample_videos"
TEST_IMAGES_DIR = DATA_DIR / "test_images"

# ==================== MAP SETTINGS ====================

# Default map center (New Delhi, India)
INITIAL_LAT = 28.6139
INITIAL_LON = 77.2090
INITIAL_ZOOM = 12

# Dictionary version for modules that import MAP_SETTINGS
MAP_SETTINGS = {
    'center_lat': INITIAL_LAT,
    'center_lon': INITIAL_LON,
    'zoom': INITIAL_ZOOM
}

# ==================== DETECTION SETTINGS ====================

# Model inference settings
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detection
IOU_THRESHOLD = 0.45  # Intersection over Union threshold for NMS
IMG_SIZE = 640  # Input image size for YOLO

# Class names (must match training order)
CLASS_NAMES = [
    'pothole',
    'longitudinal_crack',
    'transverse_crack',
    'alligator_crack'
]

# Dictionary version for modules that import DETECTION_SETTINGS
DETECTION_SETTINGS = {
    'confidence_threshold': CONFIDENCE_THRESHOLD,
    'iou_threshold': IOU_THRESHOLD,
    'img_size': IMG_SIZE,
    'class_names': CLASS_NAMES
}

# ==================== ALERT SETTINGS ====================

# Warning distance calculation parameters
WARNING_DISTANCE_MIN = 50  # Minimum warning distance in meters
REACTION_TIME = 1.5  # Average driver reaction time in seconds
FRICTION_COEFFICIENT = 0.7  # Road friction coefficient (dry asphalt)

# Severity thresholds for hazard classification
SEVERITY_THRESHOLDS = {
    'critical': {'min_area': 1.0, 'priority': 1},  # Large potholes, severe damage
    'high': {'min_area': 0.5, 'priority': 2},
    'medium': {'min_area': 0.2, 'priority': 3},
    'low': {'min_area': 0.0, 'priority': 4}
}

# Severity multipliers for warning distance calculation
SEVERITY_MULTIPLIERS = {
    'critical': 1.5,
    'high': 1.2,
    'medium': 1.0,
    'low': 0.8
}

# Dictionary version for modules that import ALERT_SETTINGS
ALERT_SETTINGS = {
    'warning_distance_min': WARNING_DISTANCE_MIN,
    'reaction_time': REACTION_TIME,
    'friction_coefficient': FRICTION_COEFFICIENT,
    'severity_thresholds': SEVERITY_THRESHOLDS,
    'severity_multipliers': SEVERITY_MULTIPLIERS
}

# ==================== PRIVACY SETTINGS ====================

# Anonymization settings
BLUR_KERNEL_SIZE = (99, 99)  # Gaussian blur kernel
BLUR_SIGMA = 30  # Gaussian blur standard deviation
PIXELATE_FACTOR = 10  # Downscale factor for pixelation

# Dictionary version
PRIVACY_SETTINGS = {
    'blur_kernel_size': BLUR_KERNEL_SIZE,
    'blur_sigma': BLUR_SIGMA,
    'pixelate_factor': PIXELATE_FACTOR
}

# ==================== UI SETTINGS ====================

# Severity colors for visualization (hex codes)
SEVERITY_COLORS = {
    'critical': '#FF0000',  # Red
    'high': '#FF8C00',      # Dark orange
    'medium': '#FFD700',    # Gold
    'low': '#90EE90'        # Light green
}

# Map marker colors
MAP_COLORS = {
    'critical': 'red',
    'high': 'orange',
    'medium': 'yellow',
    'low': 'green'
}

# Dictionary version
UI_SETTINGS = {
    'severity_colors': SEVERITY_COLORS,
    'map_colors': MAP_COLORS
}

# ==================== DATABASE SETTINGS ====================

# Deduplication settings
DEDUP_RADIUS_METERS = 10  # Cluster hazards within this radius
MIN_SAMPLES_FOR_CLUSTER = 2  # Minimum reports to confirm hazard

# Confidence decay settings
CONFIDENCE_DECAY_DAYS = 30  # Days for confidence to decay
REPAIR_THRESHOLD_DAYS = 60  # Days before marking as possibly repaired

# Dictionary version
DATABASE_SETTINGS = {
    'dedup_radius_meters': DEDUP_RADIUS_METERS,
    'min_samples_for_cluster': MIN_SAMPLES_FOR_CLUSTER,
    'confidence_decay_days': CONFIDENCE_DECAY_DAYS,
    'repair_threshold_days': REPAIR_THRESHOLD_DAYS
}

# ==================== PERFORMANCE SETTINGS ====================

# Video processing
MAX_FRAMES_TO_PROCESS = 100  # Maximum frames for demo video processing
FRAME_SKIP = 1  # Process every Nth frame (1 = process all)

# Caching
ENABLE_MODEL_CACHE = True
CACHE_TTL_SECONDS = 3600  # Time to live for cached resources

# Dictionary version
PERFORMANCE_SETTINGS = {
    'max_frames_to_process': MAX_FRAMES_TO_PROCESS,
    'frame_skip': FRAME_SKIP,
    'enable_model_cache': ENABLE_MODEL_CACHE,
    'cache_ttl_seconds': CACHE_TTL_SECONDS
}

# ==================== HELPER FUNCTIONS ====================

def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        BASE_DIR / "models",
        DATA_DIR,
        SAMPLE_VIDEOS_DIR,
        TEST_IMAGES_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def get_model_path(use_tflite=False):
    """Get the appropriate model path based on availability."""
    if use_tflite and MODEL_PATH_TFLITE.exists():
        return str(MODEL_PATH_TFLITE)
    elif MODEL_PATH_PT.exists():
        return str(MODEL_PATH_PT)
    else:
        raise FileNotFoundError(
            f"No model found. Please download models to:\n"
            f"  - {MODEL_PATH_PT}\n"
            f"  - {MODEL_PATH_TFLITE}"
        )

def get_config_dict():
    """Return all configuration as a single dictionary."""
    return {
        'paths': {
            'base_dir': str(BASE_DIR),
            'model_pt': str(MODEL_PATH_PT),
            'model_tflite': str(MODEL_PATH_TFLITE),
            'privacy_model': PRIVACY_MODEL_PATH,
            'database': str(DB_PATH),
        },
        'map': MAP_SETTINGS,
        'detection': DETECTION_SETTINGS,
        'alert': ALERT_SETTINGS,
        'privacy': PRIVACY_SETTINGS,
        'ui': UI_SETTINGS,
        'database': DATABASE_SETTINGS,
        'performance': PERFORMANCE_SETTINGS,
    }

# ==================== INITIALIZATION ====================

# Create directories on import
ensure_directories()

# ==================== EXPORTS ====================

__all__ = [
    # Paths
    'BASE_DIR', 'MODEL_PATH_PT', 'MODEL_PATH_TFLITE', 'PRIVACY_MODEL_PATH',
    'DB_PATH', 'DATA_DIR', 'SAMPLE_VIDEOS_DIR', 'TEST_IMAGES_DIR',
    
    # Map settings (both individual and dict)
    'INITIAL_LAT', 'INITIAL_LON', 'INITIAL_ZOOM', 'MAP_SETTINGS',
    
    # Detection settings (both individual and dict)
    'CONFIDENCE_THRESHOLD', 'IOU_THRESHOLD', 'IMG_SIZE', 'CLASS_NAMES',
    'DETECTION_SETTINGS',
    
    # Alert settings (both individual and dict)
    'WARNING_DISTANCE_MIN', 'REACTION_TIME', 'FRICTION_COEFFICIENT',
    'SEVERITY_THRESHOLDS', 'SEVERITY_MULTIPLIERS', 'ALERT_SETTINGS',
    
    # Privacy settings (both individual and dict)
    'BLUR_KERNEL_SIZE', 'BLUR_SIGMA', 'PIXELATE_FACTOR', 'PRIVACY_SETTINGS',
    
    # UI settings (both individual and dict)
    'SEVERITY_COLORS', 'MAP_COLORS', 'UI_SETTINGS',
    
    # Database settings (both individual and dict)
    'DEDUP_RADIUS_METERS', 'MIN_SAMPLES_FOR_CLUSTER',
    'CONFIDENCE_DECAY_DAYS', 'REPAIR_THRESHOLD_DAYS', 'DATABASE_SETTINGS',
    
    # Performance settings (both individual and dict)
    'MAX_FRAMES_TO_PROCESS', 'FRAME_SKIP', 'ENABLE_MODEL_CACHE',
    'CACHE_TTL_SECONDS', 'PERFORMANCE_SETTINGS',
    
    # Helper functions
    'ensure_directories', 'get_model_path', 'get_config_dict'
]
