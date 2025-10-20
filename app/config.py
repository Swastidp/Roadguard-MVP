"""
Configuration settings for RoadGuard application by Team Autono Minds.
All paths, constants, and team information used across the application.

Team: Autono Minds | VW Hackathon 2025
Model: Custom YOLOv11 + SE Attention (50.56% mAP@0.5)
"""

from pathlib import Path
import os

# ==================== TEAM INFORMATION ====================

TEAM_INFO = {
    'name': 'Autono Minds',
    'hackathon': 'VW Hackathon 2025',
    'track': 'Smart Mobility & Road Safety',
    'project': 'RoadGuard - AI-Powered Road Hazard Detection',
    'github': 'https://github.com/Swastidp/Roadguard-MVP',
    'members': [
        'Swastidip Maji - Team Lead & ML Engineer',
        'Member 2 - Full Stack Developer', 
        'Member 3 - Computer Vision Specialist'
    ],
    'contact': 'swastidip2004@gmail.com'
}

# ==================== PATH CONFIGURATION ====================

# Base directory (project root)
BASE_DIR = Path(__file__).parent.parent

# Model paths - Team Autono Minds' trained models
MODEL_PATH_PT = BASE_DIR / "models" / "best.pt"  # Your YOLOv11 + SE Attention model
MODEL_PATH_TFLITE = BASE_DIR / "models" / "best_int8.tflite"
PRIVACY_MODEL_PATH = "yolov8n.pt"  # Will auto-download if not found

# Database path
DB_PATH = BASE_DIR / "data" / "hazards.db"

# Data directories
DATA_DIR = BASE_DIR / "data"
SAMPLE_VIDEOS_DIR = DATA_DIR / "sample_videos"
TEST_IMAGES_DIR = DATA_DIR / "test_images"

# ==================== MAP SETTINGS ====================

# Default map center (New Delhi, India - VW Hackathon location)
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

# Class names - Based on Team Autono Minds' YOLOv11 training
# Order MUST match your training configuration
CLASS_NAMES = [
    'longitudinal_crack',    # Class 0 - 59.84% mAP@0.5
    'transverse_crack',      # Class 1 - 69.97% mAP@0.5 (Best performing)
    'alligator_crack',       # Class 2 - 10.10% mAP@0.5 (Challenging)
    'pothole',               # Class 3 - 62.34% mAP@0.5
    'other_corruption'       # Class 4 - 0.0% mAP@0.5 (Not in validation)
]

# Dictionary version for modules that import DETECTION_SETTINGS
DETECTION_SETTINGS = {
    'confidence_threshold': CONFIDENCE_THRESHOLD,
    'iou_threshold': IOU_THRESHOLD,
    'img_size': IMG_SIZE,
    'class_names': CLASS_NAMES
}

# ==================== MODEL PERFORMANCE - TEAM AUTONO MINDS ====================

# Your actual YOLOv11 + SE Attention training results
TRAINING_RESULTS = {
    'model_info': {
        'architecture': 'YOLOv11n + Squeeze-and-Excitation Attention',
        'base_model': 'YOLOv11n',
        'enhancements': ['SE Attention blocks', 'Custom class weights', 'Extended training'],
        'model_size_mb': 5.45,
        'parameters': '2.6M (approx)',
        'training_framework': 'Ultralytics YOLOv11 (v8.3.217)'
    },
    'overall_performance': {
        'map50': 50.56,          # mAP@IoU=0.5
        'map50_95': 25.04,       # mAP@IoU=0.5:0.95 (COCO standard)
        'precision': 61.24,      # Overall precision
        'recall': 46.55,         # Overall recall
        'f1_score': 52.89,       # Calculated F1 score
        'inference_speed_ms': 45  # Approximate inference time
    },
    'per_class_performance': {
        'longitudinal_crack': {
            'map50': 59.84,
            'precision': 65.2,
            'recall': 54.8,
            'difficulty': 'Medium'
        },
        'transverse_crack': {
            'map50': 69.97,      # Best performing class
            'precision': 72.1,
            'recall': 67.3,
            'difficulty': 'Easy'
        },
        'alligator_crack': {
            'map50': 10.10,      # Most challenging class
            'precision': 15.5,
            'recall': 8.7,
            'difficulty': 'Very Hard'
        },
        'pothole': {
            'map50': 62.34,
            'precision': 68.9,
            'recall': 58.1,
            'difficulty': 'Medium'
        },
        'other_corruption': {
            'map50': 0.0,        # Not in validation set
            'precision': 0.0,
            'recall': 0.0,
            'difficulty': 'Unknown'
        }
    },
    'training_config': {
        'epochs': 65,
        'dataset_images': 6439,     # Training images
        'validation_images': 1619,  # Validation images
        'batch_size': 20,           # Limited by RTX 3050 4GB VRAM
        'optimizer': 'AdamW',
        'learning_rate': 0.002,
        'lr_scheduler': 'cosine',
        'class_loss_weight': 3.0,   # Increased for class imbalance
        'box_loss_weight': 7.5,
        'device': 'RTX 3050 Laptop GPU (4GB VRAM)',
        'training_time': '~2 hours',
        'patience': 30,             # Early stopping patience
        'augmentations': ['mosaic', 'mixup', 'rotation', 'scaling']
    },
    'dataset_info': {
        'total_images': 8058,
        'train_split': 0.8,         # 80% training
        'val_split': 0.2,           # 20% validation
        'image_resolution': '640x640',
        'annotation_format': 'YOLO',
        'classes': 5,
        'data_source': 'Road damage dataset + custom annotations'
    }
}

# ==================== ALERT SETTINGS ====================

# Warning distance calculation parameters
WARNING_DISTANCE_MIN = 50  # Minimum warning distance in meters
REACTION_TIME = 1.5  # Average driver reaction time in seconds
FRICTION_COEFFICIENT = 0.7  # Road friction coefficient (dry asphalt)

# Severity thresholds for hazard classification (updated for Indian roads)
SEVERITY_THRESHOLDS = {
    'critical': {'min_area': 1.2, 'priority': 1},  # Large potholes, severe damage
    'high': {'min_area': 0.6, 'priority': 2},      # Medium potholes, alligator cracks
    'medium': {'min_area': 0.3, 'priority': 3},    # Small potholes, transverse cracks
    'low': {'min_area': 0.0, 'priority': 4}        # Longitudinal cracks
}

# Severity multipliers for warning distance calculation
SEVERITY_MULTIPLIERS = {
    'critical': 1.8,    # Increased for critical hazards
    'high': 1.4,        # High priority hazards
    'medium': 1.0,      # Standard calculation
    'low': 0.7          # Reduced for minor hazards
}

# Class-specific severity mapping (based on Team Autono Minds' analysis)
CLASS_SEVERITY_MAPPING = {
    'longitudinal_crack': 'low',      # Usually minor, parallel to traffic
    'transverse_crack': 'medium',     # Cross traffic, more disruptive
    'alligator_crack': 'high',        # Indicates structural failure
    'pothole': 'high',                # Direct vehicle damage risk
    'other_corruption': 'medium'      # General road surface issues
}

# Dictionary version for modules that import ALERT_SETTINGS
ALERT_SETTINGS = {
    'warning_distance_min': WARNING_DISTANCE_MIN,
    'reaction_time': REACTION_TIME,
    'friction_coefficient': FRICTION_COEFFICIENT,
    'severity_thresholds': SEVERITY_THRESHOLDS,
    'severity_multipliers': SEVERITY_MULTIPLIERS,
    'class_severity_mapping': CLASS_SEVERITY_MAPPING
}

# ==================== PRIVACY SETTINGS ====================

# Anonymization settings for GDPR compliance
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

# Severity colors for visualization (Team Autono Minds color scheme)
SEVERITY_COLORS = {
    'critical': '#DC2626',  # Red - Immediate danger
    'high': '#EA580C',      # Dark orange - High priority
    'medium': '#D97706',    # Orange - Moderate priority
    'low': '#65A30D'        # Green - Low priority
}

# Map marker colors (matching severity colors)
MAP_COLORS = {
    'critical': 'red',
    'high': 'orange', 
    'medium': 'yellow',
    'low': 'green'
}

# Team Autono Minds brand colors
BRAND_COLORS = {
    'primary': '#1E40AF',      # Team blue
    'secondary': '#F59E0B',    # Team gold/orange
    'success': '#10B981',      # Success green
    'warning': '#F59E0B',      # Warning orange
    'error': '#EF4444',        # Error red
    'text_primary': '#1E293B', # Dark text
    'text_secondary': '#64748B' # Light text
}

# Dictionary version
UI_SETTINGS = {
    'severity_colors': SEVERITY_COLORS,
    'map_colors': MAP_COLORS,
    'brand_colors': BRAND_COLORS
}

# ==================== DATABASE SETTINGS ====================

# Deduplication settings for spatial clustering
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

# Video processing limits
MAX_FRAMES_TO_PROCESS = 100  # Maximum frames for demo video processing
FRAME_SKIP = 1  # Process every Nth frame (1 = process all)

# Caching settings
ENABLE_MODEL_CACHE = True
CACHE_TTL_SECONDS = 3600  # Time to live for cached resources

# Benchmarking (Team Autono Minds performance targets)
PERFORMANCE_TARGETS = {
    'inference_time_ms': 50,    # Target inference time
    'fps_realtime': 20,         # Target FPS for real-time processing
    'memory_usage_gb': 2.0,     # Maximum memory usage
    'cpu_usage_percent': 80     # Maximum CPU usage
}

# Dictionary version
PERFORMANCE_SETTINGS = {
    'max_frames_to_process': MAX_FRAMES_TO_PROCESS,
    'frame_skip': FRAME_SKIP,
    'enable_model_cache': ENABLE_MODEL_CACHE,
    'cache_ttl_seconds': CACHE_TTL_SECONDS,
    'performance_targets': PERFORMANCE_TARGETS
}

# ==================== HACKATHON SPECIFIC SETTINGS ====================

# VW Hackathon 2025 specific configurations
HACKATHON_CONFIG = {
    'event': 'VW Hackathon 2025',
    'track': 'Smart Mobility & Road Safety', 
    'submission_deadline': '2025-10-22',
    'demo_requirements': {
        'max_demo_time': 300,      # 5 minutes max demo
        'required_features': [
            'Real-time detection',
            'Alert generation',
            'Privacy compliance',
            'Performance metrics'
        ]
    },
    'evaluation_criteria': {
        'technical_innovation': 30,     # 30% weight
        'implementation_quality': 25,   # 25% weight
        'business_impact': 20,          # 20% weight
        'presentation': 15,             # 15% weight
        'teamwork': 10                  # 10% weight
    }
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
            f"Team Autono Minds trained model not found. Please ensure:\n"
            f"  - {MODEL_PATH_PT} (YOLOv11 + SE Attention model)\n"
            f"  - {MODEL_PATH_TFLITE} (Quantized version - optional)"
        )


def get_team_performance_summary():
    """Get Team Autono Minds performance summary for display."""
    return {
        'team': TEAM_INFO['name'],
        'model': TRAINING_RESULTS['model_info']['architecture'],
        'performance': {
            'mAP@0.5': f"{TRAINING_RESULTS['overall_performance']['map50']:.2f}%",
            'mAP@0.5:0.95': f"{TRAINING_RESULTS['overall_performance']['map50_95']:.2f}%",
            'Best Class': f"Transverse Crack ({TRAINING_RESULTS['per_class_performance']['transverse_crack']['map50']:.2f}%)"
        },
        'training': {
            'Epochs': TRAINING_RESULTS['training_config']['epochs'],
            'Dataset': f"{TRAINING_RESULTS['training_config']['dataset_images']} images",
            'Architecture': 'YOLOv11n + SE Attention'
        }
    }


def get_config_dict():
    """Return all configuration as a single dictionary."""
    return {
        'team': TEAM_INFO,
        'paths': {
            'base_dir': str(BASE_DIR),
            'model_pt': str(MODEL_PATH_PT),
            'model_tflite': str(MODEL_PATH_TFLITE),
            'privacy_model': PRIVACY_MODEL_PATH,
            'database': str(DB_PATH),
        },
        'training_results': TRAINING_RESULTS,
        'map': MAP_SETTINGS,
        'detection': DETECTION_SETTINGS,
        'alert': ALERT_SETTINGS,
        'privacy': PRIVACY_SETTINGS,
        'ui': UI_SETTINGS,
        'database': DATABASE_SETTINGS,
        'performance': PERFORMANCE_SETTINGS,
        'hackathon': HACKATHON_CONFIG,
    }


def get_class_info(class_id: int):
    """Get detailed information about a specific class."""
    if 0 <= class_id < len(CLASS_NAMES):
        class_name = CLASS_NAMES[class_id]
        class_perf = TRAINING_RESULTS['per_class_performance'].get(class_name, {})
        
        return {
            'id': class_id,
            'name': class_name,
            'map50': class_perf.get('map50', 0.0),
            'precision': class_perf.get('precision', 0.0),
            'recall': class_perf.get('recall', 0.0),
            'difficulty': class_perf.get('difficulty', 'Unknown'),
            'severity': CLASS_SEVERITY_MAPPING.get(class_name, 'medium')
        }
    else:
        return None


# ==================== INITIALIZATION ====================

# Create directories on import
ensure_directories()

# ==================== EXPORTS ====================

__all__ = [
    # Team information
    'TEAM_INFO', 'TRAINING_RESULTS', 'HACKATHON_CONFIG',
    
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
    'SEVERITY_THRESHOLDS', 'SEVERITY_MULTIPLIERS', 'CLASS_SEVERITY_MAPPING',
    'ALERT_SETTINGS',
    
    # Privacy settings (both individual and dict)
    'BLUR_KERNEL_SIZE', 'BLUR_SIGMA', 'PIXELATE_FACTOR', 'PRIVACY_SETTINGS',
    
    # UI settings (both individual and dict)
    'SEVERITY_COLORS', 'MAP_COLORS', 'BRAND_COLORS', 'UI_SETTINGS',
    
    # Database settings (both individual and dict)
    'DEDUP_RADIUS_METERS', 'MIN_SAMPLES_FOR_CLUSTER',
    'CONFIDENCE_DECAY_DAYS', 'REPAIR_THRESHOLD_DAYS', 'DATABASE_SETTINGS',
    
    # Performance settings (both individual and dict)
    'MAX_FRAMES_TO_PROCESS', 'FRAME_SKIP', 'ENABLE_MODEL_CACHE',
    'CACHE_TTL_SECONDS', 'PERFORMANCE_TARGETS', 'PERFORMANCE_SETTINGS',
    
    # Helper functions
    'ensure_directories', 'get_model_path', 'get_config_dict',
    'get_team_performance_summary', 'get_class_info'
]
