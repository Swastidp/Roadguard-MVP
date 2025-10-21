"""
Utility functions for RoadGuard application.
Cloud deployment compatible with proper OpenCV handling.

Team: Autono Minds | VW Hackathon 2025
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union, Tuple, Any, List, Dict
import numpy as np
from PIL import Image
import streamlit as st

# Configure for headless/cloud deployment BEFORE importing OpenCV
def setup_opencv_for_cloud():
    """Configure OpenCV for cloud/headless deployment."""
    # Set environment variables for headless operation
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    os.environ['DISPLAY'] = ':99'
    
    # Disable GUI backends
    import matplotlib
    matplotlib.use('Agg')

# Setup environment before OpenCV import
setup_opencv_for_cloud()

# Import OpenCV with proper error handling for cloud deployment
try:
    import cv2
    # Verify OpenCV is working in headless mode
    cv2.setUseOptimized(True)
    print("✅ OpenCV configured for cloud deployment")
except ImportError as e:
    st.error(f"OpenCV import failed: {e}")
    st.info("Installing opencv-python-headless might resolve this issue")
    raise ImportError(f"OpenCV not available for cloud deployment: {e}")
except Exception as e:
    st.warning(f"OpenCV configuration warning: {e}")
    import cv2  # Try basic import anyway

import pandas as pd
import tempfile
import base64
from io import BytesIO


# ============================================================================
# Image Processing Utilities
# ============================================================================

def load_image_from_upload(uploaded_file) -> Optional[np.ndarray]:
    """
    Load image from Streamlit file upload.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Image as BGR numpy array or None if failed
    """
    try:
        # Read bytes
        file_bytes = uploaded_file.read()
        
        # Convert to PIL Image
        pil_image = Image.open(BytesIO(file_bytes))
        
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert PIL to OpenCV (RGB to BGR)
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return opencv_image
        
    except Exception as e:
        st.error(f"Failed to load image: {str(e)}")
        return None


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to RGB for display."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to BGR."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def resize_image(image: np.ndarray, max_width: int = 800, max_height: int = 600) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image
        max_width: Maximum width
        max_height: Maximum height
        
    Returns:
        Resized image
    """
    height, width = image.shape[:2]
    
    # Calculate scale factor
    scale_w = max_width / width
    scale_h = max_height / height
    scale = min(scale_w, scale_h, 1.0)  # Don't upscale
    
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return image


def image_to_bytes(image: np.ndarray, format: str = 'JPEG', quality: int = 90) -> bytes:
    """
    Convert OpenCV image to bytes for download.
    
    Args:
        image: OpenCV image (BGR)
        format: Output format ('JPEG', 'PNG')
        quality: JPEG quality (1-100)
        
    Returns:
        Image bytes
    """
    # Convert BGR to RGB for PIL
    rgb_image = bgr_to_rgb(image)
    pil_image = Image.fromarray(rgb_image)
    
    # Save to bytes
    img_bytes = BytesIO()
    if format.upper() == 'JPEG':
        pil_image.save(img_bytes, format='JPEG', quality=quality)
    else:
        pil_image.save(img_bytes, format=format)
    
    return img_bytes.getvalue()


# ============================================================================
# File Management
# ============================================================================

def save_uploaded_file(uploaded_file, save_dir: Path, prefix: str = "upload") -> Optional[Path]:
    """
    Save uploaded file to temporary location.
    
    Args:
        uploaded_file: Streamlit UploadedFile
        save_dir: Directory to save file
        prefix: Filename prefix
        
    Returns:
        Path to saved file or None
    """
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        file_extension = Path(uploaded_file.name).suffix
        temp_filename = f"{prefix}_{uploaded_file.name}"
        temp_path = save_dir / temp_filename
        
        # Save file
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return temp_path
        
    except Exception as e:
        st.error(f"Failed to save file: {str(e)}")
        return None


def cleanup_temp_files(temp_dir: Path, max_age_hours: int = 24):
    """
    Clean up old temporary files.
    
    Args:
        temp_dir: Temporary directory
        max_age_hours: Maximum age in hours
    """
    try:
        if not temp_dir.exists():
            return
        
        import time
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for file_path in temp_dir.iterdir():
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    file_path.unlink()
        
    except Exception as e:
        print(f"Error cleaning temp files: {e}")


# ============================================================================
# Data Processing
# ============================================================================

def normalize_coordinates(bbox: List[float], img_width: int, img_height: int) -> List[int]:
    """
    Normalize and clamp bounding box coordinates.
    
    Args:
        bbox: [x1, y1, x2, y2] coordinates
        img_width: Image width
        img_height: Image height
        
    Returns:
        Normalized coordinates as integers
    """
    x1, y1, x2, y2 = bbox
    
    # Clamp to image bounds
    x1 = max(0, min(int(x1), img_width - 1))
    y1 = max(0, min(int(y1), img_height - 1))
    x2 = max(0, min(int(x2), img_width - 1))
    y2 = max(0, min(int(y2), img_height - 1))
    
    # Ensure valid rectangle
    if x1 >= x2:
        x2 = x1 + 1
    if y1 >= y2:
        y2 = y1 + 1
    
    return [x1, y1, x2, y2]


def calculate_bbox_area(bbox: List[float]) -> float:
    """Calculate bounding box area."""
    x1, y1, x2, y2 = bbox
    return abs((x2 - x1) * (y2 - y1))


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate Intersection over Union of two bounding boxes.
    
    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]
        
    Returns:
        IoU value (0-1)
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


# ============================================================================
# Performance Monitoring
# ============================================================================

def measure_inference_time(func, *args, **kwargs) -> Tuple[Any, float]:
    """
    Measure function execution time.
    
    Args:
        func: Function to measure
        *args, **kwargs: Function arguments
        
    Returns:
        (result, execution_time_ms)
    """
    import time
    
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    
    execution_time_ms = (end_time - start_time) * 1000
    return result, execution_time_ms


def get_system_info() -> Dict[str, Any]:
    """Get system information for diagnostics."""
    import platform
    import psutil
    
    try:
        info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': os.cpu_count(),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
            'opencv_version': cv2.__version__,
        }
        
        # GPU info if available
        try:
            import torch
            info['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info['cuda_device_count'] = torch.cuda.device_count()
                info['cuda_device_name'] = torch.cuda.get_device_name()
        except ImportError:
            info['cuda_available'] = False
        
        return info
        
    except Exception as e:
        return {'error': str(e)}


# ============================================================================
# Validation
# ============================================================================

def validate_image(image: np.ndarray) -> Tuple[bool, str]:
    """
    Validate image for processing.
    
    Args:
        image: OpenCV image
        
    Returns:
        (is_valid, error_message)
    """
    if image is None:
        return False, "Image is None"
    
    if not isinstance(image, np.ndarray):
        return False, "Image is not a numpy array"
    
    if len(image.shape) not in [2, 3]:
        return False, f"Invalid image dimensions: {image.shape}"
    
    if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
        return False, f"Invalid number of channels: {image.shape[2]}"
    
    if image.size == 0:
        return False, "Empty image"
    
    # Check reasonable size limits
    height, width = image.shape[:2]
    if width < 10 or height < 10:
        return False, f"Image too small: {width}x{height}"
    
    if width > 10000 or height > 10000:
        return False, f"Image too large: {width}x{height}"
    
    return True, "Valid"


def validate_detection_results(detections: Dict) -> Tuple[bool, str]:
    """
    Validate detection results structure.
    
    Args:
        detections: Detection results dictionary
        
    Returns:
        (is_valid, error_message)
    """
    required_keys = ['boxes', 'classes', 'confidences', 'detection_count']
    
    for key in required_keys:
        if key not in detections:
            return False, f"Missing required key: {key}"
    
    boxes = detections['boxes']
    classes = detections['classes']
    confidences = detections['confidences']
    
    if len(boxes) != len(classes) or len(boxes) != len(confidences):
        return False, "Inconsistent detection array lengths"
    
    # Validate confidence values
    for conf in confidences:
        if not (0.0 <= conf <= 1.0):
            return False, f"Invalid confidence value: {conf}"
    
    return True, "Valid"


# ============================================================================
# Export Functions
# ============================================================================

__all__ = [
    # Image processing
    'load_image_from_upload',
    'bgr_to_rgb',
    'rgb_to_bgr',
    'resize_image',
    'image_to_bytes',
    
    # File management
    'save_uploaded_file',
    'cleanup_temp_files',
    
    # Data processing
    'normalize_coordinates',
    'calculate_bbox_area',
    'calculate_iou',
    
    # Performance
    'measure_inference_time',
    'get_system_info',
    
    # Validation
    'validate_image',
    'validate_detection_results',
]
