"""
Detection module for YOLO-based road hazard detection.

This module provides functions for loading YOLO models, running inference,
drawing detections, and classifying hazard severity.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Union
import streamlit as st

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics package is required. Install with: pip install ultralytics")

from ..config import (
    MODEL_PATH_PT,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    IMG_SIZE,
    CLASS_NAMES,
    SEVERITY_COLORS
)


# ============================================================================
# Model Loading
# ============================================================================

@st.cache_resource
def load_model(model_path: Union[str, Path] = MODEL_PATH_PT) -> Optional[YOLO]:
    """
    Load YOLO model with caching for efficient reuse.
    
    Args:
        model_path: Path to the YOLO model file (.pt format)
        
    Returns:
        YOLO model object if successful, None otherwise
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: For other loading errors
    """
    try:
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at: {model_path}\n"
                f"Please ensure the model is placed in the correct location."
            )
        
        # Load YOLO model
        model = YOLO(str(model_path))
        
        # Verify model loaded successfully
        if model is None:
            raise ValueError("Model loaded but returned None")
        
        print(f"✅ Model loaded successfully from: {model_path}")
        return model
        
    except FileNotFoundError as e:
        print(f"❌ Model file not found: {e}")
        st.error(f"Model file not found: {model_path}")
        return None
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        st.error(f"Error loading model: {str(e)}")
        return None


# ============================================================================
# Hazard Detection
# ============================================================================

def detect_hazards(
    model: YOLO,
    image: np.ndarray,
    conf_threshold: float = CONFIDENCE_THRESHOLD,
    iou_threshold: float = IOU_THRESHOLD,
    img_size: int = IMG_SIZE
) -> Dict[str, any]:
    """
    Detect road hazards in an image using YOLO model.
    
    Args:
        model: Loaded YOLO model object
        image: Input image as numpy array (BGR format)
        conf_threshold: Confidence threshold for detections (0.0-1.0)
        iou_threshold: IoU threshold for Non-Maximum Suppression (0.0-1.0)
        img_size: Input image size for model inference
        
    Returns:
        Dictionary containing:
            - boxes: List of bounding boxes [[x1, y1, x2, y2], ...]
            - classes: List of class IDs
            - confidences: List of confidence scores
            - raw_results: Raw YOLO results object
            - detection_count: Number of detections
            
    Raises:
        ValueError: If model is None or image is invalid
    """
    if model is None:
        raise ValueError("Model is None. Please load a valid model first.")
    
    if image is None or image.size == 0:
        raise ValueError("Invalid image provided for detection.")
    
    try:
        # Run inference
        results = model(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=img_size,
            verbose=False
        )
        
        # Extract detection results
        if len(results) == 0:
            return {
                'boxes': [],
                'classes': [],
                'confidences': [],
                'raw_results': None,
                'detection_count': 0
            }
        
        result = results[0]
        
        # Extract boxes, classes, and confidences
        boxes = []
        classes = []
        confidences = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy().tolist()  # [[x1, y1, x2, y2], ...]
            classes = result.boxes.cls.cpu().numpy().astype(int).tolist()
            confidences = result.boxes.conf.cpu().numpy().tolist()
        
        return {
            'boxes': boxes,
            'classes': classes,
            'confidences': confidences,
            'raw_results': result,
            'detection_count': len(boxes)
        }
        
    except Exception as e:
        print(f"❌ Error during detection: {e}")
        raise RuntimeError(f"Detection failed: {str(e)}")


# ============================================================================
# Visualization
# ============================================================================

def draw_detections(
    image: np.ndarray,
    detections: Dict[str, any],
    class_names: List[str] = CLASS_NAMES,
    draw_confidence: bool = True,
    line_thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding boxes and labels on image for detected hazards.
    
    Args:
        image: Input image as numpy array (BGR format)
        detections: Detection results from detect_hazards()
        class_names: List of class names corresponding to class IDs
        draw_confidence: Whether to display confidence scores
        line_thickness: Thickness of bounding box lines
        
    Returns:
        Annotated image with drawn detections
        
    Raises:
        ValueError: If image or detections are invalid
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image provided for annotation.")
    
    # Clone image to avoid modifying original
    annotated_image = image.copy()
    
    boxes = detections.get('boxes', [])
    classes = detections.get('classes', [])
    confidences = detections.get('confidences', [])
    
    if len(boxes) == 0:
        return annotated_image
    
    # Get image dimensions
    img_height, img_width = image.shape[:2]
    
    # Draw each detection
    for box, cls_id, conf in zip(boxes, classes, confidences):
        try:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            x2 = max(0, min(x2, img_width - 1))
            y2 = max(0, min(y2, img_height - 1))
            
            # Calculate bbox area for severity classification
            bbox_area = (x2 - x1) * (y2 - y1)
            severity = classify_severity(cls_id, bbox_area)
            
            # Get color based on severity
            color_hex = SEVERITY_COLORS.get(severity, "#3B82F6")
            color_bgr = hex_to_bgr(color_hex)
            
            # Draw bounding box
            cv2.rectangle(
                annotated_image,
                (x1, y1),
                (x2, y2),
                color_bgr,
                line_thickness
            )
            
            # Prepare label text
            class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
            
            if draw_confidence:
                label = f"{class_name}: {conf:.2f}"
            else:
                label = class_name
            
            # Calculate label background size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )
            
            # Draw label background
            label_y1 = max(y1 - text_height - 10, 0)
            label_y2 = y1
            cv2.rectangle(
                annotated_image,
                (x1, label_y1),
                (x1 + text_width + 10, label_y2),
                color_bgr,
                -1  # Filled rectangle
            )
            
            # Draw label text
            cv2.putText(
                annotated_image,
                label,
                (x1 + 5, y1 - 5),
                font,
                font_scale,
                (255, 255, 255),  # White text
                font_thickness,
                cv2.LINE_AA
            )
            
            # Draw severity badge
            severity_label = f"{severity.upper()}"
            severity_y = y2 + 20
            cv2.putText(
                annotated_image,
                severity_label,
                (x1, severity_y),
                font,
                0.4,
                color_bgr,
                1,
                cv2.LINE_AA
            )
            
        except Exception as e:
            print(f"⚠️ Error drawing detection: {e}")
            continue
    
    return annotated_image


# ============================================================================
# Hazard Analysis
# ============================================================================

def calculate_hazard_size(
    bbox: List[float],
    image_shape: Tuple[int, int],
    camera_height: float = 1.5,  # meters
    camera_fov: float = 60.0  # degrees
) -> float:
    """
    Estimate physical size of hazard based on bounding box dimensions.
    
    This is a simplified estimation assuming:
    - Camera is mounted at fixed height
    - Hazard is on ground plane
    - Known camera field of view
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        image_shape: Image dimensions (height, width)
        camera_height: Camera mounting height in meters
        camera_fov: Camera field of view in degrees
        
    Returns:
        Estimated hazard area in square meters
    """
    try:
        x1, y1, x2, y2 = bbox
        img_height, img_width = image_shape
        
        # Calculate bbox dimensions in pixels
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # Calculate bbox area in pixels
        bbox_area_pixels = bbox_width * bbox_height
        
        # Calculate image area in pixels
        image_area_pixels = img_height * img_width
        
        # Estimate distance to hazard (simplified)
        # Further hazards appear smaller (higher y2 value = closer)
        distance_factor = (img_height - y2) / img_height
        estimated_distance = 5.0 + (distance_factor * 50.0)  # 5-55 meters
        
        # Calculate ground plane area visible at that distance
        fov_rad = np.deg2rad(camera_fov)
        ground_width = 2 * estimated_distance * np.tan(fov_rad / 2)
        
        # Estimate physical area
        pixel_to_meter_ratio = (ground_width * ground_width) / image_area_pixels
        hazard_area_m2 = bbox_area_pixels * pixel_to_meter_ratio
        
        return round(hazard_area_m2, 2)
        
    except Exception as e:
        print(f"⚠️ Error calculating hazard size: {e}")
        return 0.0


def classify_severity(
    class_id: int,
    bbox_area: float,
    size_threshold_large: float = 50000,  # pixels
    size_threshold_medium: float = 20000  # pixels
) -> str:
    """
    Classify hazard severity based on class type and size.
    
    Severity levels:
    - critical: Large potholes or extensive damage
    - high: Medium potholes or alligator cracks
    - medium: Small potholes or transverse cracks
    - low: Minor longitudinal cracks
    
    Args:
        class_id: Detection class ID (0=pothole, 1=longitudinal, 2=transverse, 3=alligator)
        bbox_area: Bounding box area in pixels
        size_threshold_large: Pixel area threshold for large hazards
        size_threshold_medium: Pixel area threshold for medium hazards
        
    Returns:
        Severity level as string: 'critical', 'high', 'medium', or 'low'
    """
    # Class-based severity mapping
    # 0: pothole, 1: longitudinal_crack, 2: transverse_crack, 3: alligator_crack
    
    if class_id == 0:  # Pothole
        if bbox_area > size_threshold_large:
            return 'critical'
        elif bbox_area > size_threshold_medium:
            return 'high'
        else:
            return 'medium'
    
    elif class_id == 3:  # Alligator crack (most severe crack type)
        if bbox_area > size_threshold_medium:
            return 'high'
        else:
            return 'medium'
    
    elif class_id == 2:  # Transverse crack
        if bbox_area > size_threshold_large:
            return 'medium'
        else:
            return 'low'
    
    elif class_id == 1:  # Longitudinal crack (least severe)
        if bbox_area > size_threshold_large:
            return 'medium'
        else:
            return 'low'
    
    else:  # Unknown class
        return 'low'


# ============================================================================
# Utility Functions
# ============================================================================

def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """
    Convert hex color code to BGR tuple for OpenCV.
    
    Args:
        hex_color: Hex color code (e.g., '#FF5733' or 'FF5733')
        
    Returns:
        BGR color tuple (B, G, R)
    """
    # Remove '#' if present
    hex_color = hex_color.lstrip('#')
    
    # Convert hex to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    # Return as BGR for OpenCV
    return (b, g, r)


def get_detection_summary(detections: Dict[str, any], class_names: List[str] = CLASS_NAMES) -> Dict[str, any]:
    """
    Generate summary statistics from detection results.
    
    Args:
        detections: Detection results from detect_hazards()
        class_names: List of class names
        
    Returns:
        Dictionary with summary statistics
    """
    boxes = detections.get('boxes', [])
    classes = detections.get('classes', [])
    confidences = detections.get('confidences', [])
    
    if len(boxes) == 0:
        return {
            'total_detections': 0,
            'class_counts': {},
            'avg_confidence': 0.0,
            'severity_counts': {}
        }
    
    # Count detections per class
    class_counts = {}
    for cls_id in classes:
        class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Calculate severity distribution
    severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
    for box, cls_id in zip(boxes, classes):
        x1, y1, x2, y2 = box
        bbox_area = (x2 - x1) * (y2 - y1)
        severity = classify_severity(cls_id, bbox_area)
        severity_counts[severity] += 1
    
    return {
        'total_detections': len(boxes),
        'class_counts': class_counts,
        'avg_confidence': round(np.mean(confidences), 3) if confidences else 0.0,
        'severity_counts': severity_counts,
        'max_confidence': round(max(confidences), 3) if confidences else 0.0,
        'min_confidence': round(min(confidences), 3) if confidences else 0.0
    }


# ============================================================================
# Export Functions
# ============================================================================

__all__ = [
    'load_model',
    'detect_hazards',
    'draw_detections',
    'calculate_hazard_size',
    'classify_severity',
    'hex_to_bgr',
    'get_detection_summary'
]
