"""
Detection module using YOLOv11 + SE Attention trained by Team Autono Minds.
Uses the custom trained best.pt model for road hazard detection.

Team: Autono Minds | VW Hackathon 2025
"""

"""
Detection module using YOLOv11 + SE Attention trained by Team Autono Minds.
Uses the custom trained best.pt model for road hazard detection.

Team: Autono Minds | VW Hackathon 2025
"""

import os
import sys

# Handle OpenCV import for cloud deployment
try:
    import cv2
    print("✅ OpenCV imported successfully")
except ImportError as e:
    print(f"⚠️ OpenCV import issue: {e}")
    # Try alternative import
    try:
        import cv2
        print("✅ OpenCV imported on retry")
    except Exception as e2:
        print(f"❌ OpenCV failed completely: {e2}")
        raise ImportError(f"OpenCV not available: {e2}")

# Set OpenCV to headless mode for cloud deployment
if 'STREAMLIT_SERVER_PORT' in os.environ or 'STREAMLIT_CLOUD' in os.environ:
    # Running on Streamlit Cloud
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'
    print("🌐 OpenCV configured for Streamlit Cloud deployment")


# Rest of your imports...


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
# Model Loading - Team Autono Minds' Trained YOLOv11 Model
# ============================================================================

@st.cache_resource
def load_model(model_path: Union[str, Path] = MODEL_PATH_PT) -> Optional[YOLO]:
    """
    Load Team Autono Minds' trained YOLOv11 + SE Attention model.
    
    Training Achievement:
    - Architecture: YOLOv11n + Squeeze-and-Excitation blocks
    - Performance: 50.56% mAP@0.5, 25.04% mAP@0.5:0.95
    - Dataset: 6,439 training images, 65 epochs
    - Classes: longitudinal_crack, transverse_crack, alligator_crack, pothole, other_corruption
    
    Args:
        model_path: Path to the trained model file (models/best.pt)
        
    Returns:
        YOLO model object if successful, None otherwise
    """
    try:
        model_path = Path(model_path)
        
        if not model_path.exists():
            st.error(f"❌ Trained model not found at: {model_path}")
            st.info("Please ensure your best.pt file is in the models/ directory")
            return None
        
        # Load Team Autono Minds' trained YOLOv11 model
        st.info("🔥 Loading Team Autono Minds' YOLOv11 + SE Attention model...")
        model = YOLO(str(model_path))
        
        # Display model info
        st.success("✅ Custom YOLOv11 model loaded successfully!")
        
        # Show Team Autono Minds' training achievements
        with st.expander("🏆 Team Autono Minds - Training Results", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **🎯 Model Architecture:**
                - YOLOv11n + SE Attention blocks
                - Custom trained for road hazards
                - 65 epochs training
                - RTX 3050 Laptop GPU
                
                **📊 Overall Performance:**
                - mAP@0.5: **50.56%**
                - mAP@0.5:0.95: **25.04%**
                - Precision: **61.24%**
                - Recall: **46.55%**
                """)
            
            with col2:
                st.markdown("""
                **🔍 Per-Class Performance:**
                - Transverse Crack: **69.97%** mAP@0.5
                - Pothole: **62.34%** mAP@0.5  
                - Longitudinal Crack: **59.84%** mAP@0.5
                - Alligator Crack: **10.10%** mAP@0.5
                
                **⚙️ Training Config:**
                - Optimizer: AdamW
                - Learning Rate: 0.002
                - Class Loss Weight: 3.0
                """)
        
        print(f"✅ Team Autono Minds' YOLOv11 model loaded from: {model_path}")
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        st.error(f"Error loading model: {str(e)}")
        return None


# ============================================================================
# Enhanced Detection with Team Info
# ============================================================================

def detect_hazards(
    model: YOLO,
    image: np.ndarray,
    conf_threshold: float = CONFIDENCE_THRESHOLD,
    iou_threshold: float = IOU_THRESHOLD,
    img_size: int = IMG_SIZE
) -> Dict[str, any]:
    """
    Detect road hazards using Team Autono Minds' YOLOv11 + SE Attention model.
    
    Args:
        model: Team Autono Minds' trained YOLOv11 model
        image: Input image as numpy array (BGR format)
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        img_size: Input image size
        
    Returns:
        Detection results with team and model information
    """
    if model is None:
        raise ValueError("Model is None. Please load Team Autono Minds' trained model first.")
    
    if image is None or image.size == 0:
        raise ValueError("Invalid image provided for detection.")
    
    try:
        # Run inference with trained model
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
                'detection_count': 0,
                'team': 'Autono Minds',
                'model_info': 'YOLOv11 + SE Attention (Custom Trained)',
                'performance_metrics': {
                    'map50': 50.56,
                    'map50_95': 25.04,
                    'training_epochs': 65
                }
            }
        
        result = results[0]
        
        boxes = []
        classes = []
        confidences = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy().tolist()
            classes = result.boxes.cls.cpu().numpy().astype(int).tolist()
            confidences = result.boxes.conf.cpu().numpy().tolist()
        
        return {
            'boxes': boxes,
            'classes': classes,
            'confidences': confidences,
            'raw_results': result,
            'detection_count': len(boxes),
            'team': 'Autono Minds',
            'model_info': 'YOLOv11 + SE Attention (Custom Trained)',
            'performance_metrics': {
                'map50': 50.56,
                'map50_95': 25.04,
                'training_epochs': 65,
                'dataset_images': 6439
            }
        }
        
    except Exception as e:
        print(f"❌ Error during detection: {e}")
        raise RuntimeError(f"Detection failed: {str(e)}")


def draw_detections(
    image: np.ndarray,
    detections: Dict[str, any],
    class_names: List[str] = CLASS_NAMES,
    draw_confidence: bool = True,
    draw_team_info: bool = True,
    line_thickness: int = 2
) -> np.ndarray:
    """
    Draw detections with Team Autono Minds branding.
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image provided for annotation.")
    
    annotated_image = image.copy()
    
    boxes = detections.get('boxes', [])
    classes = detections.get('classes', [])
    confidences = detections.get('confidences', [])
    
    # Draw detections
    if len(boxes) > 0:
        img_height, img_width = image.shape[:2]
        
        for box, cls_id, conf in zip(boxes, classes, confidences):
            try:
                x1, y1, x2, y2 = map(int, box)
                
                # Ensure coordinates are within bounds
                x1 = max(0, min(x1, img_width - 1))
                y1 = max(0, min(y1, img_height - 1))
                x2 = max(0, min(x2, img_width - 1))
                y2 = max(0, min(y2, img_height - 1))
                
                # Calculate severity
                bbox_area = (x2 - x1) * (y2 - y1)
                severity = classify_severity(cls_id, bbox_area)
                
                # Get color
                color_hex = SEVERITY_COLORS.get(severity, "#3B82F6")
                color_bgr = hex_to_bgr(color_hex)
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color_bgr, line_thickness)
                
                # Draw label
                class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
                label = f"{class_name}: {conf:.2f}" if draw_confidence else class_name
                
                # Label background and text
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                
                (text_width, text_height), _ = cv2.getTextSize(
                    label, font, font_scale, font_thickness
                )
                
                label_y1 = max(y1 - text_height - 10, 0)
                cv2.rectangle(
                    annotated_image,
                    (x1, label_y1),
                    (x1 + text_width + 10, y1),
                    color_bgr,
                    -1
                )
                
                cv2.putText(
                    annotated_image,
                    label,
                    (x1 + 5, y1 - 5),
                    font,
                    font_scale,
                    (255, 255, 255),
                    font_thickness,
                    cv2.LINE_AA
                )
                
            except Exception as e:
                print(f"⚠️ Error drawing detection: {e}")
                continue
    
    # Add Team Autono Minds branding
    if draw_team_info:
        team_info = detections.get('team', 'Autono Minds')
        model_info = detections.get('model_info', 'YOLOv11 + SE Attention')
        _draw_team_badge(annotated_image, team_info, model_info)
    
    return annotated_image


def _draw_team_badge(image: np.ndarray, team: str, model_info: str):
    """Draw Team Autono Minds badge with model info."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # Team name
    team_text = f"🏆 Team {team}"
    model_text = f"🔥 {model_info}"
    
    # Calculate text dimensions
    (team_w, team_h), _ = cv2.getTextSize(team_text, font, font_scale, thickness)
    (model_w, model_h), _ = cv2.getTextSize(model_text, font, 0.5, 1)
    
    # Position in top-right
    img_height, img_width = image.shape[:2]
    x = img_width - max(team_w, model_w) - 15
    y_team = team_h + 15
    y_model = y_team + model_h + 10
    
    # Background for team name
    cv2.rectangle(
        image,
        (x - 5, y_team - team_h - 5),
        (x + team_w + 5, y_team + 5),
        (0, 0, 0),
        -1
    )
    
    # Background for model info
    cv2.rectangle(
        image,
        (x - 5, y_model - model_h - 5),
        (x + model_w + 5, y_model + 5),
        (0, 0, 0),
        -1
    )
    
    # Team name text (gold color)
    cv2.putText(
        image,
        team_text,
        (x, y_team),
        font,
        font_scale,
        (0, 215, 255),  # Gold
        thickness,
        cv2.LINE_AA
    )
    
    # Model info text (green)
    cv2.putText(
        image,
        model_text,
        (x, y_model),
        font,
        0.5,
        (0, 255, 0),  # Green
        1,
        cv2.LINE_AA
    )


# Keep existing utility functions
def classify_severity(class_id: int, bbox_area: float) -> str:
    """
    Classify hazard severity based on class and size.
    Updated for Team Autono Minds' YOLOv11 model classes.
    
    Args:
        class_id: Class ID (0=longitudinal_crack, 1=transverse_crack, 2=alligator_crack, 3=pothole, 4=other_corruption)
        bbox_area: Bounding box area in pixels
        
    Returns:
        Severity level: 'critical', 'high', 'medium', or 'low'
    """
    # Size thresholds (in pixels)
    LARGE_THRESHOLD = 30000
    MEDIUM_THRESHOLD = 15000
    
    # Map class IDs to your trained classes
    if class_id == 0:  # longitudinal_crack
        # Generally less severe, parallel to traffic flow
        if bbox_area > LARGE_THRESHOLD:
            return 'medium'
        else:
            return 'low'
    
    elif class_id == 1:  # transverse_crack (your best performing class - 69.97%)
        # More disruptive, crosses traffic flow
        if bbox_area > LARGE_THRESHOLD:
            return 'high'
        elif bbox_area > MEDIUM_THRESHOLD:
            return 'medium'
        else:
            return 'low'
    
    elif class_id == 2:  # alligator_crack (challenging class - 10.10%)
        # Indicates structural failure, always serious
        if bbox_area > MEDIUM_THRESHOLD:
            return 'critical'
        else:
            return 'high'
    
    elif class_id == 3:  # pothole (62.34% mAP)
        # Direct vehicle damage risk
        if bbox_area > LARGE_THRESHOLD:
            return 'critical'
        elif bbox_area > MEDIUM_THRESHOLD:
            return 'high'
        else:
            return 'medium'
    
    elif class_id == 4:  # other_corruption
        # General road surface issues
        return 'medium'
    
    else:
        # Unknown class ID - default to medium
        print(f"⚠️ Unknown class ID: {class_id}, defaulting to medium severity")
        return 'medium'


def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to BGR tuple."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)


def get_detection_summary(detections: Dict[str, any], class_names: List[str] = CLASS_NAMES) -> Dict[str, any]:
    """Generate detection summary with Team Autono Minds info and severity counts."""
    boxes = detections.get('boxes', [])
    classes = detections.get('classes', [])
    confidences = detections.get('confidences', [])
    
    if len(boxes) == 0:
        return {
            'total_detections': 0,
            'class_counts': {},
            'severity_counts': {'critical': 0, 'high': 0, 'medium': 0, 'low': 0},  # Added this
            'avg_confidence': 0.0,
            'max_confidence': 0.0,
            'min_confidence': 0.0,
            'team': detections.get('team', 'Autono Minds'),
            'model_info': detections.get('model_info', 'YOLOv11 + SE Attention'),
            'performance_metrics': detections.get('performance_metrics', {})
        }
    
    # Count per class
    class_counts = {}
    for cls_id in classes:
        class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Calculate severity distribution - THIS WAS MISSING
    severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
    for box, cls_id in zip(boxes, classes):
        x1, y1, x2, y2 = box
        bbox_area = (x2 - x1) * (y2 - y1)
        severity = classify_severity(cls_id, bbox_area)
        if severity in severity_counts:
            severity_counts[severity] += 1
    
    return {
        'total_detections': len(boxes),
        'class_counts': class_counts,
        'severity_counts': severity_counts,  # Added this field
        'avg_confidence': round(np.mean(confidences), 3) if confidences else 0.0,
        'max_confidence': round(max(confidences), 3) if confidences else 0.0,
        'min_confidence': round(min(confidences), 3) if confidences else 0.0,
        'team': detections.get('team', 'Autono Minds'),
        'model_info': detections.get('model_info', 'YOLOv11 + SE Attention'),
        'performance_metrics': detections.get('performance_metrics', {})
    }


# Export functions
__all__ = [
    'load_model',
    'detect_hazards',
    'draw_detections',
    'classify_severity',
    'hex_to_bgr',
    'get_detection_summary'
]
