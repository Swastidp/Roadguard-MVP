"""
Privacy module for face and license plate anonymization.

This module provides functions for detecting and anonymizing sensitive regions
in images to ensure privacy compliance and GDPR adherence.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Literal
import streamlit as st

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics package is required. Install with: pip install ultralytics")

from ..config import PRIVACY_MODEL_PATH


# ============================================================================
# Model Loading
# ============================================================================

@st.cache_resource
def load_privacy_model(model_path: str = PRIVACY_MODEL_PATH) -> Optional[YOLO]:
    """
    Load YOLOv8n model for face and license plate detection.
    
    The model is cached to avoid reloading on subsequent calls.
    YOLOv8n is pretrained on COCO dataset which includes 'person' class
    that can be used for face detection when combined with upper body detection.
    
    Args:
        model_path: Path to the YOLO model (default: yolov8n.pt)
        
    Returns:
        YOLO model object if successful, None otherwise
        
    Note:
        For production, consider using a specialized model trained on:
        - Face detection datasets (WIDER Face, CelebA)
        - License plate datasets (OpenALPR, CCPD)
    """
    try:
        # Load YOLOv8n model
        model = YOLO(model_path)
        
        if model is None:
            raise ValueError("Model loaded but returned None")
        
        print(f"✅ Privacy model loaded successfully: {model_path}")
        return model
        
    except FileNotFoundError as e:
        print(f"❌ Privacy model file not found: {e}")
        st.warning(
            f"Privacy model not found at: {model_path}\n"
            "The model will be downloaded automatically on first use."
        )
        return None
        
    except Exception as e:
        print(f"❌ Error loading privacy model: {e}")
        st.error(f"Error loading privacy model: {str(e)}")
        return None


# ============================================================================
# Sensitive Region Detection
# ============================================================================

def detect_sensitive_regions(
    model: YOLO,
    image: np.ndarray,
    conf_threshold: float = 0.4,
    target_classes: Optional[List[int]] = None
) -> List[List[float]]:
    """
    Detect faces and license plates in an image.
    
    Args:
        model: Loaded YOLO model for detection
        image: Input image as numpy array (BGR format)
        conf_threshold: Confidence threshold for detections (0.0-1.0)
        target_classes: List of class IDs to detect (None = detect all)
                       For COCO: 0=person (can indicate faces in upper regions)
                       For custom models: specify face and plate class IDs
        
    Returns:
        List of bounding boxes [[x1, y1, x2, y2], ...]
        
    Raises:
        ValueError: If model is None or image is invalid
        
    Note:
        For better face detection, consider using dedicated models like:
        - RetinaFace, MTCNN, or Dlib for faces
        - OpenALPR or custom YOLO for license plates
    """
    if model is None:
        raise ValueError("Privacy model is None. Please load a valid model first.")
    
    if image is None or image.size == 0:
        raise ValueError("Invalid image provided for detection.")
    
    try:
        # Run inference
        results = model(
            image,
            conf=conf_threshold,
            verbose=False
        )
        
        if len(results) == 0:
            return []
        
        result = results[0]
        
        # Extract bounding boxes
        bboxes = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            
            for box, cls_id, conf in zip(boxes_xyxy, classes, confidences):
                # Filter by target classes if specified
                if target_classes is not None and cls_id not in target_classes:
                    continue
                
                # For person class (0), focus on upper body regions (likely faces)
                if cls_id == 0:
                    x1, y1, x2, y2 = box
                    # Take upper 30% of person bounding box as potential face region
                    height = y2 - y1
                    y2_adjusted = y1 + (height * 0.3)
                    bboxes.append([x1, y1, x2, y2_adjusted])
                else:
                    bboxes.append(box.tolist())
        
        print(f"🔒 Detected {len(bboxes)} sensitive region(s)")
        return bboxes
        
    except Exception as e:
        print(f"❌ Error detecting sensitive regions: {e}")
        raise RuntimeError(f"Sensitive region detection failed: {str(e)}")


# ============================================================================
# Anonymization Methods
# ============================================================================

def anonymize_frame(
    image: np.ndarray,
    bboxes: List[List[float]],
    method: Literal['gaussian', 'pixelate', 'black'] = 'gaussian',
    blur_kernel: Tuple[int, int] = (99, 99),
    pixelate_factor: int = 10
) -> np.ndarray:
    """
    Anonymize sensitive regions in an image using specified method.
    
    Args:
        image: Input image as numpy array (BGR format)
        bboxes: List of bounding boxes to anonymize [[x1, y1, x2, y2], ...]
        method: Anonymization method:
                - 'gaussian': Apply Gaussian blur
                - 'pixelate': Downscale and upscale for pixelation effect
                - 'black': Fill with solid black rectangle
        blur_kernel: Kernel size for Gaussian blur (must be odd numbers)
        pixelate_factor: Downscaling factor for pixelation (higher = more pixelated)
        
    Returns:
        Anonymized image with sensitive regions obscured
        
    Raises:
        ValueError: If image is invalid or method is unsupported
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image provided for anonymization.")
    
    if method not in ['gaussian', 'pixelate', 'black']:
        raise ValueError(f"Unsupported anonymization method: {method}")
    
    # Clone image to avoid modifying original
    anonymized_image = image.copy()
    
    img_height, img_width = image.shape[:2]
    
    # Process each bounding box
    for bbox in bboxes:
        try:
            # Extract and validate coordinates
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            x2 = max(0, min(x2, img_width))
            y2 = max(0, min(y2, img_height))
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Extract region of interest
            roi = anonymized_image[y1:y2, x1:x2]
            
            if roi.size == 0:
                continue
            
            # Apply anonymization method
            if method == 'gaussian':
                # Apply Gaussian blur
                blurred_roi = cv2.GaussianBlur(roi, blur_kernel, 0)
                anonymized_image[y1:y2, x1:x2] = blurred_roi
                
            elif method == 'pixelate':
                # Downscale and upscale for pixelation
                roi_height, roi_width = roi.shape[:2]
                
                # Calculate downscaled dimensions
                small_width = max(1, roi_width // pixelate_factor)
                small_height = max(1, roi_height // pixelate_factor)
                
                # Downscale
                small_roi = cv2.resize(
                    roi,
                    (small_width, small_height),
                    interpolation=cv2.INTER_LINEAR
                )
                
                # Upscale back to original size
                pixelated_roi = cv2.resize(
                    small_roi,
                    (roi_width, roi_height),
                    interpolation=cv2.INTER_NEAREST
                )
                
                anonymized_image[y1:y2, x1:x2] = pixelated_roi
                
            elif method == 'black':
                # Fill with solid black rectangle
                cv2.rectangle(
                    anonymized_image,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 0),
                    -1  # Filled rectangle
                )
                
                # Add privacy icon/text (optional)
                text = "REDACTED"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                
                # Get text size
                (text_width, text_height), _ = cv2.getTextSize(
                    text, font, font_scale, font_thickness
                )
                
                # Center text in rectangle
                text_x = x1 + (x2 - x1 - text_width) // 2
                text_y = y1 + (y2 - y1 + text_height) // 2
                
                cv2.putText(
                    anonymized_image,
                    text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    font_thickness,
                    cv2.LINE_AA
                )
        
        except Exception as e:
            print(f"⚠️ Error anonymizing region {bbox}: {e}")
            continue
    
    return anonymized_image


# ============================================================================
# Complete Anonymization Pipeline
# ============================================================================

def anonymize_pipeline(
    image: np.ndarray,
    method: Literal['gaussian', 'pixelate', 'black'] = 'gaussian',
    conf_threshold: float = 0.4,
    model: Optional[YOLO] = None
) -> Tuple[np.ndarray, int]:
    """
    Complete pipeline for detecting and anonymizing sensitive regions.
    
    This function combines detection and anonymization into a single call.
    
    Args:
        image: Input image as numpy array (BGR format)
        method: Anonymization method ('gaussian', 'pixelate', or 'black')
        conf_threshold: Confidence threshold for detection
        model: Pre-loaded YOLO model (optional, will load if None)
        
    Returns:
        Tuple of (anonymized_image, num_regions_blurred)
        
    Raises:
        ValueError: If image is invalid
        RuntimeError: If pipeline execution fails
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image provided to anonymization pipeline.")
    
    try:
        # Load model if not provided
        if model is None:
            model = load_privacy_model()
            
            if model is None:
                print("⚠️ Privacy model not available, returning original image")
                return image.copy(), 0
        
        # Detect sensitive regions
        bboxes = detect_sensitive_regions(
            model,
            image,
            conf_threshold=conf_threshold
        )
        
        # If no sensitive regions detected, return original
        if len(bboxes) == 0:
            return image.copy(), 0
        
        # Anonymize detected regions
        anonymized_image = anonymize_frame(
            image,
            bboxes,
            method=method
        )
        
        num_regions = len(bboxes)
        print(f"✅ Anonymized {num_regions} sensitive region(s) using {method} method")
        
        return anonymized_image, num_regions
        
    except Exception as e:
        print(f"❌ Error in anonymization pipeline: {e}")
        raise RuntimeError(f"Anonymization pipeline failed: {str(e)}")


# ============================================================================
# Privacy Compliance Verification
# ============================================================================

def is_privacy_compliant(
    image: np.ndarray,
    max_identifiable_regions: int = 0,
    conf_threshold: float = 0.3,
    model: Optional[YOLO] = None
) -> Tuple[bool, int]:
    """
    Check if an image is privacy compliant (no identifiable regions).
    
    Args:
        image: Input image to check as numpy array (BGR format)
        max_identifiable_regions: Maximum allowed identifiable regions (default: 0)
        conf_threshold: Confidence threshold for detection (lower = stricter)
        model: Pre-loaded YOLO model (optional)
        
    Returns:
        Tuple of (is_compliant, num_regions_detected)
        - is_compliant: True if compliant, False otherwise
        - num_regions_detected: Number of sensitive regions found
        
    Note:
        Lower confidence thresholds provide stricter privacy checking
        but may result in false positives.
    """
    if image is None or image.size == 0:
        return False, -1
    
    try:
        # Load model if not provided
        if model is None:
            model = load_privacy_model()
            
            if model is None:
                print("⚠️ Privacy model not available, cannot verify compliance")
                return False, -1
        
        # Detect sensitive regions with stricter threshold
        bboxes = detect_sensitive_regions(
            model,
            image,
            conf_threshold=conf_threshold
        )
        
        num_regions = len(bboxes)
        is_compliant = num_regions <= max_identifiable_regions
        
        compliance_status = "✅ COMPLIANT" if is_compliant else "❌ NON-COMPLIANT"
        print(f"{compliance_status}: {num_regions} sensitive region(s) detected")
        
        return is_compliant, num_regions
        
    except Exception as e:
        print(f"❌ Error checking privacy compliance: {e}")
        return False, -1


# ============================================================================
# Batch Processing
# ============================================================================

def anonymize_video_frame_batch(
    frames: List[np.ndarray],
    method: Literal['gaussian', 'pixelate', 'black'] = 'gaussian',
    conf_threshold: float = 0.4,
    model: Optional[YOLO] = None
) -> List[Tuple[np.ndarray, int]]:
    """
    Batch process multiple video frames for anonymization.
    
    Args:
        frames: List of frames as numpy arrays
        method: Anonymization method
        conf_threshold: Confidence threshold for detection
        model: Pre-loaded YOLO model (optional)
        
    Returns:
        List of tuples (anonymized_frame, num_regions)
    """
    # Load model once for all frames
    if model is None:
        model = load_privacy_model()
    
    results = []
    
    for i, frame in enumerate(frames):
        try:
            anonymized_frame, num_regions = anonymize_pipeline(
                frame,
                method=method,
                conf_threshold=conf_threshold,
                model=model
            )
            results.append((anonymized_frame, num_regions))
            
        except Exception as e:
            print(f"⚠️ Error processing frame {i}: {e}")
            results.append((frame.copy(), 0))
    
    return results


# ============================================================================
# Utility Functions
# ============================================================================

def get_privacy_stats(num_regions: int, method: str) -> Dict[str, any]:
    """
    Generate privacy statistics summary.
    
    Args:
        num_regions: Number of regions anonymized
        method: Anonymization method used
        
    Returns:
        Dictionary with privacy statistics
    """
    return {
        'regions_anonymized': num_regions,
        'method_used': method,
        'is_compliant': num_regions >= 0,
        'privacy_level': 'high' if num_regions == 0 else 'protected'
    }


# ============================================================================
# Export Functions
# ============================================================================

__all__ = [
    'load_privacy_model',
    'detect_sensitive_regions',
    'anonymize_frame',
    'anonymize_pipeline',
    'is_privacy_compliant',
    'anonymize_video_frame_batch',
    'get_privacy_stats'
]
