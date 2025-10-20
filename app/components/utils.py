"""
Utility module with helper functions for image processing, file handling,
and data formatting.

This module provides common utility functions used across the RoadGuard application.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple, Any
from datetime import datetime, timedelta
from io import BytesIO
import streamlit as st

try:
    from PIL import Image
except ImportError:
    raise ImportError("Pillow package is required. Install with: pip install Pillow")


# ============================================================================
# Image Loading and Conversion
# ============================================================================

def load_image_from_upload(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> Optional[np.ndarray]:
    """
    Load and convert a Streamlit uploaded file to OpenCV format.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Image as numpy array in BGR format (OpenCV), or None if error
        
    Example:
        >>> uploaded = st.file_uploader("Upload image")
        >>> if uploaded:
        >>>     image = load_image_from_upload(uploaded)
    """
    if uploaded_file is None:
        return None
    
    try:
        # Read file bytes
        file_bytes = uploaded_file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(file_bytes, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            print(f"❌ Failed to decode image: {uploaded_file.name}")
            return None
        
        print(f"✅ Loaded image: {uploaded_file.name} - Shape: {image.shape}")
        return image
        
    except Exception as e:
        print(f"❌ Error loading image from upload: {e}")
        st.error(f"Error loading image: {str(e)}")
        return None


def load_image_from_path(image_path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Load an image from a file path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array in BGR format, or None if error
    """
    try:
        image_path = Path(image_path)
        
        if not image_path.exists():
            print(f"❌ Image file not found: {image_path}")
            return None
        
        image = cv2.imread(str(image_path))
        
        if image is None:
            print(f"❌ Failed to load image: {image_path}")
            return None
        
        return image
        
    except Exception as e:
        print(f"❌ Error loading image from path: {e}")
        return None


# ============================================================================
# File Operations
# ============================================================================

def save_uploaded_file(
    uploaded_file: st.runtime.uploaded_file_manager.UploadedFile,
    destination_dir: Union[str, Path],
    prefix: str = ""
) -> Optional[Path]:
    """
    Save a Streamlit uploaded file to disk with a unique filename.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        destination_dir: Directory to save the file
        prefix: Optional prefix for filename
        
    Returns:
        Path to saved file, or None if error
        
    Example:
        >>> uploaded = st.file_uploader("Upload")
        >>> if uploaded:
        >>>     path = save_uploaded_file(uploaded, "uploads/")
    """
    if uploaded_file is None:
        return None
    
    try:
        # Create destination directory if it doesn't exist
        destination_dir = Path(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_name = Path(uploaded_file.name)
        file_extension = original_name.suffix
        
        if prefix:
            filename = f"{prefix}_{timestamp}{file_extension}"
        else:
            filename = f"{timestamp}_{original_name.stem}{file_extension}"
        
        file_path = destination_dir / filename
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        print(f"✅ Saved file: {file_path}")
        return file_path
        
    except Exception as e:
        print(f"❌ Error saving uploaded file: {e}")
        st.error(f"Error saving file: {str(e)}")
        return None


def get_file_size_mb(filepath: Union[str, Path]) -> float:
    """
    Get file size in megabytes.
    
    Args:
        filepath: Path to file
        
    Returns:
        File size in MB, or 0.0 if file not found
    """
    try:
        filepath = Path(filepath)
        
        if not filepath.exists():
            return 0.0
        
        size_bytes = filepath.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        
        return round(size_mb, 2)
        
    except Exception as e:
        print(f"⚠️ Error getting file size: {e}")
        return 0.0


# ============================================================================
# Image Processing
# ============================================================================

def resize_image(
    image: np.ndarray,
    max_width: int = 640,
    max_height: int = 640,
    maintain_aspect: bool = True
) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image as numpy array
        max_width: Maximum width in pixels
        max_height: Maximum height in pixels
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized image as numpy array
        
    Example:
        >>> image = cv2.imread("large_image.jpg")
        >>> resized = resize_image(image, max_width=800, max_height=600)
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image provided for resizing")
    
    try:
        height, width = image.shape[:2]
        
        # If image is already smaller, return as is
        if width <= max_width and height <= max_height:
            return image
        
        if maintain_aspect:
            # Calculate scaling factor
            scale_width = max_width / width
            scale_height = max_height / height
            scale = min(scale_width, scale_height)
            
            # Calculate new dimensions
            new_width = int(width * scale)
            new_height = int(height * scale)
        else:
            new_width = max_width
            new_height = max_height
        
        # Resize using INTER_AREA for downscaling (best quality)
        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )
        
        print(f"✅ Resized image from {width}x{height} to {new_width}x{new_height}")
        return resized
        
    except Exception as e:
        print(f"❌ Error resizing image: {e}")
        return image


def crop_image(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int
) -> Optional[np.ndarray]:
    """
    Crop a region from an image.
    
    Args:
        image: Input image
        x1, y1: Top-left corner coordinates
        x2, y2: Bottom-right corner coordinates
        
    Returns:
        Cropped image or None if invalid
    """
    try:
        height, width = image.shape[:2]
        
        # Validate and clamp coordinates
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        cropped = image[y1:y2, x1:x2]
        return cropped
        
    except Exception as e:
        print(f"❌ Error cropping image: {e}")
        return None


def create_placeholder_image(
    width: int = 640,
    height: int = 480,
    text: str = "No Image",
    bg_color: Tuple[int, int, int] = (128, 128, 128),
    text_color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """
    Create a placeholder image with text.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        text: Text to display
        bg_color: Background color (B, G, R)
        text_color: Text color (B, G, R)
        
    Returns:
        Placeholder image as numpy array
        
    Example:
        >>> placeholder = create_placeholder_image(800, 600, "Coming Soon")
    """
    try:
        # Create blank image
        image = np.full((height, width, 3), bg_color, dtype=np.uint8)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_thickness = 2
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, font_thickness
        )
        
        # Calculate text position (center)
        text_x = (width - text_width) // 2
        text_y = (height + text_height) // 2
        
        # Draw text
        cv2.putText(
            image,
            text,
            (text_x, text_y),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA
        )
        
        return image
        
    except Exception as e:
        print(f"❌ Error creating placeholder image: {e}")
        # Return basic gray image
        return np.full((height, width, 3), bg_color, dtype=np.uint8)


# ============================================================================
# Image Format Conversion
# ============================================================================

def image_to_bytes(
    image: np.ndarray,
    format: str = 'JPEG',
    quality: int = 95
) -> bytes:
    """
    Convert numpy array image to bytes for download or transmission.
    
    Args:
        image: Image as numpy array (BGR or RGB)
        format: Output format ('JPEG', 'PNG', 'BMP')
        quality: JPEG quality (1-100, only for JPEG)
        
    Returns:
        Image as bytes
        
    Example:
        >>> image = cv2.imread("photo.jpg")
        >>> image_bytes = image_to_bytes(image, format='PNG')
        >>> st.download_button("Download", image_bytes, "output.png")
    """
    try:
        # Convert BGR to RGB for PIL
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Save to BytesIO
        buffer = BytesIO()
        
        if format.upper() == 'JPEG':
            pil_image.save(buffer, format='JPEG', quality=quality)
        elif format.upper() == 'PNG':
            pil_image.save(buffer, format='PNG')
        elif format.upper() == 'BMP':
            pil_image.save(buffer, format='BMP')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Get bytes
        buffer.seek(0)
        return buffer.getvalue()
        
    except Exception as e:
        print(f"❌ Error converting image to bytes: {e}")
        raise


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR image (OpenCV) to RGB (Streamlit/PIL).
    
    Args:
        image: Image in BGR format
        
    Returns:
        Image in RGB format
    """
    if image is None or image.size == 0:
        return image
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to BGR (OpenCV format).
    
    Args:
        image: Image in RGB format
        
    Returns:
        Image in BGR format
    """
    if image is None or image.size == 0:
        return image
    
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


# ============================================================================
# Time and Date Formatting
# ============================================================================

def format_timestamp(
    timestamp: Union[str, datetime],
    format_type: str = 'relative'
) -> str:
    """
    Format timestamp to human-readable string.
    
    Args:
        timestamp: Timestamp as string or datetime object
        format_type: 'relative' for "2 hours ago", 'absolute' for "Oct 18, 2025"
        
    Returns:
        Formatted timestamp string
        
    Example:
        >>> format_timestamp("2025-10-18 10:00:00", 'relative')
        "19 minutes ago"
    """
    try:
        # Convert to datetime if string
        if isinstance(timestamp, str):
            # Try common formats
            for fmt in [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d"
            ]:
                try:
                    dt = datetime.strptime(timestamp, fmt)
                    break
                except ValueError:
                    continue
            else:
                return timestamp  # Return as-is if parsing fails
        elif isinstance(timestamp, datetime):
            dt = timestamp
        else:
            return str(timestamp)
        
        if format_type == 'relative':
            # Calculate time difference
            now = datetime.now()
            diff = now - dt
            
            # Format relative time
            if diff.total_seconds() < 60:
                return "just now"
            elif diff.total_seconds() < 3600:
                minutes = int(diff.total_seconds() / 60)
                return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
            elif diff.total_seconds() < 86400:
                hours = int(diff.total_seconds() / 3600)
                return f"{hours} hour{'s' if hours != 1 else ''} ago"
            elif diff.days < 7:
                return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
            elif diff.days < 30:
                weeks = diff.days // 7
                return f"{weeks} week{'s' if weeks != 1 else ''} ago"
            elif diff.days < 365:
                months = diff.days // 30
                return f"{months} month{'s' if months != 1 else ''} ago"
            else:
                years = diff.days // 365
                return f"{years} year{'s' if years != 1 else ''} ago"
        
        else:  # absolute
            return dt.strftime("%b %d, %Y %H:%M")
        
    except Exception as e:
        print(f"⚠️ Error formatting timestamp: {e}")
        return str(timestamp)


def get_current_timestamp(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Get current timestamp as formatted string.
    
    Args:
        format: strftime format string
        
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime(format)


# ============================================================================
# Coordinate Validation
# ============================================================================

def validate_gps_coordinates(lat: float, lon: float) -> bool:
    """
    Validate GPS coordinates are within valid ranges.
    
    Args:
        lat: Latitude (-90 to 90)
        lon: Longitude (-180 to 180)
        
    Returns:
        True if coordinates are valid, False otherwise
        
    Example:
        >>> validate_gps_coordinates(28.6139, 77.2090)
        True
        >>> validate_gps_coordinates(100, 200)
        False
    """
    try:
        # Check latitude range
        if not isinstance(lat, (int, float)) or not (-90 <= lat <= 90):
            return False
        
        # Check longitude range
        if not isinstance(lon, (int, float)) or not (-180 <= lon <= 180):
            return False
        
        return True
        
    except Exception as e:
        print(f"⚠️ Error validating coordinates: {e}")
        return False


def normalize_coordinates(
    lat: float,
    lon: float,
    precision: int = 6
) -> Tuple[float, float]:
    """
    Normalize and round GPS coordinates.
    
    Args:
        lat: Latitude
        lon: Longitude
        precision: Decimal places to round to
        
    Returns:
        Tuple of (normalized_lat, normalized_lon)
    """
    try:
        # Clamp to valid ranges
        lat = max(-90, min(90, lat))
        lon = max(-180, min(180, lon))
        
        # Round to precision
        lat = round(lat, precision)
        lon = round(lon, precision)
        
        return lat, lon
        
    except Exception as e:
        print(f"⚠️ Error normalizing coordinates: {e}")
        return lat, lon


# ============================================================================
# Data Formatting
# ============================================================================

def format_confidence(confidence: float) -> str:
    """
    Format confidence score as percentage.
    
    Args:
        confidence: Confidence value (0.0-1.0)
        
    Returns:
        Formatted percentage string
    """
    return f"{confidence * 100:.1f}%"


def format_distance(distance_meters: float) -> str:
    """
    Format distance with appropriate units.
    
    Args:
        distance_meters: Distance in meters
        
    Returns:
        Formatted distance string
    """
    if distance_meters < 1000:
        return f"{int(distance_meters)}m"
    else:
        return f"{distance_meters / 1000:.1f}km"


def format_speed(speed_kmh: float) -> str:
    """
    Format speed value.
    
    Args:
        speed_kmh: Speed in km/h
        
    Returns:
        Formatted speed string
    """
    return f"{int(speed_kmh)} km/h"


# ============================================================================
# Error Handling
# ============================================================================

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if division by zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Result of division or default value
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except Exception:
        return default


# ============================================================================
# Export Functions
# ============================================================================

__all__ = [
    'load_image_from_upload',
    'load_image_from_path',
    'save_uploaded_file',
    'get_file_size_mb',
    'resize_image',
    'crop_image',
    'create_placeholder_image',
    'image_to_bytes',
    'bgr_to_rgb',
    'rgb_to_bgr',
    'format_timestamp',
    'get_current_timestamp',
    'validate_gps_coordinates',
    'normalize_coordinates',
    'format_confidence',
    'format_distance',
    'format_speed',
    'safe_divide'
]
