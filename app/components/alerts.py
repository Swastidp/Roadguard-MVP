"""
Alerts module for intelligent hazard warning system.

This module provides functions for calculating warning distances, safe speeds,
and generating context-aware alerts based on vehicle speed, hazard severity,
and proximity.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
from enum import Enum

from ..config import (
    ALERT_SETTINGS,
    SEVERITY_COLORS
)

# Extract alert settings
WARNING_DISTANCE_MIN = ALERT_SETTINGS.get('WARNING_DISTANCE_MIN', 50)
REACTION_TIME = ALERT_SETTINGS.get('REACTION_TIME', 1.5)
FRICTION_COEFFICIENT = ALERT_SETTINGS.get('FRICTION_COEFFICIENT', 0.7)


# ============================================================================
# Enums and Data Classes
# ============================================================================

class UrgencyLevel(Enum):
    """Alert urgency levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    INFO = "info"


@dataclass
class Alert:
    """Structured alert information."""
    message: str
    urgency: UrgencyLevel
    recommended_speed: float
    distance_meters: float
    hazard_type: str
    severity: str
    time_to_impact: float  # seconds


# ============================================================================
# Warning Distance Calculations
# ============================================================================

def calculate_warning_distance(
    speed_kmh: float,
    severity: str = 'medium',
    road_condition: str = 'dry'
) -> float:
    """
    Calculate the minimum warning distance based on vehicle speed and hazard severity.
    
    Formula:
    - Total stopping distance = Reaction distance + Braking distance
    - Reaction distance = speed (m/s) × reaction time (s)
    - Braking distance = v² / (2 × g × friction_coefficient)
    
    Args:
        speed_kmh: Current vehicle speed in km/h
        severity: Hazard severity level ('critical', 'high', 'medium', 'low')
        road_condition: Road condition affecting friction ('dry', 'wet', 'icy')
        
    Returns:
        Minimum warning distance in meters
        
    Example:
        >>> calculate_warning_distance(60, 'critical')
        85.5  # meters
    """
    # Validate speed
    if speed_kmh < 0:
        speed_kmh = 0
    
    # Convert speed to m/s
    speed_ms = speed_kmh / 3.6
    
    # Adjust friction coefficient based on road condition
    friction = FRICTION_COEFFICIENT
    if road_condition == 'wet':
        friction *= 0.7  # 30% reduction
    elif road_condition == 'icy':
        friction *= 0.3  # 70% reduction
    
    # Calculate reaction distance
    reaction_distance = speed_ms * REACTION_TIME
    
    # Calculate braking distance using simplified formula
    # Braking distance = v² / (2 × g × μ)
    # Where g = 9.81 m/s² (gravity), μ = friction coefficient
    # Simplified: v² / (250 × μ) for speed in km/h
    braking_distance = (speed_kmh ** 2) / (250 * friction)
    
    # Base warning distance
    base_warning = reaction_distance + braking_distance
    
    # Apply severity multiplier for safety margin
    severity_multipliers = {
        'critical': 1.5,  # 50% more distance for critical hazards
        'high': 1.2,      # 20% more distance
        'medium': 1.0,    # Standard distance
        'low': 0.8        # 20% less distance
    }
    
    multiplier = severity_multipliers.get(severity.lower(), 1.0)
    warning_distance = base_warning * multiplier
    
    # Ensure minimum warning distance
    warning_distance = max(warning_distance, WARNING_DISTANCE_MIN)
    
    return round(warning_distance, 2)


def calculate_safe_speed(
    distance_meters: float,
    severity: str = 'medium',
    road_condition: str = 'dry'
) -> float:
    """
    Calculate the safe speed for a given distance to hazard.
    
    This is the reverse calculation of warning distance, solving for speed
    given a target stopping distance.
    
    Args:
        distance_meters: Distance to hazard in meters
        severity: Hazard severity level
        road_condition: Road condition affecting friction
        
    Returns:
        Recommended safe speed in km/h
        
    Example:
        >>> calculate_safe_speed(50, 'critical')
        42.3  # km/h
    """
    # Adjust for severity (reverse the multiplier)
    severity_multipliers = {
        'critical': 1.5,
        'high': 1.2,
        'medium': 1.0,
        'low': 0.8
    }
    
    multiplier = severity_multipliers.get(severity.lower(), 1.0)
    adjusted_distance = distance_meters / multiplier
    
    # Adjust friction for road conditions
    friction = FRICTION_COEFFICIENT
    if road_condition == 'wet':
        friction *= 0.7
    elif road_condition == 'icy':
        friction *= 0.3
    
    # Solve quadratic equation: d = v*t + v²/(250*μ)
    # Where d = distance, v = speed (km/h), t = reaction time (converted)
    # Rearranged: v²/(250*μ) + v*(t/3.6) - d = 0
    
    a = 1 / (250 * friction)
    b = REACTION_TIME / 3.6
    c = -adjusted_distance
    
    # Quadratic formula: v = (-b + sqrt(b² - 4ac)) / 2a
    discriminant = b**2 - 4*a*c
    
    if discriminant < 0:
        return 0.0  # No real solution, stop completely
    
    safe_speed = (-b + np.sqrt(discriminant)) / (2 * a)
    
    # Ensure non-negative and reasonable speed limits
    safe_speed = max(0, min(safe_speed, 120))  # Cap at 120 km/h
    
    return round(safe_speed, 1)


# ============================================================================
# Alert Generation
# ============================================================================

def generate_alert_message(
    hazard_dict: Dict[str, Any],
    distance_meters: float,
    current_speed_kmh: float,
    road_condition: str = 'dry'
) -> Alert:
    """
    Generate a natural language alert message with recommendations.
    
    Args:
        hazard_dict: Dictionary with hazard information (type, severity, etc.)
        distance_meters: Distance to hazard in meters
        current_speed_kmh: Current vehicle speed in km/h
        road_condition: Current road condition
        
    Returns:
        Alert object with message, urgency, and recommendations
        
    Example:
        >>> hazard = {'class_name': 'pothole', 'severity': 'critical'}
        >>> alert = generate_alert_message(hazard, 45, 60)
        >>> print(alert.message)
        "⚠️ CRITICAL: Pothole ahead in 45m! Reduce speed to 42 km/h immediately!"
    """
    # Extract hazard information
    hazard_type = hazard_dict.get('class_name', 'hazard').replace('_', ' ').title()
    severity = hazard_dict.get('severity', 'medium').lower()
    
    # Calculate safe speed
    safe_speed = calculate_safe_speed(distance_meters, severity, road_condition)
    
    # Calculate warning distance for current speed
    required_distance = calculate_warning_distance(current_speed_kmh, severity, road_condition)
    
    # Calculate time to impact (seconds)
    if current_speed_kmh > 0:
        time_to_impact = (distance_meters / (current_speed_kmh / 3.6))
    else:
        time_to_impact = float('inf')
    
    # Determine urgency level
    urgency = _calculate_urgency(distance_meters, required_distance, time_to_impact)
    
    # Generate message based on urgency
    message = _format_alert_message(
        hazard_type,
        severity,
        distance_meters,
        current_speed_kmh,
        safe_speed,
        urgency,
        time_to_impact
    )
    
    return Alert(
        message=message,
        urgency=urgency,
        recommended_speed=safe_speed,
        distance_meters=distance_meters,
        hazard_type=hazard_type,
        severity=severity,
        time_to_impact=time_to_impact
    )


def _calculate_urgency(
    distance_meters: float,
    required_distance: float,
    time_to_impact: float
) -> UrgencyLevel:
    """
    Calculate alert urgency based on distance and time ratios.
    
    Args:
        distance_meters: Actual distance to hazard
        required_distance: Required stopping distance
        time_to_impact: Time until impact in seconds
        
    Returns:
        UrgencyLevel enum
    """
    # Calculate safety ratio (how much distance we have vs need)
    safety_ratio = distance_meters / required_distance if required_distance > 0 else 1.0
    
    # Determine urgency based on safety ratio and time
    if safety_ratio < 0.5 or time_to_impact < 2:
        return UrgencyLevel.CRITICAL
    elif safety_ratio < 0.8 or time_to_impact < 4:
        return UrgencyLevel.HIGH
    elif safety_ratio < 1.2 or time_to_impact < 6:
        return UrgencyLevel.MODERATE
    elif safety_ratio < 1.5 or time_to_impact < 10:
        return UrgencyLevel.LOW
    else:
        return UrgencyLevel.INFO


def _format_alert_message(
    hazard_type: str,
    severity: str,
    distance: float,
    current_speed: float,
    safe_speed: float,
    urgency: UrgencyLevel,
    time_to_impact: float
) -> str:
    """
    Format alert message based on urgency level.
    
    Args:
        hazard_type: Type of hazard
        severity: Severity level
        distance: Distance to hazard
        current_speed: Current vehicle speed
        safe_speed: Recommended safe speed
        urgency: Urgency level
        time_to_impact: Time to impact
        
    Returns:
        Formatted alert message
    """
    # Format distance
    if distance < 1000:
        distance_str = f"{int(distance)}m"
    else:
        distance_str = f"{distance/1000:.1f}km"
    
    # Format time
    if time_to_impact < 60:
        time_str = f"{int(time_to_impact)}s"
    else:
        time_str = f"{int(time_to_impact/60)}min"
    
    if urgency == UrgencyLevel.CRITICAL:
        message = (
            f"⚠️ **CRITICAL**: {hazard_type} ahead in {distance_str}! "
            f"**SLOW DOWN IMMEDIATELY** to {int(safe_speed)} km/h!"
        )
    elif urgency == UrgencyLevel.HIGH:
        message = (
            f"🚨 **WARNING**: {severity.capitalize()} {hazard_type} in {distance_str}. "
            f"Reduce speed to {int(safe_speed)} km/h. Time: {time_str}"
        )
    elif urgency == UrgencyLevel.MODERATE:
        message = (
            f"⚠️ **CAUTION**: {hazard_type} ahead in {distance_str}. "
            f"Recommended speed: {int(safe_speed)} km/h"
        )
    elif urgency == UrgencyLevel.LOW:
        message = (
            f"ℹ️ **NOTICE**: {hazard_type} in {distance_str}. "
            f"Maintain caution, suggested speed: {int(safe_speed)} km/h"
        )
    else:  # INFO
        message = (
            f"📍 {hazard_type} detected ahead in {distance_str}. "
            f"Current speed acceptable ({int(current_speed)} km/h)"
        )
    
    return message


# ============================================================================
# Alert Triggering Logic
# ============================================================================

def should_trigger_alert(
    hazard: Dict[str, Any],
    vehicle_position: Tuple[float, float],
    vehicle_speed: float,
    vehicle_heading: Optional[float] = None,
    heading_tolerance: float = 30.0
) -> bool:
    """
    Determine if an alert should be triggered for a given hazard.
    
    Args:
        hazard: Hazard dictionary with lat, lon, severity
        vehicle_position: (latitude, longitude) of vehicle
        vehicle_speed: Current speed in km/h
        vehicle_heading: Vehicle heading in degrees (0-360, None = ignore)
        heading_tolerance: Acceptable deviation from heading (degrees)
        
    Returns:
        True if alert should be triggered, False otherwise
    """
    # Extract positions
    hazard_lat = hazard.get('latitude')
    hazard_lon = hazard.get('longitude')
    vehicle_lat, vehicle_lon = vehicle_position
    
    # Validate coordinates
    if any(coord is None for coord in [hazard_lat, hazard_lon, vehicle_lat, vehicle_lon]):
        return False
    
    # Calculate distance to hazard
    distance = calculate_distance_haversine(
        vehicle_lat, vehicle_lon,
        hazard_lat, hazard_lon
    )
    
    # Get hazard severity
    severity = hazard.get('severity', 'medium')
    
    # Calculate warning distance threshold
    warning_threshold = calculate_warning_distance(vehicle_speed, severity)
    
    # Check if within warning distance
    if distance > warning_threshold:
        return False
    
    # If heading is provided, check if hazard is in direction of travel
    if vehicle_heading is not None:
        hazard_bearing = calculate_bearing(
            vehicle_lat, vehicle_lon,
            hazard_lat, hazard_lon
        )
        
        # Calculate angular difference
        angle_diff = abs(hazard_bearing - vehicle_heading)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        # Check if hazard is in direction of travel
        if angle_diff > heading_tolerance:
            return False
    
    return True


# ============================================================================
# Geospatial Calculations
# ============================================================================

def calculate_distance_haversine(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> float:
    """
    Calculate distance between two GPS coordinates using Haversine formula.
    
    The Haversine formula determines the great-circle distance between two points
    on a sphere given their longitudes and latitudes.
    
    Args:
        lat1: Latitude of first point (degrees)
        lon1: Longitude of first point (degrees)
        lat2: Latitude of second point (degrees)
        lon2: Longitude of second point (degrees)
        
    Returns:
        Distance in meters
        
    Example:
        >>> calculate_distance_haversine(28.6139, 77.2090, 28.6149, 77.2100)
        123.45  # meters
    """
    # Earth's radius in meters
    R = 6371000
    
    # Convert degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    distance = R * c
    
    return round(distance, 2)


def calculate_bearing(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> float:
    """
    Calculate bearing (direction) from point 1 to point 2.
    
    Args:
        lat1: Latitude of starting point (degrees)
        lon1: Longitude of starting point (degrees)
        lat2: Latitude of destination point (degrees)
        lon2: Longitude of destination point (degrees)
        
    Returns:
        Bearing in degrees (0-360, where 0 = North, 90 = East)
    """
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlon_rad = np.radians(lon2 - lon1)
    
    # Calculate bearing
    x = np.sin(dlon_rad) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon_rad)
    
    bearing_rad = np.arctan2(x, y)
    bearing_deg = np.degrees(bearing_rad)
    
    # Normalize to 0-360
    bearing_deg = (bearing_deg + 360) % 360
    
    return round(bearing_deg, 2)


# ============================================================================
# Display Formatting
# ============================================================================

def format_alert_for_display(alert: Alert) -> str:
    """
    Format an alert for Streamlit display with color coding.
    
    Args:
        alert: Alert object to format
        
    Returns:
        Markdown-formatted string with color coding
    """
    # Get color based on urgency
    urgency_colors = {
        UrgencyLevel.CRITICAL: "#DC2626",
        UrgencyLevel.HIGH: "#EA580C",
        UrgencyLevel.MODERATE: "#F59E0B",
        UrgencyLevel.LOW: "#3B82F6",
        UrgencyLevel.INFO: "#10B981"
    }
    
    color = urgency_colors.get(alert.urgency, "#3B82F6")
    
    # Format alert box
    formatted = f"""
    <div style="
        background-color: {color}15;
        border-left: 5px solid {color};
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    ">
        <div style="color: {color}; font-weight: bold; font-size: 18px; margin-bottom: 10px;">
            {alert.message}
        </div>
        <div style="color: #374151; font-size: 14px;">
            <strong>Distance:</strong> {alert.distance_meters:.0f}m | 
            <strong>Time to impact:</strong> {alert.time_to_impact:.1f}s | 
            <strong>Recommended speed:</strong> {alert.recommended_speed:.0f} km/h
        </div>
    </div>
    """
    
    return formatted


def format_alert_for_audio(alert: Alert) -> str:
    """
    Format alert message for text-to-speech output.
    
    Args:
        alert: Alert object to format
        
    Returns:
        Clean text suitable for audio output
    """
    # Remove markdown and emojis
    message = alert.message
    message = message.replace('**', '')
    message = message.replace('⚠️', 'Warning:')
    message = message.replace('🚨', 'Alert:')
    message = message.replace('ℹ️', 'Information:')
    message = message.replace('📍', '')
    
    return message


# ============================================================================
# Utility Functions
# ============================================================================

def kmh_to_ms(speed_kmh: float) -> float:
    """Convert speed from km/h to m/s."""
    return speed_kmh / 3.6


def ms_to_kmh(speed_ms: float) -> float:
    """Convert speed from m/s to km/h."""
    return speed_ms * 3.6


def get_speed_category(speed_kmh: float) -> str:
    """
    Categorize speed into ranges.
    
    Args:
        speed_kmh: Speed in km/h
        
    Returns:
        Speed category string
    """
    if speed_kmh < 20:
        return "very_slow"
    elif speed_kmh < 40:
        return "slow"
    elif speed_kmh < 60:
        return "moderate"
    elif speed_kmh < 80:
        return "fast"
    else:
        return "very_fast"


# ============================================================================
# Export Functions
# ============================================================================

__all__ = [
    'Alert',
    'UrgencyLevel',
    'calculate_warning_distance',
    'calculate_safe_speed',
    'generate_alert_message',
    'should_trigger_alert',
    'calculate_distance_haversine',
    'calculate_bearing',
    'format_alert_for_display',
    'format_alert_for_audio',
    'kmh_to_ms',
    'ms_to_kmh',
    'get_speed_category'
]
