"""
Deduplication and spatial clustering module for hazard detections.

This module provides functions for:
- Spatial clustering of nearby detections
- Deduplication of similar hazards
- Confidence decay over time
- Merging duplicate reports
- Identifying potentially repaired hazards
"""

import numpy as np
from sklearn.cluster import DBSCAN
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict


# ============================================================================
# Distance Calculations
# ============================================================================

def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> float:
    """
    Calculate the great circle distance between two points on Earth.
    
    Uses the Haversine formula to compute the shortest distance over the
    earth's surface, giving an 'as-the-crow-flies' distance between points.
    
    Args:
        lat1: Latitude of first point (degrees)
        lon1: Longitude of first point (degrees)
        lat2: Latitude of second point (degrees)
        lon2: Longitude of second point (degrees)
        
    Returns:
        Distance in meters
        
    Example:
        >>> haversine_distance(28.6139, 77.2090, 28.6149, 77.2100)
        123.45
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
    
    return float(distance)


def haversine_vectorized(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray
) -> np.ndarray:
    """
    Vectorized version of Haversine distance calculation.
    
    Args:
        lat1, lon1: Arrays of first points' coordinates
        lat2, lon2: Arrays of second points' coordinates
        
    Returns:
        Array of distances in meters
    """
    R = 6371000
    
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c


# ============================================================================
# Deduplication
# ============================================================================

def deduplicate_detections(
    detections: List[Dict[str, Any]],
    eps_meters: float = 10.0,
    min_samples: int = 2,
    selection_strategy: str = 'highest_confidence'
) -> List[Dict[str, Any]]:
    """
    Deduplicate nearby hazard detections using spatial clustering.
    
    Groups detections within eps_meters radius and selects a representative
    detection from each cluster.
    
    Args:
        detections: List of detection dictionaries with keys:
                   - latitude, longitude (required)
                   - confidence, timestamp, severity (optional)
        eps_meters: Maximum distance (meters) between detections in same cluster
        min_samples: Minimum detections to form a cluster (1 = all become clusters)
        selection_strategy: How to select representative detection:
                          - 'highest_confidence': Use detection with max confidence
                          - 'most_recent': Use most recent detection
                          - 'centroid': Use detection closest to cluster center
        
    Returns:
        List of unique hazard dictionaries with aggregated information
        
    Raises:
        ValueError: If detections list is empty or missing required fields
        
    Example:
        >>> detections = [
        ...     {'latitude': 28.6139, 'longitude': 77.2090, 'confidence': 0.85},
        ...     {'latitude': 28.6140, 'longitude': 77.2091, 'confidence': 0.90}
        ... ]
        >>> unique = deduplicate_detections(detections, eps_meters=15)
        >>> len(unique)
        1
    """
    if not detections:
        return []
    
    # Validate required fields
    required_fields = {'latitude', 'longitude'}
    if not all(required_fields.issubset(d.keys()) for d in detections):
        raise ValueError("All detections must have 'latitude' and 'longitude' fields")
    
    # Extract coordinates
    coords = np.array([[d['latitude'], d['longitude']] for d in detections])
    
    # Convert eps from meters to radians
    # eps in radians = eps in meters / Earth radius in meters
    eps_radians = eps_meters / 6371000
    
    # Perform DBSCAN clustering with haversine metric
    # Note: sklearn expects coordinates in radians for haversine
    coords_radians = np.radians(coords)
    
    clustering = DBSCAN(
        eps=eps_radians,
        min_samples=min_samples,
        metric='haversine',
        algorithm='ball_tree'
    ).fit(coords_radians)
    
    labels = clustering.labels_
    
    # Group detections by cluster
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(idx)
    
    # Select representative detection from each cluster
    unique_detections = []
    
    for cluster_id, indices in clusters.items():
        cluster_detections = [detections[i] for i in indices]
        
        # Select representative based on strategy
        if selection_strategy == 'highest_confidence':
            representative = select_by_confidence(cluster_detections)
        elif selection_strategy == 'most_recent':
            representative = select_by_timestamp(cluster_detections)
        elif selection_strategy == 'centroid':
            cluster_coords = coords[indices]
            representative = select_by_centroid(cluster_detections, cluster_coords)
        else:
            raise ValueError(f"Unknown selection strategy: {selection_strategy}")
        
        # Add cluster information
        representative['cluster_id'] = int(cluster_id)
        representative['cluster_size'] = len(indices)
        representative['report_count'] = len(indices)
        
        # Calculate cluster statistics
        if len(indices) > 1:
            confidences = [d.get('confidence', 0) for d in cluster_detections]
            representative['avg_confidence'] = float(np.mean(confidences))
            representative['max_confidence'] = float(np.max(confidences))
            representative['min_confidence'] = float(np.min(confidences))
        
        unique_detections.append(representative)
    
    return unique_detections


def select_by_confidence(detections: List[Dict]) -> Dict:
    """Select detection with highest confidence score."""
    return max(detections, key=lambda d: d.get('confidence', 0))


def select_by_timestamp(detections: List[Dict]) -> Dict:
    """Select most recent detection."""
    def parse_timestamp(d):
        ts = d.get('timestamp')
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts.replace('Z', '+00:00'))
            except:
                return datetime.min
        elif isinstance(ts, datetime):
            return ts
        else:
            return datetime.min
    
    return max(detections, key=parse_timestamp)


def select_by_centroid(detections: List[Dict], coords: np.ndarray) -> Dict:
    """Select detection closest to cluster centroid."""
    centroid = coords.mean(axis=0)
    
    distances = [
        haversine_distance(centroid[0], centroid[1], d['latitude'], d['longitude'])
        for d in detections
    ]
    
    min_idx = np.argmin(distances)
    return detections[min_idx]


# ============================================================================
# Confidence Updates
# ============================================================================

def update_hazard_confidence(
    hazard_dict: Dict[str, Any],
    time_decay_days: float = 30.0,
    current_time: Optional[datetime] = None
) -> float:
    """
    Calculate time-decayed confidence score for a hazard.
    
    Confidence decreases exponentially over time to reflect uncertainty
    about whether the hazard still exists.
    
    Args:
        hazard_dict: Hazard dictionary with 'confidence' and 'last_seen'
        time_decay_days: Half-life for confidence decay (days)
        current_time: Reference time (default: now)
        
    Returns:
        Updated confidence score (0.0 to 1.0)
        
    Formula:
        new_confidence = base_confidence * exp(-days_elapsed / time_decay_days)
        
    Example:
        >>> hazard = {
        ...     'confidence': 0.9,
        ...     'last_seen': datetime(2025, 9, 18)
        ... }
        >>> # After 30 days, confidence drops to ~0.9 * 0.37 = 0.33
        >>> update_hazard_confidence(hazard, current_time=datetime(2025, 10, 18))
        0.331
    """
    if current_time is None:
        current_time = datetime.now()
    
    # Get base confidence
    base_confidence = hazard_dict.get('confidence', 0.5)
    
    # Parse last_seen timestamp
    last_seen = hazard_dict.get('last_seen')
    if isinstance(last_seen, str):
        try:
            last_seen = datetime.fromisoformat(last_seen.replace('Z', '+00:00'))
        except:
            return base_confidence  # Can't parse, return base confidence
    elif not isinstance(last_seen, datetime):
        return base_confidence
    
    # Calculate days elapsed
    days_elapsed = (current_time - last_seen).total_seconds() / 86400
    
    if days_elapsed < 0:
        days_elapsed = 0  # Future timestamp, no decay
    
    # Apply exponential decay
    decay_factor = np.exp(-days_elapsed / time_decay_days)
    new_confidence = base_confidence * decay_factor
    
    # Clamp to valid range
    new_confidence = max(0.0, min(1.0, new_confidence))
    
    return round(float(new_confidence), 4)


# ============================================================================
# Merging Duplicate Reports
# ============================================================================

def merge_duplicate_reports(
    existing_hazard: Dict[str, Any],
    new_detection: Dict[str, Any],
    confidence_boost: float = 0.05
) -> Dict[str, Any]:
    """
    Merge a new detection report into an existing hazard record.
    
    Updates the existing hazard with information from the new detection,
    including report count, confidence, severity, and timestamps.
    
    Args:
        existing_hazard: Current hazard record
        new_detection: New detection to merge
        confidence_boost: Confidence increase per additional report (max 0.95)
        
    Returns:
        Updated hazard dictionary
        
    Rules:
        - Increment report_count
        - Update last_seen to most recent
        - Boost confidence (capped at 0.95)
        - Upgrade severity if new detection is more severe
        - Keep position of original detection
    """
    merged = existing_hazard.copy()
    
    # Increment report count
    merged['report_count'] = existing_hazard.get('report_count', 1) + 1
    
    # Update last_seen to most recent timestamp
    existing_time = existing_hazard.get('last_seen', datetime.min)
    new_time = new_detection.get('timestamp', datetime.now())
    
    if isinstance(existing_time, str):
        existing_time = datetime.fromisoformat(existing_time.replace('Z', '+00:00'))
    if isinstance(new_time, str):
        new_time = datetime.fromisoformat(new_time.replace('Z', '+00:00'))
    
    merged['last_seen'] = max(existing_time, new_time)
    
    # Boost confidence with diminishing returns
    base_confidence = existing_hazard.get('confidence', 0.5)
    new_confidence = min(0.95, base_confidence + confidence_boost)
    merged['confidence'] = round(new_confidence, 4)
    
    # Update severity if new detection is more severe
    severity_order = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
    existing_severity = existing_hazard.get('severity', 'low')
    new_severity = new_detection.get('severity', 'low')
    
    if severity_order.get(new_severity, 0) > severity_order.get(existing_severity, 0):
        merged['severity'] = new_severity
        merged['severity_upgraded'] = True
    
    # Update status to active if it was marked as resolved
    if existing_hazard.get('status') in ['resolved', 'possibly_repaired']:
        merged['status'] = 'active'
        merged['reactivated'] = True
    
    return merged


# ============================================================================
# Hazard Status Management
# ============================================================================

def mark_repaired_hazards(
    hazards_list: List[Dict[str, Any]],
    threshold_days: int = 60,
    current_time: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """
    Identify and mark hazards that may have been repaired.
    
    Hazards that haven't been reported for a long time are likely fixed
    and should be marked as 'possibly_repaired' for verification.
    
    Args:
        hazards_list: List of hazard dictionaries
        threshold_days: Days without reports before marking as possibly repaired
        current_time: Reference time (default: now)
        
    Returns:
        Updated hazards list with status changes
        
    Status transitions:
        - active → possibly_repaired (if threshold exceeded)
        - possibly_repaired → unchanged
        - resolved → unchanged
    """
    if current_time is None:
        current_time = datetime.now()
    
    updated_hazards = []
    
    for hazard in hazards_list:
        updated_hazard = hazard.copy()
        
        # Skip if already resolved
        if hazard.get('status') in ['resolved', 'verified']:
            updated_hazards.append(updated_hazard)
            continue
        
        # Parse last_seen timestamp
        last_seen = hazard.get('last_seen')
        if isinstance(last_seen, str):
            try:
                last_seen = datetime.fromisoformat(last_seen.replace('Z', '+00:00'))
            except:
                updated_hazards.append(updated_hazard)
                continue
        elif not isinstance(last_seen, datetime):
            updated_hazards.append(updated_hazard)
            continue
        
        # Calculate days since last seen
        days_elapsed = (current_time - last_seen).days
        
        # Mark as possibly repaired if threshold exceeded
        if days_elapsed >= threshold_days:
            updated_hazard['status'] = 'possibly_repaired'
            updated_hazard['days_since_seen'] = days_elapsed
            updated_hazard['marked_at'] = current_time.isoformat()
            
            # Reduce confidence
            updated_hazard['confidence'] = update_hazard_confidence(
                hazard,
                time_decay_days=threshold_days / 2,
                current_time=current_time
            )
        
        updated_hazards.append(updated_hazard)
    
    return updated_hazards


# ============================================================================
# Batch Processing
# ============================================================================

def process_detection_batch(
    new_detections: List[Dict[str, Any]],
    existing_hazards: List[Dict[str, Any]],
    eps_meters: float = 10.0,
    merge_threshold: float = 15.0
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Process a batch of new detections against existing hazards.
    
    This function:
    1. Deduplicates new detections
    2. Matches them with existing hazards
    3. Merges duplicates or creates new hazards
    
    Args:
        new_detections: List of newly detected hazards
        existing_hazards: List of existing hazard records
        eps_meters: Clustering radius for deduplication
        merge_threshold: Distance threshold (meters) for matching with existing hazards
        
    Returns:
        Tuple of (updated_existing_hazards, new_unique_hazards)
    """
    # Deduplicate new detections
    unique_detections = deduplicate_detections(new_detections, eps_meters=eps_meters)
    
    updated_existing = []
    truly_new = []
    
    # Try to match each detection with existing hazards
    for detection in unique_detections:
        det_lat = detection['latitude']
        det_lon = detection['longitude']
        
        # Find closest existing hazard
        matched = False
        min_distance = float('inf')
        closest_hazard = None
        
        for hazard in existing_hazards:
            distance = haversine_distance(
                det_lat, det_lon,
                hazard['latitude'], hazard['longitude']
            )
            
            if distance < min_distance:
                min_distance = distance
                closest_hazard = hazard
        
        # Merge if within threshold
        if closest_hazard and min_distance <= merge_threshold:
            merged = merge_duplicate_reports(closest_hazard, detection)
            updated_existing.append(merged)
            matched = True
        
        # Add as new hazard if no match
        if not matched:
            truly_new.append(detection)
    
    # Add unmatched existing hazards
    matched_ids = {h.get('id') for h in updated_existing}
    for hazard in existing_hazards:
        if hazard.get('id') not in matched_ids:
            updated_existing.append(hazard)
    
    return updated_existing, truly_new


# ============================================================================
# Utility Functions
# ============================================================================

def calculate_cluster_statistics(detections: List[Dict]) -> Dict[str, Any]:
    """
    Calculate statistics for a cluster of detections.
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        Dictionary with cluster statistics
    """
    if not detections:
        return {}
    
    confidences = [d.get('confidence', 0) for d in detections]
    
    stats = {
        'count': len(detections),
        'avg_confidence': float(np.mean(confidences)),
        'std_confidence': float(np.std(confidences)),
        'min_confidence': float(np.min(confidences)),
        'max_confidence': float(np.max(confidences))
    }
    
    # Timestamp statistics if available
    timestamps = []
    for d in detections:
        ts = d.get('timestamp')
        if isinstance(ts, str):
            try:
                timestamps.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
            except:
                pass
        elif isinstance(ts, datetime):
            timestamps.append(ts)
    
    if timestamps:
        stats['first_seen'] = min(timestamps).isoformat()
        stats['last_seen'] = max(timestamps).isoformat()
        stats['time_span_hours'] = (max(timestamps) - min(timestamps)).total_seconds() / 3600
    
    return stats


# ============================================================================
# Export Functions
# ============================================================================

__all__ = [
    'haversine_distance',
    'haversine_vectorized',
    'deduplicate_detections',
    'update_hazard_confidence',
    'merge_duplicate_reports',
    'mark_repaired_hazards',
    'process_detection_batch',
    'calculate_cluster_statistics'
]
