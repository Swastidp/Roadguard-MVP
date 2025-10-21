"""
Mapping module for geospatial hazard visualization.

This module provides functions for loading hazard data from the database,
creating interactive maps with Folium, and displaying them in Streamlit.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import streamlit as st

try:
    import folium
    from folium import plugins
except ImportError:
    raise ImportError("folium package is required. Install with: pip install folium")

# Try to import streamlit-folium, fallback to st.components if not available
try:
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    st.warning("streamlit-folium not installed. Using fallback rendering. Install with: pip install streamlit-folium")

from ..config import (
    DB_PATH,
    MAP_SETTINGS,
    SEVERITY_COLORS
)

# Extract map settings with corrected keys
INITIAL_LAT = MAP_SETTINGS.get('center_lat', 28.6139)
INITIAL_LON = MAP_SETTINGS.get('center_lon', 77.2090)
INITIAL_ZOOM = MAP_SETTINGS.get('zoom', 12)


# ============================================================================
# Database Operations
# ============================================================================

def load_hazards_from_db(
    db_path: Path = DB_PATH,
    status_filter: str = 'active'
) -> pd.DataFrame:
    """
    Load hazard data from SQLite database.
    
    Args:
        db_path: Path to SQLite database file
        status_filter: Filter hazards by status ('active', 'resolved', 'all')
        
    Returns:
        DataFrame with hazard records containing:
        - id, latitude, longitude, class_name, severity, confidence, timestamp, status
        
    Raises:
        FileNotFoundError: If database file doesn't exist
        sqlite3.Error: If database query fails
    """
    db_path = Path(db_path)
    
    if not db_path.exists():
        print(f"⚠️ Database not found at: {db_path}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            'id', 'latitude', 'longitude', 'class_name', 
            'severity', 'confidence', 'timestamp', 'status'
        ])
    
    try:
        # Connect to database
        conn = sqlite3.connect(str(db_path))
        
        # Build query based on status filter
        if status_filter == 'all':
            query = """
                SELECT id, latitude, longitude, class_name, severity, 
                       confidence, timestamp, status
                FROM hazards
                ORDER BY timestamp DESC
            """
        else:
            query = f"""
                SELECT id, latitude, longitude, class_name, severity, 
                       confidence, timestamp, status
                FROM hazards
                WHERE status = '{status_filter}'
                ORDER BY timestamp DESC
            """
        
        # Load data into DataFrame
        df = pd.read_sql_query(query, conn)
        
        # Close connection
        conn.close()
        
        # Convert timestamp to datetime if present
        if 'timestamp' in df.columns and not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print(f"✅ Loaded {len(df)} hazard(s) from database")
        return df
        
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
        st.error(f"Error loading hazards from database: {str(e)}")
        return pd.DataFrame(columns=[
            'id', 'latitude', 'longitude', 'class_name', 
            'severity', 'confidence', 'timestamp', 'status'
        ])
    
    except Exception as e:
        print(f"❌ Unexpected error loading hazards: {e}")
        st.error(f"Unexpected error: {str(e)}")
        return pd.DataFrame(columns=[
            'id', 'latitude', 'longitude', 'class_name', 
            'severity', 'confidence', 'timestamp', 'status'
        ])


# ============================================================================
# Map Creation
# ============================================================================

def create_hazard_map(
    hazards_df: pd.DataFrame,
    center_lat: float = INITIAL_LAT,
    center_lon: float = INITIAL_LON,
    zoom: int = INITIAL_ZOOM,
    use_cluster: bool = True,
    add_layer_control: bool = False,
    add_legend: bool = False,
    add_location_button: bool = True
) -> folium.Map:
    """
    Create an interactive Folium map with hazard markers.
    
    Args:
        hazards_df: DataFrame with hazard data (must have lat, lon, severity, etc.)
        center_lat: Map center latitude
        center_lon: Map center longitude
        zoom: Initial zoom level
        use_cluster: Whether to use marker clustering for dense areas
        add_layer_control: Whether to add layer control widget
        add_legend: Whether to add severity legend
        add_location_button: Whether to add "My Location" button
        
    Returns:
        Folium Map object with hazard markers
    """
    # Create base map
    hazard_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles='OpenStreetMap',
        control_scale=True
    )
    
    # Add fullscreen button
    plugins.Fullscreen(
        position='topright',
        title='Expand',
        title_cancel='Exit',
        force_separate_button=True
    ).add_to(hazard_map)
    
    # Add location button (Google Maps style)
    if add_location_button:
        _add_location_button(hazard_map)
    
    # If no hazards, return map with location button
    if hazards_df.empty:
        print("ℹ️ No hazards to display on map")
        return hazard_map
    
    # Create marker cluster if enabled
    if use_cluster:
        marker_cluster = plugins.MarkerCluster(
            name='Hazard Clusters',
            overlay=True,
            control=add_layer_control,
            icon_create_function=None
        ).add_to(hazard_map)
    
    # Add hazard markers
    for idx, row in hazards_df.iterrows():
        try:
            # Extract data
            lat = row.get('latitude')
            lon = row.get('longitude')
            class_name = row.get('class_name', 'Unknown')
            severity = row.get('severity', 'low')
            confidence = row.get('confidence', 0.0)
            timestamp = row.get('timestamp', 'N/A')
            hazard_id = row.get('id', idx)
            
            # Skip invalid coordinates
            if pd.isna(lat) or pd.isna(lon):
                continue
            
            # Get color based on severity
            color = _get_marker_color(severity)
            
            # Calculate radius based on confidence (5-15 pixels)
            radius = 5 + (confidence * 10)
            
            # Format timestamp
            if isinstance(timestamp, pd.Timestamp):
                timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            else:
                timestamp_str = str(timestamp)
            
            # Create popup HTML
            popup_html = f"""
                <div style="font-family: Arial, sans-serif; width: 200px;">
                    <h4 style="margin: 0 0 10px 0; color: {SEVERITY_COLORS.get(severity, '#3B82F6')};">
                        {severity.upper()} Hazard
                    </h4>
                    <table style="width: 100%; font-size: 12px;">
                        <tr>
                            <td><strong>ID:</strong></td>
                            <td>{hazard_id}</td>
                        </tr>
                        <tr>
                            <td><strong>Type:</strong></td>
                            <td>{class_name.replace('_', ' ').title()}</td>
                        </tr>
                        <tr>
                            <td><strong>Severity:</strong></td>
                            <td>{severity.capitalize()}</td>
                        </tr>
                        <tr>
                            <td><strong>Confidence:</strong></td>
                            <td>{confidence:.2%}</td>
                        </tr>
                        <tr>
                            <td><strong>Detected:</strong></td>
                            <td>{timestamp_str}</td>
                        </tr>
                        <tr>
                            <td><strong>Location:</strong></td>
                            <td>{lat:.6f}, {lon:.6f}</td>
                        </tr>
                    </table>
                </div>
            """
            
            popup = folium.Popup(popup_html, max_width=250)
            
            # Create circle marker
            marker = folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                popup=popup,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                weight=2,
                opacity=0.9
            )
            
            # Add to cluster or map
            if use_cluster:
                marker.add_to(marker_cluster)
            else:
                marker.add_to(hazard_map)
            
        except Exception as e:
            print(f"⚠️ Error adding marker for hazard {idx}: {e}")
            continue
    
    # Only add layer control if requested
    if add_layer_control:
        folium.LayerControl(collapsed=True).add_to(hazard_map)
    
    # Only add legend if requested
    if add_legend:
        _add_map_legend(hazard_map)
    
    print(f"✅ Created map with {len(hazards_df)} hazard marker(s)")
    return hazard_map


def _add_location_button(map_object: folium.Map) -> None:
    """
    Add a location button to the map (Google Maps style).
    
    Args:
        map_object: Folium Map object to add location button to
    """
    location_button_html = """
    <div id="location-button" style="
        position: fixed;
        top: 80px;
        right: 10px;
        width: 40px;
        height: 40px;
        background-color: white;
        border: 2px solid rgba(0,0,0,0.2);
        border-radius: 6px;
        cursor: pointer;
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        transition: all 0.2s;
    " onclick="getMyLocation()" onmouseover="this.style.backgroundColor='#f5f5f5'" 
       onmouseout="this.style.backgroundColor='white'"
       title="Go to my location">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="3"></circle>
            <path d="M12 1v6m0 6v6"></path>
            <path d="m21 12-6 0m-6 0-6 0"></path>
        </svg>
    </div>

    <style>
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>

    <script>
    function getMyLocation() {
        const button = document.getElementById('location-button');
        
        // Add loading animation
        button.style.backgroundColor = '#1976d2';
        button.style.color = 'white';
        button.innerHTML = '<div style="width: 16px; height: 16px; border: 2px solid white; border-top: 2px solid transparent; border-radius: 50%; animation: spin 1s linear infinite;"></div>';
        
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                function(position) {
                    const lat = position.coords.latitude;
                    const lon = position.coords.longitude;
                    
                    // Try to get the Leaflet map instance
                    let mapInstance = null;
                    
                    // Look for map in window objects
                    for (let key in window) {
                        if (window[key] && typeof window[key] === 'object' && 
                            window[key].setView && typeof window[key].setView === 'function') {
                            mapInstance = window[key];
                            break;
                        }
                    }
                    
                    // Alternative: look for map in global scope
                    if (!mapInstance && typeof map !== 'undefined') {
                        mapInstance = map;
                    }
                    
                    // If still no map, try to find it in the DOM
                    if (!mapInstance) {
                        const mapContainer = document.querySelector('.folium-map');
                        if (mapContainer && mapContainer._leaflet_map) {
                            mapInstance = mapContainer._leaflet_map;
                        }
                    }
                    
                    if (mapInstance) {
                        // Center map on user location
                        mapInstance.setView([lat, lon], 16);
                        
                        // Remove existing user markers
                        if (window.userLocationMarker) {
                            mapInstance.removeLayer(window.userLocationMarker);
                        }
                        if (window.userAccuracyCircle) {
                            mapInstance.removeLayer(window.userAccuracyCircle);
                        }
                        
                        // Add new user location marker
                        if (typeof L !== 'undefined') {
                            window.userLocationMarker = L.circleMarker([lat, lon], {
                                radius: 8,
                                fillColor: '#1976d2',
                                color: '#ffffff',
                                weight: 3,
                                opacity: 1,
                                fillOpacity: 0.8
                            }).addTo(mapInstance).bindPopup('You are here');
                            
                            // Add accuracy circle
                            const accuracy = position.coords.accuracy || 100;
                            window.userAccuracyCircle = L.circle([lat, lon], {
                                radius: Math.min(accuracy, 500), // Cap at 500m for visibility
                                fillColor: '#1976d2',
                                fillOpacity: 0.1,
                                color: '#1976d2',
                                weight: 1
                            }).addTo(mapInstance);
                        }
                    }
                    
                    // Reset button
                    resetLocationButton(button, true);
                },
                function(error) {
                    console.log('Geolocation error:', error);
                    resetLocationButton(button, false);
                    
                    let errorMessage = 'Unable to get your location.';
                    switch(error.code) {
                        case error.PERMISSION_DENIED:
                            errorMessage = 'Location access denied. Please enable location permissions.';
                            break;
                        case error.POSITION_UNAVAILABLE:
                            errorMessage = 'Location information unavailable.';
                            break;
                        case error.TIMEOUT:
                            errorMessage = 'Location request timed out.';
                            break;
                    }
                    
                    // Show error message
                    if (typeof alert !== 'undefined') {
                        alert(errorMessage);
                    }
                },
                {
                    enableHighAccuracy: true,
                    timeout: 10000,
                    maximumAge: 60000
                }
            );
        } else {
            resetLocationButton(button, false);
            if (typeof alert !== 'undefined') {
                alert('Geolocation is not supported by this browser.');
            }
        }
    }
    
    function resetLocationButton(button, success) {
        if (success) {
            // Success state
            button.style.backgroundColor = '#4caf50';
            button.style.color = 'white';
            button.innerHTML = '✓';
            
            setTimeout(() => {
                button.style.backgroundColor = 'white';
                button.style.color = 'black';
                button.innerHTML = `
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="3"></circle>
                        <path d="M12 1v6m0 6v6"></path>
                        <path d="m21 12-6 0m-6 0-6 0"></path>
                    </svg>
                `;
            }, 1500);
        } else {
            // Error state
            button.style.backgroundColor = '#f44336';
            button.style.color = 'white';
            button.innerHTML = '✗';
            
            setTimeout(() => {
                button.style.backgroundColor = 'white';
                button.style.color = 'black';
                button.innerHTML = `
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="3"></circle>
                        <path d="M12 1v6m0 6v6"></path>
                        <path d="m21 12-6 0m-6 0-6 0"></path>
                    </svg>
                `;
            }, 2000);
        }
    }
    
    // Initialize map reference when DOM loads
    document.addEventListener('DOMContentLoaded', function() {
        // Try to find and store map reference
        setTimeout(function() {
            const mapContainers = document.querySelectorAll('.leaflet-container');
            if (mapContainers.length > 0) {
                const mapContainer = mapContainers[0];
                if (mapContainer._leaflet_map) {
                    window.leafletMap = mapContainer._leaflet_map;
                }
            }
        }, 1000);
    });
    </script>
    """
    
    map_object.get_root().html.add_child(folium.Element(location_button_html))


def create_heatmap(
    hazards_df: pd.DataFrame,
    center_lat: float = INITIAL_LAT,
    center_lon: float = INITIAL_LON,
    zoom: int = INITIAL_ZOOM
) -> folium.Map:
    """
    Create a heatmap visualization of hazard density.
    
    Args:
        hazards_df: DataFrame with hazard data (must have lat, lon, severity)
        center_lat: Map center latitude
        center_lon: Map center longitude
        zoom: Initial zoom level
        
    Returns:
        Folium Map object with heatmap layer
    """
    # Create base map
    heatmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles='OpenStreetMap',
        control_scale=True
    )
    
    # If no hazards, return empty map
    if hazards_df.empty:
        print("ℹ️ No hazards to display on heatmap")
        return heatmap
    
    # Prepare data for heatmap
    heat_data = []
    
    for idx, row in hazards_df.iterrows():
        try:
            lat = row.get('latitude')
            lon = row.get('longitude')
            severity = row.get('severity', 'low')
            
            # Skip invalid coordinates
            if pd.isna(lat) or pd.isna(lon):
                continue
            
            # Weight by severity
            weight = _get_severity_weight(severity)
            
            heat_data.append([lat, lon, weight])
            
        except Exception as e:
            print(f"⚠️ Error processing hazard {idx} for heatmap: {e}")
            continue
    
    # Add heatmap layer
    if heat_data:
        plugins.HeatMap(
            heat_data,
            name='Hazard Density',
            min_opacity=0.4,
            max_opacity=0.8,
            radius=25,
            blur=20,
            gradient={
                0.0: 'blue',
                0.3: 'lime',
                0.5: 'yellow',
                0.7: 'orange',
                1.0: 'red'
            }
        ).add_to(heatmap)
    
    # Add layer control
    folium.LayerControl().add_to(heatmap)
    
    # Add fullscreen button
    plugins.Fullscreen(
        position='topright',
        title='Expand',
        title_cancel='Exit',
        force_separate_button=True
    ).add_to(heatmap)
    
    print(f"✅ Created heatmap with {len(heat_data)} data point(s)")
    return heatmap


# ============================================================================
# Proximity Search
# ============================================================================

def get_hazards_in_radius(
    lat: float,
    lon: float,
    radius_meters: float = 1000,
    db_path: Path = DB_PATH,
    status_filter: str = 'active'
) -> List[Dict[str, Any]]:
    """
    Find hazards within a specified radius using Haversine formula.
    
    Args:
        lat: Center point latitude
        lon: Center point longitude
        radius_meters: Search radius in meters
        db_path: Path to SQLite database
        status_filter: Filter by hazard status
        
    Returns:
        List of hazard dictionaries with distance added
        
    Note:
        Uses Haversine formula for accurate distance calculation on Earth's surface.
    """
    db_path = Path(db_path)
    
    if not db_path.exists():
        print(f"⚠️ Database not found at: {db_path}")
        return []
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Haversine formula in SQL (distance in meters)
        # Earth radius = 6371 km = 6371000 meters
        query = f"""
            SELECT id, latitude, longitude, class_name, severity, 
                   confidence, timestamp, status,
                   (6371000 * acos(
                       cos(radians(?)) * cos(radians(latitude)) * 
                       cos(radians(longitude) - radians(?)) + 
                       sin(radians(?)) * sin(radians(latitude))
                   )) AS distance
            FROM hazards
            WHERE status = ?
            HAVING distance <= ?
            ORDER BY distance ASC
        """
        
        cursor.execute(query, (lat, lon, lat, status_filter, radius_meters))
        
        # Fetch results
        columns = [desc[0] for desc in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            hazard = dict(zip(columns, row))
            results.append(hazard)
        
        conn.close()
        
        print(f"✅ Found {len(results)} hazard(s) within {radius_meters}m radius")
        return results
        
    except sqlite3.Error as e:
        print(f"❌ Database error in proximity search: {e}")
        return []
    
    except Exception as e:
        print(f"❌ Error in proximity search: {e}")
        return []


# ============================================================================
# Streamlit Display
# ============================================================================

def display_map_in_streamlit(
    map_object: folium.Map,
    height: int = 600,
    width: Optional[int] = None
) -> None:
    """
    Display a Folium map in Streamlit.
    
    Args:
        map_object: Folium Map object to display
        height: Map height in pixels
        width: Map width in pixels (None for full width)
    """
    try:
        if FOLIUM_AVAILABLE:
            # Use streamlit-folium for better integration
            st_folium(
                map_object,
                width=width,
                height=height,
                returned_objects=[]
            )
        else:
            # Fallback to HTML rendering
            map_html = map_object._repr_html_()
            st.components.v1.html(
                map_html,
                height=height,
                scrolling=False
            )
            
    except Exception as e:
        st.error(f"Error displaying map: {str(e)}")
        print(f"❌ Error displaying map: {e}")


# ============================================================================
# Utility Functions
# ============================================================================

def _get_marker_color(severity: str) -> str:
    """
    Get marker color based on severity level.
    
    Args:
        severity: Severity level (critical, high, medium, low)
        
    Returns:
        Color string for Folium marker
    """
    color_map = {
        'critical': '#DC2626',  # Red
        'high': '#EA580C',      # Orange
        'medium': '#F59E0B',    # Amber
        'low': '#EAB308'        # Yellow
    }
    return color_map.get(severity.lower(), '#3B82F6')  # Default blue


def _get_severity_weight(severity: str) -> float:
    """
    Get numerical weight for severity level (for heatmap).
    
    Args:
        severity: Severity level
        
    Returns:
        Weight value (0.0-1.0)
    """
    weight_map = {
        'critical': 1.0,
        'high': 0.7,
        'medium': 0.5,
        'low': 0.3
    }
    return weight_map.get(severity.lower(), 0.3)


def _add_map_legend(map_object: folium.Map) -> None:
    """
    Add a legend to the map showing severity levels.
    
    Args:
        map_object: Folium Map object to add legend to
    """
    legend_html = """
    <div style="
        position: fixed;
        bottom: 50px;
        left: 50px;
        width: 180px;
        background-color: white;
        border: 2px solid grey;
        border-radius: 5px;
        z-index: 9999;
        font-size: 14px;
        padding: 10px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
    ">
        <h4 style="margin: 0 0 10px 0; font-size: 16px;">Severity Legend</h4>
        <div style="margin: 5px 0;">
            <span style="background-color: #DC2626; width: 20px; height: 20px; 
                         display: inline-block; border-radius: 50%; margin-right: 5px;">
            </span>
            Critical
        </div>
        <div style="margin: 5px 0;">
            <span style="background-color: #EA580C; width: 20px; height: 20px; 
                         display: inline-block; border-radius: 50%; margin-right: 5px;">
            </span>
            High
        </div>
        <div style="margin: 5px 0;">
            <span style="background-color: #F59E0B; width: 20px; height: 20px; 
                         display: inline-block; border-radius: 50%; margin-right: 5px;">
            </span>
            Medium
        </div>
        <div style="margin: 5px 0;">
            <span style="background-color: #EAB308; width: 20px; height: 20px; 
                         display: inline-block; border-radius: 50%; margin-right: 5px;">
            </span>
            Low
        </div>
    </div>
    """
    map_object.get_root().html.add_child(folium.Element(legend_html))


def calculate_map_bounds(hazards_df: pd.DataFrame) -> Optional[List[List[float]]]:
    """
    Calculate map bounds to fit all hazards.
    
    Args:
        hazards_df: DataFrame with hazard locations
        
    Returns:
        Bounds as [[south, west], [north, east]] or None if empty
    """
    if hazards_df.empty:
        return None
    
    try:
        min_lat = hazards_df['latitude'].min()
        max_lat = hazards_df['latitude'].max()
        min_lon = hazards_df['longitude'].min()
        max_lon = hazards_df['longitude'].max()
        
        return [[min_lat, min_lon], [max_lat, max_lon]]
        
    except Exception as e:
        print(f"⚠️ Error calculating map bounds: {e}")
        return None


# ============================================================================
# Export Functions
# ============================================================================

__all__ = [
    'load_hazards_from_db',
    'create_hazard_map',
    'create_heatmap',
    'get_hazards_in_radius',
    'display_map_in_streamlit',
    'calculate_map_bounds'
]
