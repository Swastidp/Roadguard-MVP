"""
Map view page for hazard visualization and analysis.

This module provides an interactive map interface for viewing and analyzing detected road hazards.
Simplified to show user location and hazards without clustering issues.
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime, timedelta
import io
from streamlit.components.v1 import html as st_html

# Import components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.components import mapping
from app.config import (
    DB_PATH,
    MAP_SETTINGS,
    CLASS_NAMES,
    SEVERITY_COLORS
)

# Extract map settings with correct keys
INITIAL_LAT = MAP_SETTINGS.get('center_lat', 28.6139)
INITIAL_LON = MAP_SETTINGS.get('center_lon', 77.2090)
INITIAL_ZOOM = MAP_SETTINGS.get('zoom', 12)

# Geolocation JavaScript to get user's current location
GEO_JS = """
<script>
const sendCoords = (lat, lon) => {
  const data = {"lat": lat, "lon": lon};
  window.parent.postMessage({isStreamlitMessage: true, type: "streamlit:setComponentValue", value: data}, "*");
};

navigator.geolocation.getCurrentPosition(
  (pos) => { sendCoords(pos.coords.latitude, pos.coords.longitude); },
  (err) => { 
    console.log("Geolocation error:", err); 
    sendCoords(null, null); 
  },
  {enableHighAccuracy: true, timeout: 8000, maximumAge: 0}
);
</script>
"""

def request_browser_location():
    """Request user's current location via browser geolocation API."""
    st_html(GEO_JS, height=0)


# ============================================================================
# Main Map View Page
# ============================================================================

def show():
    """Main map view page function."""
    
    # Page header
    st.title("Hazard Map")
    
    st.markdown("""
        Explore detected road hazards on an interactive map. The map centers on your current location 
        and shows nearby hazards with severity indicators.
    """)
    
    # Request browser geolocation
    if 'user_location' not in st.session_state:
        st.session_state.user_location = None
    
    request_browser_location()
    
    # Get user location from geolocation or use default
    center_latlon = (INITIAL_LAT, INITIAL_LON)
    
    # Check if we received location data from browser
    if hasattr(st.session_state, '_component_value') and isinstance(st.session_state._component_value, dict):
        lat = st.session_state._component_value.get('lat')
        lon = st.session_state._component_value.get('lon')
        if lat is not None and lon is not None:
            st.session_state.user_location = (float(lat), float(lon))
            st.success(f"📍 Location detected: {lat:.4f}, {lon:.4f}")
    
    if st.session_state.user_location:
        center_latlon = st.session_state.user_location
    else:
        st.info("📍 Using default location. Allow location access for better experience.")
    
    st.markdown("---")
    
    # Initialize session state
    if 'map_center' not in st.session_state:
        st.session_state.map_center = center_latlon
    if 'map_zoom' not in st.session_state:
        st.session_state.map_zoom = INITIAL_ZOOM
    
    # Sidebar filters
    filtered_df = create_sidebar_filters()
    
    # Main content
    if filtered_df is not None:
        # Display summary metrics
        display_summary_metrics(filtered_df)
        
        st.markdown("---")
        
        # Simplified tabs - removed heatmap and location search
        tab1, tab2 = st.tabs([
            "Map",
            "Statistics"
        ])
        
        with tab1:
            display_simple_map(filtered_df, center_latlon)
        
        with tab2:
            display_statistics(filtered_df)
        
        st.markdown("---")
        
        # Export section
        display_export_section(filtered_df)
    
    else:
        st.info("No hazard data available. Please check the database connection.")


# ============================================================================
# Sidebar Filters
# ============================================================================

def create_sidebar_filters() -> Optional[pd.DataFrame]:
    """Create sidebar filters and return filtered DataFrame."""
    
    with st.sidebar:
        st.header("Filters")
        
        # Load all hazards
        with st.spinner("Loading hazards from database..."):
            hazards_df = mapping.load_hazards_from_db(DB_PATH, status_filter='all')
        
        if hazards_df.empty:
            st.warning("No hazards found in database")
            return None
        
        # Status filter
        st.subheader("Status")
        status_options = ['active', 'resolved', 'pending']
        available_statuses = hazards_df['status'].unique().tolist()
        selected_statuses = st.multiselect(
            "Hazard Status",
            options=[s for s in status_options if s in available_statuses],
            default=['active'] if 'active' in available_statuses else available_statuses[:1],
            help="Filter by hazard status"
        )
        
        # Hazard type filter
        st.subheader("Hazard Types")
        available_types = hazards_df['class_name'].unique().tolist()
        selected_types = st.multiselect(
            "Select Types",
            options=available_types,
            default=available_types,
            help="Filter by hazard type",
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        # Severity filter
        st.subheader("Severity Levels")
        severity_options = ['critical', 'high', 'medium', 'low']
        available_severities = hazards_df['severity'].unique().tolist()
        selected_severities = st.multiselect(
            "Select Severities",
            options=[s for s in severity_options if s in available_severities],
            default=available_severities,
            help="Filter by severity level",
            format_func=lambda x: x.title()
        )
        
        # Date range filter
        st.subheader("Date Range")
        
        if 'timestamp' in hazards_df.columns and not hazards_df['timestamp'].isna().all():
            min_date = pd.to_datetime(hazards_df['timestamp']).min().date()
            max_date = pd.to_datetime(hazards_df['timestamp']).max().date()
            
            date_range = st.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                help="Filter hazards by detection date"
            )
            
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date = end_date = date_range
        else:
            start_date = end_date = None
        
        # Confidence filter
        st.subheader("Confidence")
        min_confidence = st.slider(
            "Minimum Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            help="Filter by minimum confidence score"
        )
        
        st.markdown("---")
        
        # Display filter summary
        st.info(f"Total hazards in database: {len(hazards_df)}")
        
        # Apply filters button
        if st.button("Refresh Data", use_container_width=True):
            st.rerun()
    
    # Apply filters
    filtered_df = hazards_df.copy()
    
    # Status filter
    if selected_statuses:
        filtered_df = filtered_df[filtered_df['status'].isin(selected_statuses)]
    
    # Type filter
    if selected_types:
        filtered_df = filtered_df[filtered_df['class_name'].isin(selected_types)]
    
    # Severity filter
    if selected_severities:
        filtered_df = filtered_df[filtered_df['severity'].isin(selected_severities)]
    
    # Date filter
    if start_date and end_date and 'timestamp' in filtered_df.columns:
        filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'])
        filtered_df = filtered_df[
            (filtered_df['timestamp'].dt.date >= start_date) &
            (filtered_df['timestamp'].dt.date <= end_date)
        ]
    
    # Confidence filter
    if 'confidence' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['confidence'] >= min_confidence]
    
    return filtered_df


# ============================================================================
# Summary Metrics
# ============================================================================

def display_summary_metrics(df: pd.DataFrame):
    """Display summary metrics for filtered hazards."""
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_hazards = len(df)
        st.metric(
            "Total Hazards",
            total_hazards,
            help="Total number of filtered hazards"
        )
    
    with col2:
        critical_count = len(df[df['severity'] == 'critical'])
        st.metric(
            "Critical",
            critical_count,
            delta=f"{critical_count/total_hazards*100:.0f}%" if total_hazards > 0 else "0%",
            delta_color="inverse",
            help="Critical severity hazards"
        )
    
    with col3:
        active_count = len(df[df['status'] == 'active'])
        st.metric(
            "Active",
            active_count,
            help="Hazards requiring attention"
        )
    
    with col4:
        if 'confidence' in df.columns and not df.empty:
            avg_confidence = df['confidence'].mean()
            st.metric(
                "Avg Confidence",
                f"{avg_confidence:.1%}",
                help="Average detection confidence"
            )
        else:
            st.metric("Avg Confidence", "N/A")
    
    with col5:
        if 'timestamp' in df.columns and not df.empty:
            latest = pd.to_datetime(df['timestamp']).max()
            hours_ago = (datetime.now() - latest).total_seconds() / 3600
            if hours_ago < 24:
                time_str = f"{int(hours_ago)}h ago"
            else:
                time_str = f"{int(hours_ago/24)}d ago"
            st.metric(
                "Last Detection",
                time_str,
                help="Time of most recent detection"
            )
        else:
            st.metric("Last Detection", "N/A")


# ============================================================================
# Simple Map (Replaces Cluster Map)
# ============================================================================

def display_simple_map(df: pd.DataFrame, center_latlon: tuple):
    """Display a simple hazard map centered at user's location without clustering options."""
    
    st.subheader("Hazard Map")
    
    if df.empty:
        st.warning("No hazards to display with current filters")
        # Still show map with user location
        if center_latlon:
            with st.spinner("Generating map..."):
                center_lat, center_lon = center_latlon
                
                # Create simple map with just user location
                m = folium.Map(
                    location=[center_lat, center_lon],
                    zoom_start=14
                )
                
                # Add user location marker
                folium.CircleMarker(
                    location=[center_lat, center_lon],
                    radius=10,
                    color="#2563EB",
                    fill=True,
                    fillColor="#3B82F6",
                    fillOpacity=0.7,
                    popup="📍 You are here",
                    tooltip="Your current location"
                ).add_to(m)
                
                # Display map
                mapping.display_map_in_streamlit(m, height=600)
        return
    
    # Create map with hazards
    with st.spinner("Generating map..."):
        center_lat, center_lon = center_latlon
        
        # Create hazard map without clustering to avoid blank map issue
        hazard_map = mapping.create_hazard_map(
            df,
            center_lat=center_lat,
            center_lon=center_lon,
            zoom=14,  # Slightly zoomed for local view
            use_cluster=False  # Force no clustering to avoid blank map bug
        )
        
        # Add user location marker if available
        if center_latlon:
            folium.CircleMarker(
                location=[center_lat, center_lon],
                radius=10,
                color="#2563EB",
                fill=True,
                fillColor="#3B82F6",
                fillOpacity=0.7,
                popup="📍 You are here",
                tooltip="Your current location"
            ).add_to(hazard_map)
        
        # Display map
        mapping.display_map_in_streamlit(hazard_map, height=600)
    
    # Simple info about the map
    st.info(f"Showing {len(df)} hazards. Click on markers for details.")


# ============================================================================
# Statistics (Unchanged)
# ============================================================================

def display_statistics(df: pd.DataFrame):
    """Display statistical analysis."""
    
    st.subheader("Statistical Analysis")
    
    if df.empty:
        st.warning("No data available for statistics")
        return
    
    # Hazards by type
    st.markdown("### Hazards by Type")
    type_counts = df['class_name'].value_counts()
    type_df = pd.DataFrame({
        'Type': [t.replace('_', ' ').title() for t in type_counts.index],
        'Count': type_counts.values
    })
    st.bar_chart(type_df.set_index('Type'))
    
    # Hazards by severity

    
    # Time series
    if 'timestamp' in df.columns and not df['timestamp'].isna().all():
        st.markdown("### Detections Over Time")
        
        df_time = df.copy()
        df_time['timestamp'] = pd.to_datetime(df_time['timestamp'])
        df_time['date'] = df_time['timestamp'].dt.date
        
        daily_counts = df_time.groupby('date').size().reset_index(name='count')
        daily_counts = daily_counts.set_index('date')
        
        st.line_chart(daily_counts)
    
    # Confidence distribution
    if 'confidence' in df.columns:
        st.markdown("### Confidence Distribution")
        
        fig_data = pd.DataFrame({
            'Confidence Range': ['0-0.5', '0.5-0.7', '0.7-0.85', '0.85-1.0'],
            'Count': [
                len(df[df['confidence'] < 0.5]),
                len(df[(df['confidence'] >= 0.5) & (df['confidence'] < 0.7)]),
                len(df[(df['confidence'] >= 0.7) & (df['confidence'] < 0.85)]),
                len(df[df['confidence'] >= 0.85])
            ]
        })
        st.bar_chart(fig_data.set_index('Confidence Range'))
    
    # Data table
    st.markdown("### Detailed Data")
    
    display_columns = ['id', 'class_name', 'severity', 'confidence', 'timestamp', 'status']
    display_columns = [col for col in display_columns if col in df.columns]
    
    display_df = df[display_columns].copy()
    
    # Format columns
    if 'class_name' in display_df.columns:
        display_df['class_name'] = display_df['class_name'].apply(
            lambda x: x.replace('_', ' ').title()
        )
    if 'confidence' in display_df.columns:
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
    if 'timestamp' in display_df.columns:
        display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ============================================================================
# Export Section (Unchanged)
# ============================================================================

def display_export_section(df: pd.DataFrame):
    """Display export options."""
    
    st.subheader("Export Data")
    
    if df.empty:
        st.warning("No data to export")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV export
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="Download as CSV",
            data=csv_data,
            file_name=f"hazards_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # JSON export
        json_data = df.to_json(orient='records', date_format='iso')
        
        st.download_button(
            label="Download as JSON",
            data=json_data,
            file_name=f"hazards_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    st.info(f"Exporting {len(df)} hazard record(s)")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    show()
