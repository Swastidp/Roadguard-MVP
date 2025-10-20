"""
Map view page for hazard visualization and analysis.

This module provides an interactive map interface for viewing, filtering,
and analyzing detected road hazards.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime, timedelta
import io

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

# Extract map settings
INITIAL_LAT = MAP_SETTINGS.get('INITIAL_LAT', 28.6139)
INITIAL_LON = MAP_SETTINGS.get('INITIAL_LON', 77.2090)
INITIAL_ZOOM = MAP_SETTINGS.get('INITIAL_ZOOM', 12)


# ============================================================================
# Main Map View Page
# ============================================================================

def show():
    """Main map view page function."""
    
    # Page header
    st.title("🗺️ Hazard Map Dashboard")
    
    st.markdown("""
        Explore detected road hazards on an interactive map. Filter by type, severity, 
        and location to analyze patterns and prioritize maintenance.
    """)
    
    st.markdown("---")
    
    # Initialize session state
    if 'map_center' not in st.session_state:
        st.session_state.map_center = (INITIAL_LAT, INITIAL_LON)
    if 'map_zoom' not in st.session_state:
        st.session_state.map_zoom = INITIAL_ZOOM
    
    # Sidebar filters
    filtered_df = create_sidebar_filters()
    
    # Main content
    if filtered_df is not None:
        # Display summary metrics
        display_summary_metrics(filtered_df)
        
        st.markdown("---")
        
        # Map and statistics tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🗺️ Cluster Map",
            "🔥 Heat Map",
            "📊 Statistics",
            "📍 Location Search"
        ])
        
        with tab1:
            display_cluster_map(filtered_df)
        
        with tab2:
            display_heatmap(filtered_df)
        
        with tab3:
            display_statistics(filtered_df)
        
        with tab4:
            display_location_search(filtered_df)
        
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
        st.header("🔍 Filters")
        
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
        st.info(f"📊 Total hazards in database: {len(hazards_df)}")
        
        # Apply filters button
        if st.button("🔄 Refresh Data", use_container_width=True):
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
# Cluster Map
# ============================================================================

def display_cluster_map(df: pd.DataFrame):
    """Display cluster map with individual markers."""
    
    st.subheader("🗺️ Hazard Cluster Map")
    
    if df.empty:
        st.warning("No hazards to display with current filters")
        return
    
    # Map options
    col1, col2 = st.columns([3, 1])
    
    with col2:
        use_clustering = st.checkbox(
            "Enable Clustering",
            value=True,
            help="Group nearby markers"
        )
        
        auto_fit = st.checkbox(
            "Auto-fit Bounds",
            value=True,
            help="Zoom to show all hazards"
        )
    
    # Create map
    with st.spinner("Generating map..."):
        # Calculate center if auto-fit
        if auto_fit and not df.empty:
            center_lat = df['latitude'].mean()
            center_lon = df['longitude'].mean()
            zoom = 12
        else:
            center_lat, center_lon = st.session_state.map_center
            zoom = st.session_state.map_zoom
        
        # Create hazard map
        hazard_map = mapping.create_hazard_map(
            df,
            center_lat=center_lat,
            center_lon=center_lon,
            zoom=zoom,
            use_cluster=use_clustering
        )
        
        # Display map
        mapping.display_map_in_streamlit(hazard_map, height=600)
    
    # Map legend
    st.markdown("### Legend")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"<span style='color: {SEVERITY_COLORS['critical']}'>● Critical</span>",
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"<span style='color: {SEVERITY_COLORS['high']}'>● High</span>",
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"<span style='color: {SEVERITY_COLORS['medium']}'>● Medium</span>",
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            f"<span style='color: {SEVERITY_COLORS['low']}'>● Low</span>",
            unsafe_allow_html=True
        )


# ============================================================================
# Heat Map
# ============================================================================

def display_heatmap(df: pd.DataFrame):
    """Display density heatmap."""
    
    st.subheader("🔥 Hazard Density Heatmap")
    
    st.markdown("""
        The heatmap shows hazard density across the area. 
        Brighter colors indicate higher concentration of hazards.
    """)
    
    if df.empty:
        st.warning("No hazards to display with current filters")
        return
    
    # Create heatmap
    with st.spinner("Generating heatmap..."):
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()
        
        heatmap = mapping.create_heatmap(
            df,
            center_lat=center_lat,
            center_lon=center_lon,
            zoom=12
        )
        
        # Display heatmap
        mapping.display_map_in_streamlit(heatmap, height=600)
    
    # Heatmap statistics
    st.markdown("### Density Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Data Points", len(df))
    
    with col2:
        # Calculate area coverage (rough estimate)
        if len(df) > 1:
            lat_range = df['latitude'].max() - df['latitude'].min()
            lon_range = df['longitude'].max() - df['longitude'].min()
            area_km2 = lat_range * lon_range * 111 * 111  # Rough conversion
            st.metric("Coverage Area", f"{area_km2:.1f} km²")
        else:
            st.metric("Coverage Area", "N/A")
    
    with col3:
        if len(df) > 1:
            density = len(df) / max(area_km2, 0.1)
            st.metric("Avg Density", f"{density:.1f} /km²")
        else:
            st.metric("Avg Density", "N/A")


# ============================================================================
# Statistics
# ============================================================================

def display_statistics(df: pd.DataFrame):
    """Display statistical analysis."""
    
    st.subheader("📊 Statistical Analysis")
    
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
    st.markdown("### Hazards by Severity")
    severity_counts = df['severity'].value_counts()
    
    col1, col2, col3, col4 = st.columns(4)
    for col, severity in zip([col1, col2, col3, col4], ['critical', 'high', 'medium', 'low']):
        with col:
            count = severity_counts.get(severity, 0)
            color = SEVERITY_COLORS.get(severity, '#3B82F6')
            st.markdown(
                f"<div style='text-align: center; padding: 20px; "
                f"background-color: {color}20; border-radius: 10px; "
                f"border-left: 5px solid {color};'>"
                f"<h2 style='color: {color}; margin: 0;'>{count}</h2>"
                f"<p style='margin: 5px 0 0 0;'>{severity.title()}</p>"
                f"</div>",
                unsafe_allow_html=True
            )
    
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
# Location Search
# ============================================================================

def display_location_search(df: pd.DataFrame):
    """Display location search interface."""
    
    st.subheader("📍 Search by Location")
    
    st.markdown("""
        Search for hazards near a specific location by entering coordinates 
        or using the current map center.
    """)
    
    # Input method selection
    search_method = st.radio(
        "Search Method",
        options=["Coordinates", "Current Map Center"],
        horizontal=True
    )
    
    if search_method == "Coordinates":
        col1, col2 = st.columns(2)
        
        with col1:
            search_lat = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=INITIAL_LAT,
                step=0.0001,
                format="%.6f"
            )
        
        with col2:
            search_lon = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=INITIAL_LON,
                step=0.0001,
                format="%.6f"
            )
    else:
        search_lat, search_lon = st.session_state.map_center
        st.info(f"Using map center: {search_lat:.6f}, {search_lon:.6f}")
    
    # Search radius
    search_radius = st.slider(
        "Search Radius (meters)",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        help="Radius to search for nearby hazards"
    )
    
    if st.button("🔍 Search", type="primary", use_container_width=True):
        with st.spinner("Searching for nearby hazards..."):
            nearby_hazards = mapping.get_hazards_in_radius(
                search_lat,
                search_lon,
                search_radius,
                DB_PATH,
                status_filter='active'
            )
        
        if nearby_hazards:
            st.success(f"Found {len(nearby_hazards)} hazard(s) within {search_radius}m")
            
            # Display results
            nearby_df = pd.DataFrame(nearby_hazards)
            
            # Format display
            display_cols = ['id', 'class_name', 'severity', 'distance', 'confidence']
            display_cols = [col for col in display_cols if col in nearby_df.columns]
            
            result_df = nearby_df[display_cols].copy()
            result_df['class_name'] = result_df['class_name'].apply(
                lambda x: x.replace('_', ' ').title()
            )
            result_df['distance'] = result_df['distance'].apply(lambda x: f"{x:.0f}m")
            result_df['confidence'] = result_df['confidence'].apply(lambda x: f"{x:.1%}")
            
            result_df.columns = ['ID', 'Type', 'Severity', 'Distance', 'Confidence']
            
            st.dataframe(result_df, use_container_width=True, hide_index=True)
            
            # Show on map
            if st.checkbox("Show on map"):
                mini_map = mapping.create_hazard_map(
                    nearby_df,
                    center_lat=search_lat,
                    center_lon=search_lon,
                    zoom=15,
                    use_cluster=False
                )
                mapping.display_map_in_streamlit(mini_map, height=400)
        
        else:
            st.info(f"No hazards found within {search_radius}m of the specified location")


# ============================================================================
# Export Section
# ============================================================================

def display_export_section(df: pd.DataFrame):
    """Display export options."""
    
    st.subheader("💾 Export Data")
    
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
            label="📥 Download as CSV",
            data=csv_data,
            file_name=f"hazards_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # JSON export
        json_data = df.to_json(orient='records', date_format='iso')
        
        st.download_button(
            label="📥 Download as JSON",
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
