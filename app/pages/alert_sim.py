"""
Alert simulator page for testing hazard warning logic.

This module provides an interactive interface for testing and visualizing
the alert system under different scenarios.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import time

# Import components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.components import alerts, mapping, utils
from app.config import (
    DB_PATH,
    MAP_SETTINGS,
    ALERT_SETTINGS,
    CLASS_NAMES,
    SEVERITY_COLORS
)

# Extract settings
INITIAL_LAT = MAP_SETTINGS.get('INITIAL_LAT', 28.6139)
INITIAL_LON = MAP_SETTINGS.get('INITIAL_LON', 77.2090)


# ============================================================================
# Main Alert Simulator Page
# ============================================================================

def show():
    """Main alert simulator page function."""
    
    # Page header
    st.title("⚠️ Alert Simulator")
    
    st.markdown("""
        Test the **RoadGuard Alert System** to understand how warnings are calculated 
        based on vehicle speed, road conditions, and hazard severity.
        
        This simulator helps you:
        - 🚗 Calculate safe warning distances
        - 📊 Visualize braking zones
        - 🎯 Test multi-hazard scenarios
        - ⚙️ Fine-tune alert parameters
    """)
    
    st.markdown("---")
    
    # Load hazards
    hazards_df = mapping.load_hazards_from_db(DB_PATH, status_filter='active')
    
    if hazards_df.empty:
        st.warning("No active hazards found in database. Please add hazards to test alerts.")
        display_demo_mode()
        return
    
    # Tabs for different modes
    tab1, tab2, tab3 = st.tabs([
        "🎯 Single Hazard Test",
        "🛣️ Route Simulation",
        "⚙️ Advanced Settings"
    ])
    
    with tab1:
        single_hazard_simulator(hazards_df)
    
    with tab2:
        route_simulator(hazards_df)
    
    with tab3:
        advanced_settings()


# ============================================================================
# Single Hazard Simulator
# ============================================================================

def single_hazard_simulator(hazards_df: pd.DataFrame):
    """Simulate alerts for a single hazard."""
    
    st.subheader("🎯 Single Hazard Alert Test")
    
    # Vehicle parameters
    st.markdown("### 🚗 Vehicle Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        vehicle_speed = st.number_input(
            "Current Speed (km/h)",
            min_value=0,
            max_value=120,
            value=60,
            step=5,
            help="Vehicle speed in kilometers per hour"
        )
    
    with col2:
        road_condition = st.selectbox(
            "Road Condition",
            options=['dry', 'wet', 'icy'],
            index=0,
            help="Current road surface condition"
        )
    
    with col3:
        vehicle_heading = st.number_input(
            "Vehicle Heading (°)",
            min_value=0,
            max_value=360,
            value=0,
            step=15,
            help="Direction of travel (0° = North)"
        )
    
    st.markdown("---")
    
    # Hazard selection
    st.markdown("### 🎯 Select Hazard")
    
    # Create hazard options
    hazard_options = {}
    for idx, row in hazards_df.iterrows():
        label = (
            f"ID {row['id']}: {row['class_name'].replace('_', ' ').title()} "
            f"({row['severity'].title()}) - {row['latitude']:.6f}, {row['longitude']:.6f}"
        )
        hazard_options[label] = row
    
    selected_hazard_label = st.selectbox(
        "Choose a hazard to test",
        options=list(hazard_options.keys()),
        help="Select a hazard from the database"
    )
    
    selected_hazard = hazard_options[selected_hazard_label]
    
    # Display hazard details
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Hazard Details")
        
        detail_col1, detail_col2, detail_col3 = st.columns(3)
        
        with detail_col1:
            st.metric("Type", selected_hazard['class_name'].replace('_', ' ').title())
        
        with detail_col2:
            severity_color = SEVERITY_COLORS.get(selected_hazard['severity'], '#3B82F6')
            st.markdown(
                f"**Severity:** <span style='color: {severity_color}; font-size: 1.2em;'>"
                f"{selected_hazard['severity'].title()}</span>",
                unsafe_allow_html=True
            )
        
        with detail_col3:
            st.metric("Confidence", f"{selected_hazard['confidence']:.1%}")
    
    with col2:
        # Vehicle position input
        st.markdown("#### Vehicle Position")
        vehicle_lat = st.number_input(
            "Latitude",
            value=float(selected_hazard['latitude']) + 0.001,
            format="%.6f",
            key="vehicle_lat"
        )
        vehicle_lon = st.number_input(
            "Longitude",
            value=float(selected_hazard['longitude']),
            format="%.6f",
            key="vehicle_lon"
        )
    
    st.markdown("---")
    
    # Calculate alert
    if st.button("🚨 Calculate Warning", type="primary", use_container_width=True):
        simulate_single_alert(
            selected_hazard,
            vehicle_speed,
            road_condition,
            (vehicle_lat, vehicle_lon),
            vehicle_heading
        )


def simulate_single_alert(
    hazard: pd.Series,
    speed: float,
    road_condition: str,
    vehicle_pos: Tuple[float, float],
    heading: float
):
    """Simulate and display alert for a single hazard."""
    
    with st.spinner("Calculating alert parameters..."):
        # Calculate distance to hazard
        distance = alerts.calculate_distance_haversine(
            vehicle_pos[0], vehicle_pos[1],
            hazard['latitude'], hazard['longitude']
        )
        
        # Calculate bearing to hazard
        bearing = alerts.calculate_bearing(
            vehicle_pos[0], vehicle_pos[1],
            hazard['latitude'], hazard['longitude']
        )
        
        # Check if heading towards hazard
        angle_diff = abs(bearing - heading)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        is_approaching = angle_diff <= 30  # Within 30° of heading
        
        # Calculate warning distance
        warning_distance = alerts.calculate_warning_distance(
            speed,
            hazard['severity'],
            road_condition
        )
        
        # Calculate safe speed
        safe_speed = alerts.calculate_safe_speed(
            distance,
            hazard['severity'],
            road_condition
        )
        
        # Generate alert
        hazard_dict = hazard.to_dict()
        alert = alerts.generate_alert_message(
            hazard_dict,
            distance,
            speed,
            road_condition
        )
    
    # Display results
    st.markdown("### 📊 Alert Calculation Results")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Distance to Hazard", f"{distance:.0f}m")
    
    with col2:
        st.metric("Warning Distance", f"{warning_distance:.0f}m")
    
    with col3:
        st.metric("Safe Speed", f"{safe_speed:.0f} km/h")
    
    with col4:
        st.metric("Time to Impact", f"{alert.time_to_impact:.1f}s")
    
    st.markdown("---")
    
    # Alert message with color coding
    urgency_colors = {
        alerts.UrgencyLevel.CRITICAL: "#DC2626",
        alerts.UrgencyLevel.HIGH: "#EA580C",
        alerts.UrgencyLevel.MODERATE: "#F59E0B",
        alerts.UrgencyLevel.LOW: "#3B82F6",
        alerts.UrgencyLevel.INFO: "#10B981"
    }
    
    alert_color = urgency_colors.get(alert.urgency, "#3B82F6")
    
    st.markdown(
        f"""
        <div style="
            background-color: {alert_color}20;
            border-left: 5px solid {alert_color};
            border-radius: 5px;
            padding: 20px;
            margin: 20px 0;
        ">
            <h3 style="color: {alert_color}; margin-top: 0;">
                {alert.urgency.value.upper()} ALERT
            </h3>
            <p style="font-size: 1.1em; margin: 10px 0;">
                {alert.message}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Additional context
    st.markdown("### 📍 Situation Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Directional Analysis:**")
        st.write(f"- Bearing to hazard: {bearing:.1f}°")
        st.write(f"- Vehicle heading: {heading:.1f}°")
        st.write(f"- Angle difference: {angle_diff:.1f}°")
        
        if is_approaching:
            st.success("✅ Vehicle is approaching hazard")
        else:
            st.info("ℹ️ Vehicle is not heading towards hazard")
    
    with col2:
        st.markdown("**Safety Margin:**")
        
        if distance > warning_distance:
            margin = distance - warning_distance
            st.success(f"✅ Safe margin: {margin:.0f}m")
        elif distance > warning_distance * 0.5:
            margin = warning_distance - distance
            st.warning(f"⚠️ Reduced margin: {margin:.0f}m deficit")
        else:
            margin = warning_distance - distance
            st.error(f"🚨 Critical: {margin:.0f}m deficit")
        
        speed_diff = speed - safe_speed
        if speed_diff > 0:
            st.warning(f"Reduce speed by {speed_diff:.0f} km/h")
        else:
            st.success(f"Current speed is safe")
    
    # Visualization
    st.markdown("### 📈 Warning Distance Curve")
    visualize_warning_curve(hazard['severity'], road_condition, speed, warning_distance)


# ============================================================================
# Visualization
# ============================================================================

def visualize_warning_curve(severity: str, road_condition: str, current_speed: float, current_warning: float):
    """Visualize warning distance vs speed curve."""
    
    # Generate data points
    speeds = np.arange(0, 121, 5)
    warning_distances = [
        alerts.calculate_warning_distance(s, severity, road_condition)
        for s in speeds
    ]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Speed (km/h)': speeds,
        'Warning Distance (m)': warning_distances
    })
    
    # Display chart
    st.line_chart(df.set_index('Speed (km/h)'))
    
    # Add current position marker
    st.markdown(
        f"**Current Position:** {current_speed:.0f} km/h → {current_warning:.0f}m warning distance"
    )
    
    # Safety zones
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            "<div style='background-color: #10B98120; padding: 10px; border-radius: 5px;'>"
            "<strong style='color: #10B981;'>🟢 Safe Zone</strong><br>"
            "Distance > Warning Distance"
            "</div>",
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            "<div style='background-color: #F59E0B20; padding: 10px; border-radius: 5px;'>"
            "<strong style='color: #F59E0B;'>🟡 Caution Zone</strong><br>"
            "Distance ≈ Warning Distance"
            "</div>",
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            "<div style='background-color: #DC262620; padding: 10px; border-radius: 5px;'>"
            "<strong style='color: #DC2626;'>🔴 Danger Zone</strong><br>"
            "Distance < Warning Distance"
            "</div>",
            unsafe_allow_html=True
        )


# ============================================================================
# Route Simulator
# ============================================================================

def route_simulator(hazards_df: pd.DataFrame):
    """Simulate driving along a route with multiple hazards."""
    
    st.subheader("🛣️ Route Simulation")
    
    st.markdown("""
        Simulate driving along a route and see how alerts are triggered 
        as you approach multiple hazards.
    """)
    
    # Route parameters
    col1, col2 = st.columns(2)
    
    with col1:
        route_speed = st.slider(
            "Cruising Speed (km/h)",
            min_value=20,
            max_value=100,
            value=60,
            step=10,
            help="Constant speed for simulation"
        )
    
    with col2:
        route_condition = st.selectbox(
            "Road Condition",
            options=['dry', 'wet', 'icy'],
            index=0,
            key="route_condition"
        )
    
    # Start/End points
    st.markdown("#### Define Route")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Start Point**")
        start_lat = st.number_input("Start Latitude", value=INITIAL_LAT, format="%.6f", key="start_lat")
        start_lon = st.number_input("Start Longitude", value=INITIAL_LON, format="%.6f", key="start_lon")
    
    with col2:
        st.markdown("**End Point**")
        end_lat = st.number_input("End Latitude", value=INITIAL_LAT + 0.01, format="%.6f", key="end_lat")
        end_lon = st.number_input("End Longitude", value=INITIAL_LON + 0.01, format="%.6f", key="end_lon")
    
    # Simulation controls
    num_steps = st.slider(
        "Simulation Steps",
        min_value=10,
        max_value=100,
        value=50,
        help="Number of position updates along route"
    )
    
    if st.button("🚀 Start Route Simulation", type="primary", use_container_width=True):
        run_route_simulation(
            hazards_df,
            (start_lat, start_lon),
            (end_lat, end_lon),
            route_speed,
            route_condition,
            num_steps
        )


def run_route_simulation(
    hazards_df: pd.DataFrame,
    start: Tuple[float, float],
    end: Tuple[float, float],
    speed: float,
    road_condition: str,
    num_steps: int
):
    """Run route simulation and display alerts."""
    
    st.markdown("### 🎬 Simulation Running...")
    
    # Generate route waypoints
    lats = np.linspace(start[0], end[0], num_steps)
    lons = np.linspace(start[1], end[1], num_steps)
    
    # Calculate total distance
    total_distance = alerts.calculate_distance_haversine(
        start[0], start[1], end[0], end[1]
    )
    
    st.info(f"Route distance: {total_distance:.0f}m | Speed: {speed} km/h")
    
    # Simulation results
    alert_timeline = []
    triggered_hazards = set()
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    alert_container = st.container()
    
    # Simulate
    for step, (lat, lon) in enumerate(zip(lats, lons)):
        position = (lat, lon)
        distance_traveled = (step / num_steps) * total_distance
        
        status_text.text(f"Position: {distance_traveled:.0f}m / {total_distance:.0f}m")
        
        # Check each hazard
        for idx, hazard in hazards_df.iterrows():
            hazard_id = hazard['id']
            
            # Calculate distance to hazard
            dist = alerts.calculate_distance_haversine(
                lat, lon,
                hazard['latitude'], hazard['longitude']
            )
            
            # Calculate warning distance
            warning_dist = alerts.calculate_warning_distance(
                speed, hazard['severity'], road_condition
            )
            
            # Check if alert should trigger
            if dist <= warning_dist and hazard_id not in triggered_hazards:
                triggered_hazards.add(hazard_id)
                
                # Generate alert
                alert = alerts.generate_alert_message(
                    hazard.to_dict(),
                    dist,
                    speed,
                    road_condition
                )
                
                alert_timeline.append({
                    'step': step,
                    'distance_traveled': distance_traveled,
                    'hazard_id': hazard_id,
                    'hazard_type': hazard['class_name'],
                    'severity': hazard['severity'],
                    'distance_to_hazard': dist,
                    'urgency': alert.urgency.value,
                    'message': alert.message
                })
        
        progress_bar.progress((step + 1) / num_steps)
        time.sleep(0.05)  # Slow down for visualization
    
    status_text.text("✅ Simulation Complete!")
    
    # Display results
    st.markdown("---")
    st.markdown("### 📋 Alert Timeline")
    
    if alert_timeline:
        st.success(f"🚨 {len(alert_timeline)} alert(s) triggered during simulation")
        
        # Display each alert
        for alert_data in alert_timeline:
            urgency_color = {
                'critical': '#DC2626',
                'high': '#EA580C',
                'moderate': '#F59E0B',
                'low': '#3B82F6',
                'info': '#10B981'
            }.get(alert_data['urgency'], '#3B82F6')
            
            st.markdown(
                f"""
                <div style="
                    background-color: {urgency_color}15;
                    border-left: 4px solid {urgency_color};
                    padding: 10px;
                    margin: 10px 0;
                    border-radius: 5px;
                ">
                    <strong>@ {alert_data['distance_traveled']:.0f}m:</strong> 
                    {alert_data['message']}
                    <br>
                    <small>Hazard ID: {alert_data['hazard_id']} | 
                    Distance: {alert_data['distance_to_hazard']:.0f}m</small>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Summary chart
        st.markdown("### 📊 Alert Distribution")
        timeline_df = pd.DataFrame(alert_timeline)
        
        severity_counts = timeline_df['severity'].value_counts()
        st.bar_chart(severity_counts)
        
    else:
        st.info("✅ No alerts triggered during simulation. Route is clear!")


# ============================================================================
# Advanced Settings
# ============================================================================

def advanced_settings():
    """Display and modify advanced alert settings."""
    
    st.subheader("⚙️ Advanced Alert Settings")
    
    st.markdown("""
        Fine-tune the alert system parameters to customize warning behavior.
        These settings affect how warnings are calculated and displayed.
    """)
    
    st.warning("⚠️ Modifying these settings will affect all alert calculations. Use with caution.")
    
    # Physics parameters
    st.markdown("### 🔬 Physics Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        reaction_time = st.slider(
            "Driver Reaction Time (seconds)",
            min_value=0.5,
            max_value=3.0,
            value=1.5,
            step=0.1,
            help="Average time for driver to react to hazard"
        )
        
        st.info(f"Reaction distance at 60 km/h: {(60/3.6) * reaction_time:.1f}m")
    
    with col2:
        friction_coeff = st.slider(
            "Road Friction Coefficient",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Coefficient for dry asphalt (0.7), wet (0.5), icy (0.2)"
        )
        
        st.info("Standard values: Dry=0.7, Wet=0.5, Ice=0.2")
    
    # Safety margins
    st.markdown("### 🛡️ Safety Margins")
    
    st.markdown("**Severity Multipliers** (increases warning distance)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        critical_mult = st.number_input("Critical", value=1.5, min_value=1.0, max_value=3.0, step=0.1)
    with col2:
        high_mult = st.number_input("High", value=1.2, min_value=1.0, max_value=2.0, step=0.1)
    with col3:
        medium_mult = st.number_input("Medium", value=1.0, min_value=0.8, max_value=1.5, step=0.1)
    with col4:
        low_mult = st.number_input("Low", value=0.8, min_value=0.5, max_value=1.2, step=0.1)
    
    # Minimum warning distance
    min_warning = st.slider(
        "Minimum Warning Distance (meters)",
        min_value=20,
        max_value=200,
        value=50,
        step=10,
        help="Minimum distance regardless of speed"
    )
    
    st.markdown("---")
    
    # Test configuration
    st.markdown("### 🧪 Test Configuration")
    
    test_speed = st.slider("Test Speed (km/h)", 0, 120, 60, 5)
    test_severity = st.selectbox("Test Severity", ['critical', 'high', 'medium', 'low'])
    
    if st.button("Calculate with Custom Settings", type="primary"):
        # Calculate with custom parameters (simplified)
        speed_ms = test_speed / 3.6
        reaction_dist = speed_ms * reaction_time
        braking_dist = (test_speed ** 2) / (250 * friction_coeff)
        
        multipliers = {
            'critical': critical_mult,
            'high': high_mult,
            'medium': medium_mult,
            'low': low_mult
        }
        
        mult = multipliers.get(test_severity, 1.0)
        warning_dist = max((reaction_dist + braking_dist) * mult, min_warning)
        
        st.success(f"**Warning Distance:** {warning_dist:.0f}m")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Reaction Distance", f"{reaction_dist:.0f}m")
        with col2:
            st.metric("Braking Distance", f"{braking_dist:.0f}m")
        with col3:
            st.metric("Safety Margin", f"{(mult-1)*100:.0f}%")


# ============================================================================
# Demo Mode
# ============================================================================

def display_demo_mode():
    """Display demo mode when no hazards available."""
    
    st.markdown("### 📚 Demo Mode")
    
    st.info("No hazards in database. Try these example scenarios:")
    
    # Example scenarios
    scenarios = [
        {
            'name': 'Urban Pothole',
            'speed': 40,
            'distance': 50,
            'severity': 'high',
            'condition': 'dry'
        },
        {
            'name': 'Highway Crack',
            'speed': 100,
            'distance': 150,
            'severity': 'medium',
            'condition': 'dry'
        },
        {
            'name': 'Wet Road Hazard',
            'speed': 60,
            'distance': 80,
            'severity': 'critical',
            'condition': 'wet'
        }
    ]
    
    for scenario in scenarios:
        with st.expander(f"📋 {scenario['name']}"):
            warning_dist = alerts.calculate_warning_distance(
                scenario['speed'],
                scenario['severity'],
                scenario['condition']
            )
            
            safe_speed = alerts.calculate_safe_speed(
                scenario['distance'],
                scenario['severity'],
                scenario['condition']
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Speed:** {scenario['speed']} km/h")
                st.write(f"**Distance:** {scenario['distance']}m")
                st.write(f"**Severity:** {scenario['severity'].title()}")
            
            with col2:
                st.metric("Warning Distance", f"{warning_dist:.0f}m")
                st.metric("Safe Speed", f"{safe_speed:.0f} km/h")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    show()
