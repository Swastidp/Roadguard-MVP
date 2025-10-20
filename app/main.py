"""
RoadGuard: Road Hazard Detection System by Team Autono Minds
Main entry point for the Streamlit application.

Team: Autono Minds | VW Hackathon 2025
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.config import *

# Import page modules
try:
    from app.pages import demo, map_view, alert_sim, privacy_test, metrics
except ImportError as e:
    st.error(f"Error importing page modules: {e}")
    st.stop()


# ============================================================================
# Page Configuration
# ============================================================================

def configure_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="RoadGuard - Team Autono Minds",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/Swastidp/Roadguard-MVP',
            'Report a bug': 'https://github.com/Swastidp/Roadguard-MVP/issues',
            'About': """
            # RoadGuard 🚗
            
            Team Autono Minds - VW Hackathon 2025
            YOLOv11 + SE Attention for Road Hazard Detection
            
            **Performance**: 50.56% mAP@0.5
            """
        }
    )


# ============================================================================
# Minimal CSS - ONLY HIDE STREAMLIT BRANDING
# ============================================================================

def apply_custom_css():
    """Apply minimal CSS - only hide Streamlit branding, keep all default styling."""
    st.markdown("""
        <style>
        /* Hide Streamlit branding only */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Let everything else use Streamlit defaults */
        </style>
    """, unsafe_allow_html=True)


# ============================================================================
# Simple Sidebar - Streamlit Default Styling
# ============================================================================

def create_sidebar():
    """Create sidebar using pure Streamlit components with default styling."""
    with st.sidebar:
        # Simple team branding with Streamlit defaults
        st.title("🚗 RoadGuard")
        st.subheader("AI-Powered Hazard Detection")
        
        # Team info using Streamlit default styling
        st.info("**Team Autono Minds**")
        st.caption("VW Hackathon 2025 | YOLOv11 + SE Attention")
        
        st.markdown("---")
        
        # Navigation with default radio buttons
        st.markdown("### 🧭 Navigation")
        page = st.radio(
            "Select a page:",
            options=[
                "🏠 Home",
                "🎥 Live Demo", 
                "🗺️ Hazard Map",
                "⚠️ Alert Simulator",
                "🔒 Privacy Test",
                "📊 Metrics Dashboard"
            ],
            key="navigation"
        )
        
        st.markdown("---")
        
        # Model performance with default metrics
        st.markdown("### 🏆 Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("mAP@0.5", "50.56%")
        with col2:
            st.metric("Epochs", "65")
        
        st.markdown("---")
        
        # Simple footer with default styling
        st.markdown("### 📍 Info")
        st.success("**VW Hackathon 2025**")
        st.info("**Team Autono Minds**")
        st.caption("YOLOv11n + SE Attention  \nCustom Trained Model")
        
        # GitHub link
        st.markdown("---")
        st.markdown("⭐ [Star us on GitHub](https://github.com/Swastidp/Roadguard-MVP)")
        st.caption("Built with ❤️ using Streamlit")
        
        return page


# ============================================================================
# Simple Home Page - Pure Streamlit Components
# ============================================================================

def display_home():
    """Display home page using only Streamlit default components."""
    
    # Simple headers
    st.title("🏆 Team Autono Minds - RoadGuard")
    st.subheader("AI-Powered Road Hazard Detection")
    
    # Team info
    st.info("**Team Autono Minds** | VW Hackathon 2025 | YOLOv11 + SE Attention")
    
    # Project overview
    st.markdown("### 🎯 Project Overview")
    
    st.markdown("""
    Custom **YOLOv11 + SE Attention** architecture achieving **50.56% mAP@0.5** on road hazard detection 
    with **65 epochs** of dedicated training on 6,439 images.
    """)
    
    st.success("🔍 Real-time detection • 🔒 Privacy-compliant • 🗺️ Spatial intelligence")
    
    # Features
    st.markdown("## 🎯 Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔍 Real-time Detection")
        st.markdown("""
        Custom YOLOv11 + SE Attention model detects potholes, cracks, 
        and road damage instantly with **50.56% mAP@0.5** accuracy.
        """)
        st.success("⚡ <45ms inference time")
    
    with col2:
        st.markdown("### 🗺️ Interactive Mapping")
        st.markdown("""
        Visualize detected hazards on interactive maps with DBSCAN clustering 
        and spatial deduplication for comprehensive road monitoring.
        """)
        st.info("📍 GPS-based alert zones")
    
    with col3:
        st.markdown("### 🔒 Privacy Protected")
        st.markdown("""
        Automatic face and license plate detection with real-time blurring 
        ensures GDPR compliance and user privacy protection.
        """)
        st.warning("🛡️ 100% GDPR compliant")
    
    st.markdown("---")
    
    # Training results
    st.markdown("## 📊 Team Autono Minds - Training Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Overall mAP@0.5", "50.56%", "YOLOv11 + SE Attention")
    
    with col2:
        st.metric("Best Class", "69.97%", "Transverse Crack")
    
    with col3:
        st.metric("Training Dataset", "6,439", "Images (65 epochs)")
    
    with col4:
        st.metric("Model Size", "2.6M", "Parameters")
    
    st.markdown("---")
    
    # Performance table
    st.markdown("## 🎯 Per-Class Performance Analysis")
    
    import pandas as pd
    
    performance_data = {
        'Class': ['🛣️ Transverse Crack', '🕳️ Pothole', '📏 Longitudinal Crack', '🕸️ Alligator Crack'],
        'mAP@0.5': ['69.97%', '62.34%', '59.84%', '10.10%'],
        'Difficulty': ['✅ Easy', '🟡 Medium', '🟡 Medium', '🔴 Very Hard']
    }
    
    df = pd.DataFrame(performance_data)
    st.dataframe(df, use_container_width=True)
    
    # Analysis insights
    st.markdown("### 🔍 Analysis Insights")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.success("✅ **Transverse Cracks** perform best (69.97%) - clear perpendicular patterns")
        st.info("🎯 **Potholes** show solid detection (62.34%) - distinct circular shapes")
        st.info("📏 **Longitudinal Cracks** achieve good accuracy (59.84%) - parallel patterns")
        st.warning("⚠️ **Alligator Cracks** need improvement (10.10%) - complex interconnected patterns")
    
    with col2:
        st.metric("Training Time", "~2 hours", "RTX 3050 GPU")
        st.metric("Dataset Split", "80/20", "Train/Validation")
        st.metric("Best Epoch", "65", "Final model")
    
    st.markdown("---")
    
    # Getting started
    st.markdown("## 🚀 Getting Started")
    
    st.markdown("""
    1. **🎥 Live Demo** - Test our YOLOv11 + SE Attention model with your own images or webcam feed
    2. **🗺️ Hazard Map** - View detected hazards on an interactive map with clustering  
    3. **⚠️ Alert Simulator** - Simulate physics-based driver alerts based on vehicle speed
    4. **🔒 Privacy Test** - Test GDPR compliance with face and license plate blurring
    5. **📊 Metrics Dashboard** - View detailed performance metrics and benchmarks
    """)
    
    st.success("👈 **Use the sidebar navigation** to explore different features of RoadGuard!")


# ============================================================================
# Page Routing - UNCHANGED
# ============================================================================

def route_page(page_selection: str):
    """Route to the appropriate page based on user selection."""
    try:
        if page_selection == "🏠 Home":
            display_home()
        elif page_selection == "🎥 Live Demo":
            with st.spinner("Loading Team Autono Minds demo interface..."):
                demo.show()
        elif page_selection == "🗺️ Hazard Map":
            with st.spinner("Loading hazard map with spatial clustering..."):
                map_view.show()
        elif page_selection == "⚠️ Alert Simulator":
            with st.spinner("Loading physics-based alert simulator..."):
                alert_sim.show()
        elif page_selection == "🔒 Privacy Test":
            with st.spinner("Loading GDPR compliance testing..."):
                privacy_test.show()
        elif page_selection == "📊 Metrics Dashboard":
            with st.spinner("Loading YOLOv11 performance metrics..."):
                metrics.show()
    except Exception as e:
        st.error(f"⚠️ Error loading page: {str(e)}")
        st.exception(e)


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application entry point with minimal styling."""
    try:
        # Configure page
        configure_page()
        
        # Apply minimal CSS (only hide Streamlit branding)
        apply_custom_css()
        
        # Create sidebar and get page selection
        page_selection = create_sidebar()
        
        # Route to selected page
        route_page(page_selection)
        
    except Exception as e:
        st.error(f"🚨 RoadGuard Application Error: {str(e)}")
        st.exception(e)
        
        st.info("""
            **Recovery Options:**
            1. Refresh the page (F5)
            2. Check if models/best.pt exists
            3. Verify all dependencies: `uv sync`
            4. Contact Team Autono Minds for support
        """)


if __name__ == "__main__":
    main()
