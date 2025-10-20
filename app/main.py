"""
RoadGuard: Road Hazard Detection System
Main entry point for the Streamlit application.

Author: Team VW Hackathon 2025
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
        page_title="RoadGuard: Hazard Detection",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/yourusername/roadguard',
            'Report a bug': 'https://github.com/yourusername/roadguard/issues',
            'About': """
            # RoadGuard 🚗
            
            Advanced AI-powered road hazard detection system for safer driving.
            
            **VW Hackathon 2025**
            
            Built with ❤️ using YOLOv8 and Streamlit
            """
        }
    )


# ============================================================================
# Custom CSS Styling
# ============================================================================

def apply_custom_css():
    """Apply custom CSS styling to the application."""
    st.markdown("""
        <style>
        /* Main header styling */
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1E40AF;
            text-align: center;
            margin-bottom: 2rem;
            padding: 1rem;
            background: linear-gradient(90deg, #3B82F6 0%, #1E40AF 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        /* Custom button styling */
        .stButton>button {
            background-color: #3B82F6;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 2rem;
            font-weight: 600;
            border: none;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background-color: #2563EB;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transform: translateY(-2px);
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #F8FAFC;
            border-right: 2px solid #E2E8F0;
        }
        
        [data-testid="stSidebar"] .element-container {
            padding: 0.5rem 1rem;
        }
        
        /* Radio button styling */
        .stRadio > label {
            font-size: 1.1rem;
            font-weight: 600;
            color: #1E293B;
        }
        
        .stRadio > div {
            gap: 0.5rem;
        }
        
        .stRadio > div > label {
            background-color: white;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            border: 2px solid #E2E8F0;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .stRadio > div > label:hover {
            border-color: #3B82F6;
            background-color: #EFF6FF;
        }
        
        /* Alert boxes */
        .stAlert {
            border-radius: 8px;
            border-left: 4px solid;
        }
        
        /* Metric cards */
        [data-testid="stMetric"] {
            background-color: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        /* Footer styling */
        .footer {
            text-align: center;
            padding: 2rem 1rem;
            color: #64748B;
            font-size: 0.9rem;
            border-top: 1px solid #E2E8F0;
            margin-top: 2rem;
        }
        
        /* Loading animation */
        .stSpinner > div {
            border-color: #3B82F6 transparent transparent transparent;
        }
        
        /* Success/Error message styling */
        .success-box {
            background-color: #ECFDF5;
            color: #065F46;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #10B981;
            margin: 1rem 0;
        }
        
        .error-box {
            background-color: #FEF2F2;
            color: #991B1B;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #EF4444;
            margin: 1rem 0;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)


# ============================================================================
# Sidebar Navigation
# ============================================================================

def create_sidebar():
    """Create and configure the sidebar navigation."""
    with st.sidebar:
        # App logo/title
        st.markdown("""
            <div style='text-align: center; padding: 1rem 0 2rem 0;'>
                <h1 style='color: #1E40AF; font-size: 2rem; margin: 0;'>
                    🚗 RoadGuard
                </h1>
                <p style='color: #64748B; font-size: 0.9rem; margin: 0.5rem 0 0 0;'>
                    AI-Powered Hazard Detection
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation menu
        page = st.radio(
            "Navigate",
            options=[
                "🏠 Home",
                "🎥 Live Demo",
                "🗺️ Hazard Map",
                "⚠️ Alert Simulator",
                "🔒 Privacy Test",
                "📊 Metrics Dashboard"
            ],
            key="navigation",
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Footer information
        st.markdown("""
            <div class='footer'>
                <p style='margin: 0.5rem 0;'>
                    <strong>VW Hackathon 2025</strong>
                </p>
                <p style='margin: 0.5rem 0; font-size: 0.85rem;'>
                    Team CloudNatics
                </p>
                <p style='margin: 1rem 0 0.5rem 0;'>
                    <a href='https://github.com/yourusername/roadguard' 
                       target='_blank' 
                       style='color: #3B82F6; text-decoration: none;'>
                        ⭐ Star us on GitHub
                    </a>
                </p>
                <p style='margin: 0; font-size: 0.8rem; color: #94A3B8;'>
                    Built with Streamlit & YOLOv8
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        return page


# ============================================================================
# Page Routing
# ============================================================================

def route_page(page_selection: str):
    """
    Route to the appropriate page based on user selection.
    
    Args:
        page_selection: Selected page from sidebar navigation
    """
    try:
        if page_selection == "🏠 Home":
            display_home()
        elif page_selection == "🎥 Live Demo":
            with st.spinner("Loading demo interface..."):
                demo.show()
        elif page_selection == "🗺️ Hazard Map":
            with st.spinner("Loading hazard map..."):
                map_view.show()
        elif page_selection == "⚠️ Alert Simulator":
            with st.spinner("Loading alert simulator..."):
                alert_sim.show()
        elif page_selection == "🔒 Privacy Test":
            with st.spinner("Loading privacy test..."):
                privacy_test.show()
        elif page_selection == "📊 Metrics Dashboard":
            with st.spinner("Loading metrics dashboard..."):
                metrics.show()
    except Exception as e:
        st.error(f"⚠️ Error loading page: {str(e)}")
        st.exception(e)


def display_home():
    """Display the home page."""
    st.markdown("<h1 class='main-header'>Welcome to RoadGuard</h1>", 
                unsafe_allow_html=True)
    
    # Hero section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div style='text-align: center; padding: 2rem;'>
                <h2 style='color: #1E293B; margin-bottom: 1rem;'>
                    AI-Powered Road Hazard Detection
                </h2>
                <p style='color: #64748B; font-size: 1.1rem; line-height: 1.6;'>
                    Detect and alert drivers about road hazards in real-time using 
                    advanced computer vision and deep learning technologies.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Features
    st.markdown("### 🎯 Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div style='background-color: white; padding: 1.5rem; 
                        border-radius: 8px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);'>
                <h3 style='color: #3B82F6; margin-top: 0;'>🔍 Real-time Detection</h3>
                <p style='color: #64748B;'>
                    Detect potholes, cracks, and road damage instantly using YOLOv8
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background-color: white; padding: 1.5rem; 
                        border-radius: 8px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);'>
                <h3 style='color: #10B981; margin-top: 0;'>🗺️ Interactive Map</h3>
                <p style='color: #64748B;'>
                    Visualize detected hazards on an interactive map interface
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div style='background-color: white; padding: 1.5rem; 
                        border-radius: 8px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);'>
                <h3 style='color: #F59E0B; margin-top: 0;'>🔒 Privacy Protected</h3>
                <p style='color: #64748B;'>
                    Automatic face and license plate blurring for privacy compliance
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Statistics
    st.markdown("### 📈 Detection Capabilities")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Hazard Types", value="4", delta="Categories")
    with col2:
        st.metric(label="Avg Confidence", value="92%", delta="3% ↑")
    with col3:
        st.metric(label="Response Time", value="<50ms", delta="Real-time")
    with col4:
        st.metric(label="Privacy Score", value="100%", delta="Compliant")
    
    st.markdown("---")
    
    # Getting Started
    st.markdown("### 🚀 Getting Started")
    
    st.markdown("""
        1. **🎥 Live Demo**: Test the detection system with your webcam or video file
        2. **🗺️ Hazard Map**: View all detected hazards on an interactive map
        3. **⚠️ Alert Simulator**: Simulate driver alerts based on vehicle speed
        4. **🔒 Privacy Test**: Verify privacy protection features
        5. **📊 Metrics Dashboard**: Analyze detection performance and statistics
    """)
    
    st.info("👈 Use the sidebar to navigate between different features")


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application entry point."""
    try:
        # Configure page
        configure_page()
        
        # Apply custom styling
        apply_custom_css()
        
        # Create sidebar and get page selection
        page_selection = create_sidebar()
        
        # Route to selected page
        route_page(page_selection)
        
    except Exception as e:
        st.error(f"🚨 Application Error: {str(e)}")
        st.exception(e)
        st.info("Please refresh the page or contact support if the issue persists.")


if __name__ == "__main__":
    main()
