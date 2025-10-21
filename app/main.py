"""
RoadGuard: Road Hazard Detection System by Team Autono Minds
Main entry point for the Streamlit application.

Team: Autono Minds | VW Hackathon 2025
"""

import streamlit as st
import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.config import *

# Import page modules (removed alert_sim and metrics)
try:
    from app.pages import demo, map_view, privacy_test
    from app.components import detection, privacy, utils
except ImportError as e:
    st.error(f"Error importing page modules: {e}")
    st.stop()


# ============================================================================
# Page Configuration - HIDE STREAMLIT PAGES
# ============================================================================

def configure_page():
    """Configure Streamlit page settings and hide default pages."""
    st.set_page_config(
        page_title="RoadGuard - Team Autono Minds",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/Swastidp/Roadguard-MVP',
            'Report a bug': 'https://github.com/Swastidp/Roadguard-MVP/issues',
            'About': """
            # RoadGuard
            
            Team Autono Minds - VW Hackathon 2025
            YOLOv11n Custom for Road Hazard Detection
            
            **Performance**: 50.3% mAP@0.5
            """
        }
    )


# ============================================================================
# Minimal CSS - HIDE STREAMLIT BRANDING AND DEFAULT PAGES
# ============================================================================

def apply_custom_css():
    """Apply CSS to hide Streamlit branding and default page navigation."""
    st.markdown("""
        <style>
        /* Hide Streamlit branding and default navigation */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Hide the default Streamlit page selector */
        .css-1d391kg {display: none;}
        .css-1rs6os {display: none;}
        .css-17ziqus {display: none;}
        .stSelectbox {display: none;}
        
        /* Hide any auto-generated navigation */
        [data-testid="stSidebar"] .css-1d391kg {display: none;}
        [data-testid="stSidebar"] .css-1rs6os {display: none;}
        [data-testid="stSidebar"] .css-17ziqus {display: none;}
        
        /* Custom sidebar styling */
        .sidebar .sidebar-content {
            padding-top: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)


# ============================================================================
# Custom Sidebar - COMPLETE CONTROL
# ============================================================================

def create_sidebar():
    """Create custom sidebar with exact page order and naming."""
    with st.sidebar:
        # Simple team branding with Streamlit defaults
        st.title("RoadGuard")
        st.subheader("AI-Powered Hazard Detection")
        
        # Team info using Streamlit default styling
        st.info("**Team Autono Minds**")
        st.caption("VW Hackathon 2025 | YOLOv11n Custom")
        
        st.markdown("---")
        
        # Custom navigation with exact order you want
        st.markdown("### Navigation")
        page = st.radio(
            "Select a page:",
            options=[
                "Demo",
                "Map view", 
                "Privacy test",
                "Overview"
            ],
            format_func=lambda x: x.replace("_", " ").title(),
            key="main_navigation"
        )
        
        st.markdown("---")
        
        # Model performance with default metrics
        st.markdown("### Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("mAP@0.5", "50.3%")
        with col2:
            st.metric("Epochs", "65")
        
        st.markdown("---")
        
        # Simple footer with default styling
        st.markdown("### Info")
        st.success("**VW Hackathon 2025**")
        st.info("**Team Autono Minds**")
        st.caption("YOLOv11n Custom Trained Model")
        
        # GitHub link
        st.markdown("---")
        st.markdown("[Star us on GitHub](https://github.com/Swastidp/Roadguard-MVP)")
        st.caption("Built with Streamlit")
        
        return page


# ============================================================================
# Home Page with Integrated Demo and Metrics
# ============================================================================

def display_home():
    """Display home page with integrated demo and metrics functionality."""
    
    # Simple headers
    st.title("Team Autono Minds - RoadGuard")
    st.subheader("AI-Powered Road Hazard Detection System")
    
    # Team info
    st.info("**Team Autono Minds** | VW Hackathon 2025 | YOLOv11n Custom Training")
    
    # Project overview
    st.markdown("### Project Overview")
    
    st.markdown("""
    Custom **YOLOv11n** model achieving **50.3% mAP@0.5** on road hazard detection 
    with **65 epochs** of training on **6,439 images** using RTX 3050 GPU.
    """)
    
    st.success("Real-time detection • Privacy-compliant • Spatial intelligence")
    
    # Training results from notebook
    st.markdown("## Training Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Overall mAP@0.5", "50.3%", "Final epoch")
    
    with col2:
        st.metric("mAP@0.5:0.95", "23.0%", "COCO standard")
    
    with col3:
        st.metric("Training Images", "6,439", "65 epochs")
    
    with col4:
        st.metric("Training Time", "~2 hours", "RTX 3050")
    
    st.markdown("---")
    
    # Per-class performance from notebook
    st.markdown("## Per-Class Performance Analysis")
    
    performance_data = {
        'Class': ['Longitudinal Crack', 'Transverse Crack', 'Alligator Crack', 'Pothole', 'Other Corruption'],
        'mAP@0.5': [59.8, 71.7, 10.1, 60.1, 0.0],
        'Precision': [61.0, 78.1, 12.0, 69.9, 0.0],
        'Recall': [54.5, 62.8, 8.3, 51.5, 0.0],
        'Difficulty': ['Medium', 'Easy', 'Very Hard', 'Medium', 'Unknown']
    }
    
    df = pd.DataFrame(performance_data)
    st.dataframe(df, use_container_width=True)
    
    # Performance visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### mAP@0.5 by Class")
        fig_map = px.bar(
            x=performance_data['Class'][:4],  # Exclude 'Other Corruption' 
            y=performance_data['mAP@0.5'][:4],
            title="mAP@0.5 Performance by Class",
            labels={'x': 'Class', 'y': 'mAP@0.5 (%)'}
        )
        fig_map.update_traces(marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        st.plotly_chart(fig_map, use_container_width=True)
    
    with col2:
        st.markdown("### Precision vs Recall")
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=performance_data['Recall'][:4],
            y=performance_data['Precision'][:4],
            mode='markers+text',
            text=performance_data['Class'][:4],
            textposition="top center",
            marker=dict(size=12, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ))
        fig_pr.update_layout(
            title="Precision vs Recall by Class",
            xaxis_title="Recall (%)",
            yaxis_title="Precision (%)"
        )
        st.plotly_chart(fig_pr, use_container_width=True)
    
    # Analysis insights
    st.markdown("### Key Insights from Training")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.success("**Transverse Cracks** perform best (71.7% mAP) - clear perpendicular patterns")
        st.info("**Longitudinal Cracks** show good detection (59.8% mAP) - parallel patterns")
        st.info("**Potholes** achieve solid accuracy (60.1% mAP) - distinct circular shapes")
        st.warning("**Alligator Cracks** need improvement (10.1% mAP) - complex interconnected patterns")
    
    with col2:
        st.metric("Best Class", "Transverse Crack", "71.7%")
        st.metric("Most Challenging", "Alligator Crack", "10.1%")
        st.metric("Dataset Split", "80/20", "Train/Val")
        st.metric("Final Epoch", "65", "Best model")
    
    st.markdown("---")
    
    # System Performance Metrics (merged from metrics page)
    st.markdown("## System Performance Benchmarks")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Inference Performance")
        perf_data = {
            'Resolution': ['640×640', '1280×720', '1920×1080'],
            'Inference Time (ms)': [45, 78, 142],
            'FPS': [22, 13, 7],
            'Memory (GB)': [1.2, 1.8, 2.4]
        }
        perf_df = pd.DataFrame(perf_data)
        st.dataframe(perf_df, use_container_width=True)
    
    with col2:
        st.markdown("### Training Configuration")
        config_data = {
            'Parameter': ['Batch Size', 'Learning Rate', 'Optimizer', 'Device', 'VRAM Used'],
            'Value': ['20', '0.002→0.001', 'AdamW', 'RTX 3050', '~2.9GB']
        }
        config_df = pd.DataFrame(config_data)
        st.dataframe(config_df, use_container_width=True)
    
    with col3:
        st.markdown("### Model Statistics")
        model_data = {
            'Metric': ['Parameters', 'Model Size', 'Architecture', 'Classes', 'Input Size'],
            'Value': ['2.6M', '5.45 MB', 'YOLOv11n', '5', '640×640']
        }
        model_df = pd.DataFrame(model_data)
        st.dataframe(model_df, use_container_width=True)
    
    st.markdown("---")
    
    # Integrated Demo Section
    st.markdown("## Live Detection Demo")
    
    st.markdown("""
    Test our YOLOv11n model directly below. Upload an image to see real-time 
    road hazard detection with confidence scores and bounding boxes.
    """)
    
    # Demo interface
    uploaded_file = st.file_uploader(
        "Choose an image for detection",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a road image to test hazard detection"
    )
    
    if uploaded_file is not None:
        # Settings in columns
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("Detection Settings")
            confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
            enable_privacy = st.checkbox("Enable Privacy Protection", True)
            show_stats = st.checkbox("Show Detection Statistics", True)
            show_charts = st.checkbox("Show Performance Charts", True)
            
        with col1:
            # Process image
            try:
                image = utils.load_image_from_upload(uploaded_file)
                
                if image is not None:
                    # Display original and processed side by side
                    img_col1, img_col2 = st.columns(2)
                    
                    with img_col1:
                        st.subheader("Original Image")
                        st.image(utils.bgr_to_rgb(image), use_column_width=True)
                    
                    # Load model and detect
                    with st.spinner("Detecting hazards..."):
                        model = detection.load_model(MODEL_PATH_PT)
                        if model is not None:
                            start_time = time.time()
                            detections = detection.detect_hazards(model, image, confidence, IOU_THRESHOLD)
                            detection_time = time.time() - start_time
                            
                            # Apply privacy if enabled
                            processed_image = image.copy()
                            if enable_privacy:
                                privacy_model = privacy.load_privacy_model()
                                if privacy_model is not None:
                                    processed_image, _ = privacy.anonymize_pipeline(
                                        processed_image, method='gaussian', model=privacy_model
                                    )
                            
                            # Draw results
                            processed_image = detection.draw_detections(
                                processed_image, detections, CLASS_NAMES, draw_confidence=True
                            )
                            
                            with img_col2:
                                st.subheader("Detection Results")
                                st.image(utils.bgr_to_rgb(processed_image), use_column_width=True)
                            
                            # Show stats
                            if show_stats:
                                st.markdown("### Detection Statistics")
                                
                                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                                
                                with stat_col1:
                                    st.metric("Hazards Detected", detections['detection_count'])
                                
                                with stat_col2:
                                    if detections['confidences']:
                                        avg_conf = np.mean(detections['confidences'])
                                        st.metric("Avg Confidence", f"{avg_conf:.1%}")
                                    else:
                                        st.metric("Avg Confidence", "N/A")
                                
                                with stat_col3:
                                    st.metric("Processing Time", f"{detection_time*1000:.0f}ms")
                                
                                with stat_col4:
                                    fps = 1 / detection_time if detection_time > 0 else 0
                                    st.metric("Inference FPS", f"{fps:.1f}")
                                
                                if detections['detection_count'] > 0:
                                    # Create details table
                                    details_data = []
                                    for i, (box, cls, conf) in enumerate(zip(
                                        detections['boxes'], detections['classes'], detections['confidences']
                                    )):
                                        details_data.append({
                                            'ID': i + 1,
                                            'Type': CLASS_NAMES[cls].replace('_', ' ').title(),
                                            'Confidence': f"{conf:.1%}",
                                            'Bbox': f"({int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])})"
                                        })
                                    
                                    if details_data:
                                        st.subheader("Detection Details")
                                        details_df = pd.DataFrame(details_data)
                                        st.dataframe(details_df, use_container_width=True)
                                    
                                    # Show performance charts if enabled
                                    if show_charts and len(detections['confidences']) > 1:
                                        chart_col1, chart_col2 = st.columns(2)
                                        
                                        with chart_col1:
                                            # Confidence distribution
                                            fig_conf = px.histogram(
                                                x=detections['confidences'],
                                                nbins=10,
                                                title="Confidence Distribution"
                                            )
                                            fig_conf.update_layout(
                                                xaxis_title="Confidence",
                                                yaxis_title="Count"
                                            )
                                            st.plotly_chart(fig_conf, use_container_width=True)
                                        
                                        with chart_col2:
                                            # Class distribution
                                            class_names = [CLASS_NAMES[cls] for cls in detections['classes']]
                                            fig_classes = px.pie(
                                                names=class_names,
                                                title="Hazard Type Distribution"
                                            )
                                            st.plotly_chart(fig_classes, use_container_width=True)
                                
                            elif detections['detection_count'] == 0:
                                st.info("No hazards detected in this image")
                        else:
                            st.error("Failed to load detection model")
                            
            except Exception as e:
                st.error(f"Detection error: {str(e)}")
    
    st.markdown("---")
    
    # Features overview
    st.markdown("## Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Real-time Detection")
        st.markdown("""
        Custom YOLOv11n model detects potholes, cracks, 
        and road damage instantly with **50.3% mAP@0.5** accuracy.
        """)
        st.success("< 45ms inference time")
    
    with col2:
        st.markdown("### Interactive Mapping")
        st.markdown("""
        Visualize detected hazards on interactive maps with DBSCAN clustering 
        and spatial deduplication for comprehensive road monitoring.
        """)
        st.info("GPS-based alert zones")
    
    with col3:
        st.markdown("### Privacy Protected")
        st.markdown("""
        Automatic face and license plate detection with real-time blurring 
        ensures GDPR compliance and user privacy protection.
        """)
        st.warning("100% GDPR compliant")
    
    st.markdown("---")
    
    # Getting started
    st.markdown("## Navigation Guide")
    
    st.markdown("""
    1. **Demo** - Test detection with uploaded images or webcam feed
    2. **Map View** - View detected hazards on an interactive map with clustering  
    3. **Privacy Test** - Test GDPR compliance with face and license plate blurring
    4. **Overview** - Complete project overview, metrics, and integrated demo
    """)
    
    st.success("Use the sidebar navigation to explore different features of RoadGuard!")


# ============================================================================
# Page Routing - EXACT MATCHING
# ============================================================================

def route_page(page_selection: str):
    """Route to the appropriate page based on user selection."""
    try:
        if page_selection == "Demo":
            with st.spinner("Loading Team Autono Minds demo interface..."):
                demo.show()
        elif page_selection == "Map view":
            with st.spinner("Loading hazard map with spatial clustering..."):
                map_view.show()
        elif page_selection == "Privacy test":
            with st.spinner("Loading GDPR compliance testing..."):
                privacy_test.show()
        elif page_selection == "Overview":
            display_home()
        else:
            # Fallback to overview if unknown page
            demo.show()
    except Exception as e:
        st.error(f"Error loading page: {str(e)}")
        st.exception(e)


# ============================================================================
# Main Application - HIDE DEFAULT STREAMLIT NAVIGATION
# ============================================================================

def main():
    """Main application entry point with custom navigation only."""
    try:
        # Configure page
        configure_page()
        
        # Apply CSS to hide default Streamlit navigation
        apply_custom_css()
        
        # Create custom sidebar and get page selection
        page_selection = create_sidebar()
        
        # Route to selected page
        route_page(page_selection)
        
    except Exception as e:
        st.error(f"RoadGuard Application Error: {str(e)}")
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
