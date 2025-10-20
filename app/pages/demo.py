"""
Demo page for live hazard detection.

This module provides an interactive demo interface for testing the hazard detection
system with uploaded images/videos or sample data.
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import time

# Import components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.components import detection, privacy, utils
from app.config import (
    MODEL_PATH_PT,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    CLASS_NAMES,
    SEVERITY_COLORS,
    BASE_DIR
)


# ============================================================================
# Main Demo Page
# ============================================================================

def show():
    """Main demo page function."""
    # Page header
    st.title("🎥 Live Hazard Detection Demo")
    
    st.markdown("""
        Welcome to the **RoadGuard Hazard Detection Demo**! Upload an image or video 
        to test our AI-powered road hazard detection system in real-time.
        
        Our system can detect:
        - 🕳️ **Potholes** - Road surface depressions
        - 📏 **Longitudinal Cracks** - Cracks parallel to road direction
        - 📐 **Transverse Cracks** - Cracks perpendicular to road direction
        - 🕸️ **Alligator Cracks** - Interconnected crack patterns
    """)
    
    st.markdown("---")
    
    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Detection Settings")
        
        # Model settings
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(CONFIDENCE_THRESHOLD),
            step=0.05,
            help="Minimum confidence score for detections"
        )
        
        iou = st.slider(
            "IoU Threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(IOU_THRESHOLD),
            step=0.05,
            help="Intersection over Union threshold for NMS"
        )
        
        st.markdown("---")
        
        # Privacy settings
        st.subheader("🔒 Privacy Protection")
        enable_privacy = st.checkbox(
            "Enable Privacy Blurring",
            value=True,
            help="Automatically blur faces and license plates"
        )
        
        if enable_privacy:
            privacy_method = st.selectbox(
                "Anonymization Method",
                options=['gaussian', 'pixelate', 'black'],
                index=0,
                help="Method for blurring sensitive regions"
            )
        else:
            privacy_method = 'gaussian'
        
        st.markdown("---")
        
        # Display settings
        st.subheader("🎨 Display Options")
        show_boxes = st.checkbox(
            "Show Bounding Boxes",
            value=True,
            help="Draw detection boxes on output"
        )
        
        show_confidence = st.checkbox(
            "Show Confidence Scores",
            value=True,
            help="Display confidence values on boxes"
        )
        
        show_stats = st.checkbox(
            "Show Detection Statistics",
            value=True,
            help="Display detailed statistics"
        )
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📤 Upload", "🎬 Sample Videos", "📊 Batch Processing"])
    
    # Tab 1: Upload
    with tab1:
        process_uploaded_media(
            confidence, iou, enable_privacy, privacy_method,
            show_boxes, show_confidence, show_stats
        )
    
    # Tab 2: Sample Videos
    with tab2:
        process_sample_videos(
            confidence, iou, enable_privacy, privacy_method,
            show_boxes, show_confidence, show_stats
        )
    
    # Tab 3: Batch Processing
    with tab3:
        process_batch(
            confidence, iou, enable_privacy, privacy_method,
            show_boxes, show_confidence, show_stats
        )


# ============================================================================
# Upload Processing
# ============================================================================

def process_uploaded_media(
    confidence: float,
    iou: float,
    enable_privacy: bool,
    privacy_method: str,
    show_boxes: bool,
    show_confidence: bool,
    show_stats: bool
):
    """Process uploaded images or videos."""
    
    st.subheader("📤 Upload Image or Video")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'],
        help="Upload an image or video for hazard detection"
    )
    
    if uploaded_file is not None:
        # Determine file type
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        if file_extension in ['.jpg', '.jpeg', '.png']:
            process_image(
                uploaded_file, confidence, iou, enable_privacy,
                privacy_method, show_boxes, show_confidence, show_stats
            )
        elif file_extension in ['.mp4', '.avi', '.mov']:
            process_video(
                uploaded_file, confidence, iou, enable_privacy,
                privacy_method, show_boxes, show_confidence, show_stats
            )
        else:
            st.error("Unsupported file format")


def process_image(
    uploaded_file,
    confidence: float,
    iou: float,
    enable_privacy: bool,
    privacy_method: str,
    show_boxes: bool,
    show_confidence: bool,
    show_stats: bool
):
    """Process a single image."""
    
    st.info("📸 Processing image...")
    
    try:
        # Load image
        image = utils.load_image_from_upload(uploaded_file)
        
        if image is None:
            st.error("Failed to load image")
            return
        
        # Display original image
        st.subheader("Original Image")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(utils.bgr_to_rgb(image), caption="Original", use_container_width=True)
        
        # Load model
        with st.spinner("Loading detection model..."):
            model = detection.load_model(MODEL_PATH_PT)
            
            if model is None:
                st.error("Failed to load detection model")
                return
        
        # Run detection
        with st.spinner("Detecting hazards..."):
            start_time = time.time()
            detections = detection.detect_hazards(
                model, image, confidence, iou
            )
            detection_time = time.time() - start_time
        
        # Apply privacy blurring if enabled
        processed_image = image.copy()
        num_privacy_regions = 0
        
        if enable_privacy:
            with st.spinner("Applying privacy protection..."):
                privacy_model = privacy.load_privacy_model()
                if privacy_model is not None:
                    processed_image, num_privacy_regions = privacy.anonymize_pipeline(
                        processed_image,
                        method=privacy_method,
                        model=privacy_model
                    )
        
        # Draw detections if enabled
        if show_boxes:
            processed_image = detection.draw_detections(
                processed_image,
                detections,
                CLASS_NAMES,
                draw_confidence=show_confidence
            )
        
        # Display processed image
        with col2:
            st.image(
                utils.bgr_to_rgb(processed_image),
                caption="Processed",
                use_container_width=True
            )
        
        # Display statistics
        if show_stats:
            display_detection_stats(detections, detection_time, num_privacy_regions)
        
        # Detection details
        if detections['detection_count'] > 0:
            st.subheader("📋 Detection Details")
            details_df = create_detection_dataframe(detections)
            st.dataframe(details_df, use_container_width=True)
        
        # Download button
        st.subheader("💾 Download Results")
        output_bytes = utils.image_to_bytes(processed_image, format='JPEG')
        st.download_button(
            label="⬇️ Download Processed Image",
            data=output_bytes,
            file_name=f"processed_{uploaded_file.name}",
            mime="image/jpeg"
        )
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.exception(e)


def process_video(
    uploaded_file,
    confidence: float,
    iou: float,
    enable_privacy: bool,
    privacy_method: str,
    show_boxes: bool,
    show_confidence: bool,
    show_stats: bool
):
    """Process a video file."""
    
    st.subheader("🎬 Video Processing")
    
    # Frame processing options
    col1, col2 = st.columns(2)
    with col1:
        max_frames = st.number_input(
            "Max frames to process",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            help="Limit processing for faster results"
        )
    
    with col2:
        frame_skip = st.number_input(
            "Process every Nth frame",
            min_value=1,
            max_value=30,
            value=5,
            step=1,
            help="Skip frames to speed up processing"
        )
    
    if st.button("🚀 Start Processing", type="primary"):
        try:
            # Save uploaded video
            video_path = utils.save_uploaded_file(
                uploaded_file,
                BASE_DIR / "temp",
                prefix="upload"
            )
            
            if video_path is None:
                st.error("Failed to save video")
                return
            
            # Load model
            with st.spinner("Loading models..."):
                det_model = detection.load_model(MODEL_PATH_PT)
                priv_model = privacy.load_privacy_model() if enable_privacy else None
            
            # Process video
            results = process_video_frames(
                video_path,
                det_model,
                priv_model,
                confidence,
                iou,
                enable_privacy,
                privacy_method,
                show_boxes,
                show_confidence,
                max_frames,
                frame_skip
            )
            
            if results:
                display_video_results(results)
            
            # Cleanup
            video_path.unlink(missing_ok=True)
            
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            st.exception(e)


# ============================================================================
# Sample Videos
# ============================================================================

def process_sample_videos(
    confidence: float,
    iou: float,
    enable_privacy: bool,
    privacy_method: str,
    show_boxes: bool,
    show_confidence: bool,
    show_stats: bool
):
    """Process pre-loaded sample videos."""
    
    st.subheader("🎬 Sample Videos")
    
    # Check for sample videos
    sample_dir = BASE_DIR / "data" / "sample_videos"
    
    if not sample_dir.exists():
        st.warning("No sample videos directory found. Please upload your own media.")
        st.info(f"Expected location: {sample_dir}")
        return
    
    # List available samples
    sample_files = list(sample_dir.glob("*.mp4")) + list(sample_dir.glob("*.avi"))
    
    if not sample_files:
        st.warning("No sample videos found in the directory.")
        return
    
    # Select sample
    sample_names = [f.name for f in sample_files]
    selected_sample = st.selectbox(
        "Choose a sample video",
        options=sample_names,
        help="Select a pre-loaded sample video"
    )
    
    if selected_sample and st.button("🎥 Process Sample Video", type="primary"):
        sample_path = sample_dir / selected_sample
        
        # Load model
        with st.spinner("Loading models..."):
            det_model = detection.load_model(MODEL_PATH_PT)
            priv_model = privacy.load_privacy_model() if enable_privacy else None
        
        # Process video
        results = process_video_frames(
            sample_path,
            det_model,
            priv_model,
            confidence,
            iou,
            enable_privacy,
            privacy_method,
            show_boxes,
            show_confidence,
            max_frames=100,
            frame_skip=5
        )
        
        if results:
            display_video_results(results)


# ============================================================================
# Batch Processing
# ============================================================================

def process_batch(
    confidence: float,
    iou: float,
    enable_privacy: bool,
    privacy_method: str,
    show_boxes: bool,
    show_confidence: bool,
    show_stats: bool
):
    """Batch process multiple files."""
    
    st.subheader("📊 Batch Processing")
    
    uploaded_files = st.file_uploader(
        "Upload multiple images",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Upload multiple images for batch processing"
    )
    
    if uploaded_files and st.button("🚀 Process Batch", type="primary"):
        # Load model once
        with st.spinner("Loading model..."):
            model = detection.load_model(MODEL_PATH_PT)
            priv_model = privacy.load_privacy_model() if enable_privacy else None
        
        # Process each file
        batch_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            
            try:
                image = utils.load_image_from_upload(uploaded_file)
                
                if image is not None:
                    # Detect
                    detections = detection.detect_hazards(model, image, confidence, iou)
                    
                    # Privacy
                    num_privacy = 0
                    if enable_privacy and priv_model:
                        _, num_privacy = privacy.anonymize_pipeline(
                            image, method=privacy_method, model=priv_model
                        )
                    
                    batch_results.append({
                        'filename': uploaded_file.name,
                        'hazards_detected': detections['detection_count'],
                        'avg_confidence': np.mean(detections['confidences']) if detections['confidences'] else 0,
                        'privacy_regions': num_privacy
                    })
            
            except Exception as e:
                st.warning(f"Error processing {uploaded_file.name}: {str(e)}")
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        status_text.text("✅ Batch processing complete!")
        
        # Display results
        if batch_results:
            st.subheader("📊 Batch Results Summary")
            results_df = pd.DataFrame(batch_results)
            st.dataframe(results_df, use_container_width=True)
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Images", len(batch_results))
            with col2:
                total_hazards = results_df['hazards_detected'].sum()
                st.metric("Total Hazards", total_hazards)
            with col3:
                avg_conf = results_df['avg_confidence'].mean()
                st.metric("Avg Confidence", f"{avg_conf:.1%}")


# ============================================================================
# Video Frame Processing
# ============================================================================

def process_video_frames(
    video_path: Path,
    det_model,
    priv_model,
    confidence: float,
    iou: float,
    enable_privacy: bool,
    privacy_method: str,
    show_boxes: bool,
    show_confidence: bool,
    max_frames: int = 100,
    frame_skip: int = 5
) -> Optional[Dict[str, Any]]:
    """Process video frames and return results."""
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            st.error("Failed to open video")
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        st.info(f"Video: {fps:.1f} FPS, {total_frames} total frames")
        
        # Process frames
        frame_results = []
        frame_count = 0
        processed_count = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        start_time = time.time()
        
        while cap.isOpened() and processed_count < max_frames:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames
            if frame_count % frame_skip != 0:
                continue
            
            processed_count += 1
            status_text.text(f"Processing frame {processed_count}/{max_frames}...")
            
            # Detect hazards
            detections = detection.detect_hazards(det_model, frame, confidence, iou)
            
            frame_results.append({
                'frame': frame_count,
                'detections': detections['detection_count'],
                'avg_confidence': np.mean(detections['confidences']) if detections['confidences'] else 0
            })
            
            progress_bar.progress(processed_count / max_frames)
        
        cap.release()
        
        processing_time = time.time() - start_time
        processing_fps = processed_count / processing_time
        
        status_text.text("✅ Video processing complete!")
        
        return {
            'frame_results': frame_results,
            'total_frames': frame_count,
            'processed_frames': processed_count,
            'processing_time': processing_time,
            'processing_fps': processing_fps,
            'original_fps': fps
        }
        
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return None


# ============================================================================
# Display Functions
# ============================================================================

def display_detection_stats(detections: Dict, detection_time: float, num_privacy: int):
    """Display detection statistics."""
    
    st.subheader("📊 Detection Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Hazards Detected",
            detections['detection_count'],
            delta=None
        )
    
    with col2:
        avg_conf = np.mean(detections['confidences']) if detections['confidences'] else 0
        st.metric(
            "Avg Confidence",
            f"{avg_conf:.1%}",
            delta=None
        )
    
    with col3:
        st.metric(
            "Processing Time",
            f"{detection_time*1000:.0f}ms",
            delta=None
        )
    
    with col4:
        st.metric(
            "Privacy Regions",
            num_privacy,
            delta=None
        )
    
    # Detection summary
    if detections['detection_count'] > 0:
        summary = detection.get_detection_summary(detections, CLASS_NAMES)
        
        st.markdown("**Detection Breakdown:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**By Type:**")
            for class_name, count in summary['class_counts'].items():
                st.write(f"- {class_name.replace('_', ' ').title()}: {count}")
        
        with col2:
            st.markdown("**By Severity:**")
            for severity, count in summary['severity_counts'].items():
                if count > 0:
                    color = SEVERITY_COLORS.get(severity, '#3B82F6')
                    st.markdown(
                        f"<span style='color: {color}'>● {severity.title()}: {count}</span>",
                        unsafe_allow_html=True
                    )


def create_detection_dataframe(detections: Dict) -> pd.DataFrame:
    """Create a DataFrame from detection results."""
    
    data = []
    
    for i, (box, cls, conf) in enumerate(zip(
        detections['boxes'],
        detections['classes'],
        detections['confidences']
    )):
        x1, y1, x2, y2 = box
        bbox_area = (x2 - x1) * (y2 - y1)
        
        data.append({
            'ID': i + 1,
            'Type': CLASS_NAMES[cls].replace('_', ' ').title(),
            'Confidence': f"{conf:.1%}",
            'Severity': detection.classify_severity(cls, bbox_area).title(),
            'Bbox': f"({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})",
            'Area': int(bbox_area)
        })
    
    return pd.DataFrame(data)


def display_video_results(results: Dict):
    """Display video processing results."""
    
    st.subheader("📊 Video Processing Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Processed Frames", results['processed_frames'])
    
    with col2:
        st.metric("Total Frames", results['total_frames'])
    
    with col3:
        st.metric("Processing FPS", f"{results['processing_fps']:.1f}")
    
    with col4:
        st.metric("Processing Time", f"{results['processing_time']:.1f}s")
    
    # Frame detection chart
    if results['frame_results']:
        st.subheader("📈 Detections per Frame")
        
        df = pd.DataFrame(results['frame_results'])
        st.line_chart(df.set_index('frame')['detections'])


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    show()
