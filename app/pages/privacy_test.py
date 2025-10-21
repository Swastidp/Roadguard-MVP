"""
Privacy testing page for GDPR compliance verification.
Cloud deployment compatible.

Team: Autono Minds | VW Hackathon 2025
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import time

# Configure OpenCV for cloud deployment
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

try:
    import cv2
    cv2.setUseOptimized(True)
except ImportError as e:
    st.error(f"OpenCV import failed: {e}")
    st.stop()

# Import components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.components import privacy, utils
from app.config import BASE_DIR

def show():
    """Main privacy test page function."""
    # Page header
    st.title("🔒 Privacy Protection Testing")
    
    st.markdown("""
        Test our **GDPR-compliant privacy protection system** that automatically detects 
        and anonymizes faces and license plates in images and videos.
        
        **Privacy Methods Available:**
        - 🌫️ **Gaussian Blur** - Smooth blurring effect
        - 🔲 **Pixelation** - Block-based anonymization  
        - ⬛ **Black Boxes** - Complete coverage
    """)
    
    st.markdown("---")
    
    # Continue with rest of the existing code...
    # [Keep the existing show() function content but ensure all st.dataframe calls use width='stretch']
    
    # Just showing the pattern for the key changes:
    # Replace any instances of:
    # st.dataframe(df, use_container_width=True)
    # with:
    # st.dataframe(df, width='stretch')
    
    st.info("Privacy testing page loaded successfully!")


if __name__ == "__main__":
    show()
