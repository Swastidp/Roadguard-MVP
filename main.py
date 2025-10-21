"""
Main entry point for RoadGuard application.
Launches Streamlit app with proper configuration.

Team: Autono Minds | VW Hackathon 2025
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch Streamlit application."""
    print("🚗 Starting RoadGuard - Team Autono Minds")
    print("🔥 YOLOv11 + SE Attention Model Ready")
    
    # Path to main Streamlit app
    app_path = Path(__file__).parent / "app" / "main.py"
    
    if not app_path.exists():
        print(f"❌ App file not found: {app_path}")
        return 1
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.address", "0.0.0.0",
            "--server.port", "8501"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Streamlit: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n🛑 RoadGuard stopped by user")
        return 0
    
    return 0


if __name__ == "__main__":
    exit(main())
