# RoadGuard: AI-Powered Road Hazard Detection System

**Real-time AI-powered detection and alerting system for road hazards**

Built for VW Hackathon 2025 | Team Autono Minds

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Performance Metrics](#performance-metrics)
- [Technology Stack](#technology-stack)
- [Hackathon Context](#hackathon-context)
- [Future Roadmap](#future-roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Team & Acknowledgments](#team--acknowledgments)

## Overview

**RoadGuard** is an intelligent road hazard detection system that uses YOLOv11n to identify and alert drivers about road damage in real-time. The system detects potholes, cracks, and other road hazards, providing timely warnings while ensuring GDPR compliance through automatic face and license plate anonymization.

### Demo Video

*Coming Soon*

### Hackathon Achievement

- **Event**: VW Hackathon 2025
- **Track**: Smart Mobility & Road Safety
- **Team**: Autono Minds

## Features

### Core Detection

- **Real-time Hazard Detection**: Detects 5 types of road damage using YOLOv11n
  - Potholes - Road surface depressions
  - Longitudinal Cracks - Cracks parallel to road direction
  - Transverse Cracks - Cracks perpendicular to road direction
  - Alligator Cracks - Interconnected crack patterns
  - Other Corruption - General road surface damage
- **High Accuracy**: 50.3% mAP@0.5, 23.0% mAP@0.5:0.95
- **Fast Inference**: ~45ms per image (640×640) on RTX 3050

### Intelligent Features

- **Physics-Based Warnings**: Calculates safe warning distances based on vehicle parameters
- **Dynamic Speed Recommendations**: Real-time speed calculations for hazard approach
- **Multi-Hazard Support**: Handles multiple hazards in view
- **Directional Awareness**: Context-aware hazard alerting

### Geospatial Features

- **Interactive Hazard Map**: Cluster and heatmap visualizations
- **Spatial Deduplication**: DBSCAN clustering (10m radius)
- **Proximity Search**: Find hazards within configurable radius
- **Time-based Filtering**: Filter by detection date range
- **Status Tracking**: Active, resolved, pending states

### Privacy Protection (GDPR Compliant)

- **Automatic Anonymization**: Detects and blurs faces/license plates
- **Multiple Methods**: Gaussian blur, pixelation, black boxes
- **Verification System**: Confirms no identifiable data remains
- **Batch Processing**: Efficient handling of large datasets

### Analytics Dashboard

- **Model Performance Metrics**: Precision, recall, F1-score by class
- **Detection Statistics**: Time series, distribution charts
- **System Benchmarks**: Latency, FPS, memory usage
- **Export Reports**: CSV, JSON, executive summaries

## System Architecture

┌─────────────────────────────────────────────────────────────────┐
│ RoadGuard System │
├─────────────────────────────────────────────────────────────────┤
│ │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
│ │ Camera/ │────▶│ YOLOv11n │────▶│ Detection │ │
│ │ Video │ │ Model │ │ Pipeline │ │
│ │ Input │ │ (best.pt) │ └──────┬───────┘ │
│ └──────────────┘ └──────────────┘ │ │
│ │ │
│ ┌─────────────────────────────────────────────────┘ │
│ │ │
│ ┌──────────────▼──────────────┐ │
│ │ Privacy Protection Module │ │
│ ├─────────────────────────────┤ │
│ │ - Face Detection │ │
│ │ - Plate Detection │ │
│ │ - Gaussian Blur/Pixelate │ │
│ └──────────────┬──────────────┘ │
│ │ │
│ ┌──────────────▼──────────────┐ │
│ │ SQLite Database │ │
│ ├─────────────────────────────┤ │
│ │ - Hazard Records │ │
│ │ - Detection Metadata │ │
│ │ - Status Tracking │ │
│ └─────────────────┬───────────┘ │
│ │ │
│ ┌─────────────────▼─────────────────────┐ │
│ │ Streamlit Web Interface │ │
│ ├───────────────────────────────────────┤ │
│ │ Home | Demo | Map | Privacy | Metrics │ │
│ └───────────────────────────────────────┘ │
│ │
└─────────────────────────────────────────────────────────────────────────┘

text

### Data Flow

1. **Input**: Video/image from camera or upload
2. **Detection**: YOLOv11n inference (~45ms @ 640×640)
3. **Privacy**: Automatic face/plate anonymization
4. **Storage**: SQLite database with spatial indexes
5. **Visualization**: Interactive Streamlit dashboard

## Installation

### Prerequisites

- **Python**: 3.10 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 2GB free space
- **Optional**: GPU with CUDA support for faster inference

### Step 1: Clone Repository

git clone https://github.com/Swastidp/Roadguard-MVP.git
cd Roadguard-MVP

text

### Step 2: Install UV Package Manager

On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

On Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

Verify installation
uv --version

text

### Step 3: Setup Environment

Sync all dependencies (creates .venv automatically)
uv sync --all-extras

Activate virtual environment
On Unix/macOS
source .venv/bin/activate

On Windows
.venv\Scripts\activate

text

### Step 4: Download Models

Create models directory
mkdir -p models

Place your trained best.pt model in models/ directory
The model should be the YOLOv11n model trained on road hazards
text

### Step 5: Initialize Database

Create database with sample data
uv run python scripts/init_database.py --with-samples

Or without sample data
uv run python scripts/init_database.py

text

### Verify Installation

Run tests
uv run pytest tests/ -v

Check imports
uv run python -c "from app import main; print('✅ Installation successful!')"

text

## Usage

### Running the Application

#### Development Mode

Start Streamlit app
uv run streamlit run app/main.py

App will open at http://localhost:8501
text

#### Production Mode

Run with production settings
uv run streamlit run app/main.py --server.port 8080 --server.headless true

text

### Using the Application

#### 1. Home Page (with Integrated Demo)

- Welcome screen with live detection demo
- Upload images directly for testing
- View model performance metrics
- Real-time hazard detection results

#### 2. Demo Page

- **Upload Media**: Test with images or videos
- **Adjust Settings**: Confidence threshold, privacy options
- **View Results**: Annotated images with detection boxes
- **Download**: Save processed images/videos

#### 3. Map View

- **Interactive Map**: View hazards with clustering
- **Filter Hazards**: By type, severity, date range
- **Search Location**: Find hazards near GPS coordinates
- **Export Data**: Download filtered hazards as CSV/JSON

#### 4. Privacy Test

- **Upload Images**: Test anonymization on photos
- **Choose Method**: Gaussian blur, pixelate, or black boxes
- **Verify Compliance**: Automated privacy checking
- **Batch Process**: Anonymize multiple images

#### 5. Metrics Dashboard

- **Model Performance**: Precision, recall, mAP scores
- **Detection Stats**: Time series, distribution charts
- **System Benchmarks**: Inference speed, FPS

### Command Line Tools

Initialize or reset database
uv run python scripts/init_database.py --force

Run specific page tests
uv run pytest tests/test_detection.py -v

Check code quality
uv run ruff check .
uv run black --check .

Generate coverage report
uv run pytest --cov=app --cov-report=html

text

## Project Structure

roadguard-mvp/
├── .github/
│ └── workflows/
│ └── ci.yml # CI/CD pipeline
├── app/
│ ├── main.py # Streamlit entry point
│ ├── config.py # Configuration settings
│ ├── components/
│ │ ├── detection.py # YOLO detection logic
│ │ ├── privacy.py # Face/plate anonymization
│ │ ├── alerts.py # Warning calculations
│ │ ├── mapping.py # Geospatial features
│ │ └── utils.py # Helper functions
│ └── pages/
│ ├── demo.py # Live demo interface
│ ├── map_view.py # Interactive map
│ ├── privacy_test.py # Privacy testing
│ └── metrics.py # Performance dashboard
├── src/
│ └── postprocessing/
│ └── deduplication.py # Spatial clustering
├── tests/
│ ├── test_detection.py # Detection tests
│ └── test_deduplication.py # Deduplication tests
├── scripts/
│ └── init_database.py # DB initialization
├── models/
│ └── best.pt # Trained YOLOv11n model
├── data/
│ ├── hazards.db # SQLite database
│ └── sample_videos/ # Test videos
├── docs/
│ └── images/ # Documentation images
├── pyproject.toml # Project metadata & deps
├── uv.lock # Locked dependencies
├── README.md # This file
└── LICENSE # MIT License

text

### Key Components

| Component | Purpose | Lines of Code |
|-----------|---------|---------------|
| `detection.py` | YOLO inference, drawing | ~400 |
| `privacy.py` | Face/plate detection & blur | ~350 |
| `mapping.py` | Folium maps, DB queries | ~400 |
| `deduplication.py` | DBSCAN clustering | ~300 |
| **Total** | **Core functionality** | **~1500** |

## Model Training

### Training Details

- **Framework**: Ultralytics YOLOv11n
- **Architecture**: YOLOv11n (2.6M parameters)
- **Dataset**: Custom road hazard dataset (6,439+ images)
- **Training Time**: ~2 hours on RTX 3050 GPU
- **Augmentations**: Mosaic, mixup, rotation, scaling, HSV

### Training Configuration

Training configuration used
model = YOLO('yolo11n.pt')
results = model.train(
data='road_hazards.yaml',
epochs=65,
imgsz=640,
batch=20,
optimizer='AdamW',
lr0=0.002,
lrf=0.001,
augment=True,
cos_lr=True,
cls=3.0, # Increased class loss weight
box=7.5,
patience=30
)

text

### Dataset Statistics

- **Total Images**: 6,439 (training) + 1,619 (validation)
- **Training Split**: 80% / 20%
- **Classes**: 5 road hazard types
- **Training Device**: RTX 3050 Laptop GPU (4GB VRAM)

**Class Distribution**:

| Class | Training Instances | Difficulty |
|-------|-------------------|------------|
| Longitudinal Crack | 1,176 | Medium |
| Transverse Crack | 1,183 | Easy |
| Alligator Crack | 12 | Very Hard |
| Pothole | 439 | Medium |
| Other Corruption | Variable | Unknown |

## Performance Metrics

### Model Accuracy

| Metric | Value | Benchmark |
|--------|-------|-----------|
| **mAP@0.5** | 50.3% | Good |
| **mAP@0.5:0.95** | 23.0% | COCO Standard |
| **Precision** | 57.9% | Good |
| **Recall** | 43.1% | Fair |

### Per-Class Performance

| Class | mAP@0.5 | Precision | Recall | F1-Score |
|-------|---------|-----------|--------|----------|
| Longitudinal Crack | 58.9% | 61.0% | 54.5% | 57.6% |
| Transverse Crack | 71.7% | 78.1% | 62.8% | 69.7% |
| Alligator Crack | 10.1% | 12.0% | 8.3% | 9.8% |
| Pothole | 60.1% | 69.9% | 51.5% | 59.4% |
| **Average** | **50.3%** | **57.9%** | **43.1%** | **49.1%** |

### System Performance

| Resolution | Inference Time | FPS | Memory |
|------------|----------------|-----|--------|
| 640×640 | ~45 ms | 22 | 1.2 GB |
| 1280×720 | ~78 ms | 13 | 1.8 GB |
| 1920×1080 | ~142 ms | 7 | 2.4 GB |

**Hardware**: RTX 3050 Laptop GPU, 8GB RAM

### Training Hardware

- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)
- **Training Time**: ~2 hours for 65 epochs
- **Batch Size**: 20 (limited by VRAM)
- **Memory Usage**: ~2.9GB GPU memory during training

## Technology Stack

### Core Technologies

| Category | Technology | Version | Purpose |
|----------|------------|---------|---------|
| **Language** | Python | 3.11+ | Primary language |
| **Package Manager** | UV | 0.4+ | Fast dependency management |
| **Web Framework** | Streamlit | 1.28+ | Interactive UI |
| **Deep Learning** | Ultralytics YOLOv11 | 8.3+ | Object detection |
| **Computer Vision** | OpenCV | 4.8+ | Image processing |
| **Data Science** | NumPy, Pandas | Latest | Data manipulation |
| **Database** | SQLite | 3 | Hazard storage |
| **Maps** | Folium | 0.15+ | Interactive maps |
| **ML Utilities** | scikit-learn | 1.3+ | Clustering (DBSCAN) |

### Development Tools

- **Testing**: pytest, pytest-cov
- **Linting**: ruff, black, isort
- **Type Checking**: mypy
- **Security**: bandit, safety
- **CI/CD**: GitHub Actions
- **Version Control**: Git, GitHub

### Visualization Libraries

- **Plotly**: Interactive charts
- **Matplotlib**: Static plots
- **Streamlit Components**: Custom widgets

## Hackathon Context

### Challenge

**VW Hackathon 2025**: Smart Mobility & Road Safety Track

**Problem Statement**: Develop an AI-powered system to detect and alert drivers about road hazards in real-time, improving road safety and enabling predictive maintenance.

### Our Solution: RoadGuard

#### Unique Selling Points (USPs)

1. **Custom YOLOv11n Training**
   - Trained specifically on road hazard dataset
   - 50.3% mAP@0.5 performance on 5 hazard classes
   - Optimized for real-time inference (45ms)
   - RTX 3050 training with 65 epochs

2. **GDPR-Compliant Privacy**
   - Automatic face and license plate anonymization
   - Multiple blurring methods (Gaussian, pixelate, blackout)
   - Verification system confirms compliance
   - Production-ready for public deployment

3. **Spatial Intelligence**
   - DBSCAN clustering prevents duplicate reports
   - Time-based confidence decay
   - Geographic heatmaps for infrastructure planning
   - Proximity search with Haversine distance

4. **Production-Ready Architecture**
   - Comprehensive test coverage
   - CI/CD pipeline with GitHub Actions
   - Modular design for easy extension
   - Well-documented codebase

#### Business Value

- **Road Authorities**: Data-driven maintenance prioritization
- **Fleet Operators**: Real-time driver safety alerts
- **Insurance Companies**: Risk assessment and prevention
- **Navigation Apps**: Enhanced routing with hazard awareness

## Future Roadmap

### Short-term (1-3 months)

- **Mobile App Development**: React Native iOS/Android app
- **Cloud Deployment**: AWS/Azure deployment with auto-scaling
- **Real-time Dashboard**: Fleet management interface
- **API Development**: RESTful API for third-party integration
- **Enhanced Models**: YOLOv11s/m for higher accuracy

### Mid-term (3-6 months)

- **Edge Device Deployment**: Raspberry Pi, Jetson Nano
- **Multi-language Support**: Hindi, Spanish, French
- **Weather Integration**: Adjust alerts based on weather
- **Crowdsourcing**: Community-based hazard reporting
- **Advanced Analytics**: Predictive maintenance ML models

### Long-term (6-12 months)

- **V2X Integration**: Vehicle-to-Everything communication
- **3D Depth Estimation**: Stereo cameras for depth
- **Semantic Segmentation**: Pixel-level road condition mapping
- **Government Partnerships**: Integration with DOT systems

## Contributing

We welcome contributions from the community! Here's how you can help:

### Ways to Contribute

1. **Report Bugs**: Open an issue with detailed reproduction steps
2. **Suggest Features**: Share your ideas in discussions
3. **Improve Documentation**: Fix typos, add examples
4. **Write Tests**: Increase test coverage
5. **Enhance UI**: Improve Streamlit interface
6. **Fix Issues**: Pick up "good first issue" labels

### Development Workflow

1. Fork the repository
2. Clone your fork
3. Create a feature branch
4. Make your changes
5. Run tests: `uv run pytest tests/ -v`
6. Check code quality: `uv run ruff check .`
7. Commit with conventional commits
8. Push to your fork
9. Open a Pull Request

### Code Style

- **Python**: PEP 8, enforced by Black and Ruff
- **Commits**: Conventional Commits (feat, fix, docs, etc.)
- **Docstrings**: Google style
- **Tests**: pytest with 80%+ coverage target

## License

This project is licensed under the **MIT License** - see the LICENSE file for details.

## Team & Acknowledgments

### Team Autono Minds

- **Swastidip Maji** - Team Lead, ML Engineer
  - Role: Model training, system architecture
  - Contact: swastidip2004@gmail.com

### Acknowledgments

- **VW Hackathon 2025** - For organizing the event and providing the challenge
- **Ultralytics** - For the excellent YOLOv11 framework
- **Kaggle** - For providing GPU compute for training
- **Streamlit** - For the intuitive web framework
- **Open Source Community** - For the amazing tools and libraries

### Citations

@software{roadguard2025,
title = {RoadGuard: AI-Powered Road Hazard Detection},
author = {Team Autono Minds},
year = {2025},
url = {https://github.com/Swastidp/Roadguard-MVP}
}

@software{yolov11_ultralytics,
title = {Ultralytics YOLOv11},
author = {Glenn Jocher and others},
year = {2023},
url = {https://github.com/ultralytics/ultralytics}
}

text

## Contact & Support

- **Email**: swastidip2004@gmail.com
- **GitHub**: [Roadguard-MVP Issues](https://github.com/Swastidp/Roadguard-MVP/issues)
- **Repository**: [Roadguard-MVP](https://github.com/Swastidp/Roadguard-MVP)

### Stay Updated

- Star this repo to stay notified
- Watch releases for new versions
- Fork to experiment with your own ideas

**Made by Team Autono Minds for VW Hackathon 2025**

---

## About

Road hazard detection system using YOLOv11n for real-time identification of potholes, cracks, and road damage with privacy protection and spatial intelligence.

### Resources

- [Documentation](./docs/)
- [Training Notebook](./docs/notebooks/)
- [Performance Metrics](./docs/metrics/)

### Languages

- Python 100.0%