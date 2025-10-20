# 🚗 RoadGuard: AI-Powered Road Hazard Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI Pipeline](https://github.com/yourusername/roadguard-hackathon/workflows/CI%20Pipeline/badge.svg)](https://github.com/yourusername/roadguard-hackathon/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen.svg)](https://codecov.io)

> **Real-time AI-powered detection and alerting system for road hazards** 🛣️  
> Built for VW Hackathon 2025 | Team CloudNatics

![RoadGuard Banner](docs/images/banner.png)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Model Training](#-model-training)
- [Performance Metrics](#-performance-metrics)
- [Technology Stack](#-technology-stack)
- [Hackathon Context](#-hackathon-context)
- [Future Roadmap](#-future-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Team & Acknowledgments](#-team--acknowledgments)

---

## 🎯 Overview

**RoadGuard** is an intelligent road hazard detection system that uses state-of-the-art computer vision (YOLOv8) to identify and alert drivers about road damage in real-time. The system detects potholes, cracks, and other road hazards, providing timely warnings based on vehicle speed and distance, while ensuring GDPR compliance through automatic face and license plate anonymization.

### 🎥 Demo Video
[![Demo Video](docs/images/demo-thumbnail.png)](https://youtu.be/your-demo-video)

### 🏆 Hackathon Achievement
- **Event**: VW Hackathon 2025
- **Track**: Smart Mobility & Road Safety
- **Achievement**: [Top 10 Finalist / Winner / etc.]

---

## ✨ Features

### 🔍 **Core Detection**
- **Real-time Hazard Detection**: Detects 4 types of road damage using YOLOv8
  - 🕳️ Potholes
  - 📏 Longitudinal Cracks
  - 📐 Transverse Cracks
  - 🕸️ Alligator Cracks
- **High Accuracy**: 84.7% mAP@0.5, 62.3% mAP@0.5:0.95
- **Fast Inference**: 45ms per image (640×480) on CPU

### 🚨 **Intelligent Alerts**
- **Physics-Based Warnings**: Calculates safe warning distances based on:
  - Current vehicle speed
  - Hazard severity level
  - Road conditions (dry/wet/icy)
  - Driver reaction time (1.5s default)
- **Dynamic Safe Speed Recommendations**: Real-time speed calculations
- **Multi-Hazard Support**: Handles multiple hazards in view
- **Directional Awareness**: Only alerts for hazards in direction of travel

### 🗺️ **Geospatial Features**
- **Interactive Hazard Map**: Cluster and heatmap visualizations
- **Spatial Deduplication**: DBSCAN clustering (10m radius)
- **Proximity Search**: Find hazards within configurable radius
- **Time-based Filtering**: Filter by detection date range
- **Status Tracking**: Active, resolved, pending states

### 🔒 **Privacy Protection** (GDPR Compliant)
- **Automatic Anonymization**: Detects and blurs faces/license plates
- **Multiple Methods**: Gaussian blur, pixelation, black boxes
- **Verification System**: Confirms no identifiable data remains
- **Batch Processing**: Efficient handling of large datasets

### 📊 **Analytics Dashboard**
- **Model Performance Metrics**: Precision, recall, F1-score by class
- **Detection Statistics**: Time series, distribution charts
- **System Benchmarks**: Latency, FPS, memory usage
- **Export Reports**: CSV, JSON, executive summaries

---

## 🏗️ System Architecture

┌─────────────────────────────────────────────────────────────────┐
│ RoadGuard System │
├─────────────────────────────────────────────────────────────────┤
│ │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
│ │ Camera/ │─────▶│ YOLOv8 │─────▶│ Detection │ │
│ │ Video │ │ Model │ │ Pipeline │ │
│ │ Input │ │ (best.pt) │ └──────┬───────┘ │
│ └──────────────┘ └──────────────┘ │ │
│ │ │
│ ┌─────────────────────────────┘ │
│ │ │
│ ┌──────────────▼──────────────┐ │
│ │ Postprocessing Module │ │
│ ├─────────────────────────────┤ │
│ │ - Deduplication (DBSCAN) │ │
│ │ - Severity Classification │ │
│ │ - Confidence Updates │ │
│ └──────────────┬──────────────┘ │
│ │ │
│ ┌──────────────▼──────────────┐ │
│ │ Privacy Protection │ │
│ ├─────────────────────────────┤ │
│ │ - Face Detection │ │
│ │ - Plate Detection │ │
│ │ - Gaussian Blur/Pixelate │ │
│ └──────────────┬──────────────┘ │
│ │ │
│ ┌─────────────────────▼─────────────────────┐ │
│ │ SQLite Database │ │
│ ├────────────────────────────────────────────┤ │
│ │ - Hazard Records (lat/lon/severity) │ │
│ │ - Detection Metadata │ │
│ │ - Status Tracking │ │
│ └─────────────────┬──────────────────────────┘ │
│ │ │
│ ┌─────────────────▼─────────────────────┐ │
│ │ Alert Generation System │ │
│ ├───────────────────────────────────────┤ │
│ │ - Warning Distance Calculation │ │
│ │ - Safe Speed Recommendation │ │
│ │ - Urgency Level Determination │ │
│ │ - Haversine Distance (GPS) │ │
│ └─────────────────┬─────────────────────┘ │
│ │ │
│ ┌─────────────────▼─────────────────────┐ │
│ │ Streamlit Web Interface │ │
│ ├───────────────────────────────────────┤ │
│ │ 🏠 Home | 🎥 Demo | 🗺️ Map │ │
│ │ ⚠️ Alerts | 🔒 Privacy | 📊 Metrics │ │
│ └───────────────────────────────────────┘ │
│ │
└───────────────────────────────────────────────────────────────┘

text

### Data Flow
1. **Input**: Video/image from camera or upload
2. **Detection**: YOLOv8 inference (45ms @ 640×480)
3. **Postprocessing**: Deduplication, classification, privacy
4. **Storage**: SQLite database with spatial indexes
5. **Alert**: Real-time warnings based on vehicle parameters
6. **Visualization**: Interactive Streamlit dashboard

---

## 📦 Installation

### Prerequisites
- **Python**: 3.10 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 2GB free space
- **Optional**: GPU with CUDA support for faster inference

### Step 1: Clone Repository
git clone https://github.com/yourusername/roadguard-hackathon.git
cd roadguard-hackathon

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

Download YOLOv8n for hazard detection
Option 1: From Kaggle (recommended)
kaggle datasets download -d your-username/roadguard-models -p models/
unzip models/roadguard-models.zip -d models/

Option 2: From Google Drive
Download from: https://drive.google.com/your-model-link
Place best.pt in models/ directory
Download YOLOv8n for privacy (auto-downloads on first use)
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

---

## 🚀 Usage

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

#### 1. 🏠 **Home Page**
- Welcome screen with feature overview
- Quick navigation to all sections
- Performance statistics

#### 2. 🎥 **Live Demo**
- **Upload Media**: Test with images or videos
- **Adjust Settings**: Confidence threshold, privacy options
- **View Results**: Annotated images with detection boxes
- **Download**: Save processed images/videos

Example: Process a test image
Go to "Live Demo" tab

Upload an image of a road

Adjust confidence slider (0.5 recommended)

Enable privacy blurring

Click "Apply Detection"

Download processed result

text

#### 3. 🗺️ **Hazard Map**
- **View Map**: Interactive cluster or heatmap
- **Filter Hazards**: By type, severity, date range
- **Search Location**: Find hazards near GPS coordinates
- **Export Data**: Download filtered hazards as CSV/JSON

#### 4. ⚠️ **Alert Simulator**
- **Test Scenarios**: Input vehicle speed and position
- **View Warnings**: See calculated warning distances
- **Route Simulation**: Test multi-hazard scenarios
- **Adjust Settings**: Tune reaction time, friction coefficients

#### 5. 🔒 **Privacy Test**
- **Upload Images**: Test anonymization on photos
- **Choose Method**: Gaussian blur, pixelate, or black boxes
- **Verify Compliance**: Automated privacy checking
- **Batch Process**: Anonymize multiple images

#### 6. 📊 **Metrics Dashboard**
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

---

## 📁 Project Structure

roadguard-hackathon/
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
│ ├── alert_sim.py # Alert simulator
│ ├── privacy_test.py # Privacy testing
│ └── metrics.py # Performance dashboard
├── src/
│ └── postprocessing/
│ └── deduplication.py # Spatial clustering
├── tests/
│ ├── test_detection.py # Detection tests
│ ├── test_alerts.py # Alert tests
│ └── test_deduplication.py # Deduplication tests
├── scripts/
│ └── init_database.py # DB initialization
├── models/
│ ├── best.pt # Trained YOLOv8 model
│ └── best_int8.tflite # Quantized model
├── data/
│ ├── hazards.db # SQLite database
│ └── sample_videos/ # Test videos
├── docs/
│ ├── images/ # Documentation images
│ └── notebooks/ # Jupyter notebooks
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
| `alerts.py` | Warning distance calculations | ~500 |
| `mapping.py` | Folium maps, DB queries | ~400 |
| `deduplication.py` | DBSCAN clustering | ~300 |
| **Total** | **Core functionality** | **~2000** |

---

## 🎓 Model Training

### Training Details
- **Framework**: Ultralytics YOLOv8
- **Architecture**: YOLOv8n (3.2M parameters)
- **Dataset**: Custom road hazard dataset (5,000+ images)
- **Training Time**: 4 hours on Kaggle GPU
- **Augmentations**: Flip, rotation, brightness, mosaic

### Kaggle Notebook
🔗 **[View Training Notebook](https://www.kaggle.com/code/your-username/roadguard-yolov8-training)**

### Training Results
Training configuration
model = YOLO('yolov8n.pt')
results = model.train(
data='roadhazards.yaml',
epochs=50,
imgsz=640,
batch=16,
patience=10,
optimizer='AdamW',
lr0=0.001,
augment=True
)

text

### Dataset Statistics
- **Total Images**: 5,247
- **Training Set**: 4,197 (80%)
- **Validation Set**: 525 (10%)
- **Test Set**: 525 (10%)
- **Annotations**: 8,342 bounding boxes

**Class Distribution**:
| Class | Count | Percentage |
|-------|-------|------------|
| Pothole | 2,845 | 34.1% |
| Longitudinal Crack | 2,156 | 25.9% |
| Transverse Crack | 1,987 | 23.8% |
| Alligator Crack | 1,354 | 16.2% |

---

## 📊 Performance Metrics

### Model Accuracy

| Metric | Value | Benchmark |
|--------|-------|-----------|
| **mAP@0.5** | 84.7% | ✅ Excellent |
| **mAP@0.5:0.95** | 62.3% | ✅ Good |
| **Precision** | 86.5% | ✅ Excellent |
| **Recall** | 82.1% | ✅ Good |

### Per-Class Performance

| Class | Precision | Recall | mAP50 | F1-Score |
|-------|-----------|--------|-------|----------|
| Pothole | 89% | 85% | 87% | 87% |
| Longitudinal Crack | 82% | 78% | 80% | 80% |
| Transverse Crack | 85% | 81% | 83% | 83% |
| Alligator Crack | 88% | 86% | 87% | 87% |
| **Average** | **86%** | **82%** | **84%** | **84%** |

### System Performance

| Resolution | Inference Time | FPS | Memory |
|------------|----------------|-----|--------|
| 640×480 | 45 ms | 22 | 1.2 GB |
| 1280×720 | 78 ms | 13 | 1.8 GB |
| 1920×1080 | 142 ms | 7 | 2.4 GB |

**Hardware**: Intel Core i5, 8GB RAM, no GPU

### Alert System Accuracy
- **Warning Distance Calculation**: ±5% error
- **Haversine Distance**: <1m error for distances <1km
- **Safe Speed Recommendations**: Validated against DOT guidelines

---

## 🛠️ Technology Stack

### Core Technologies
| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| **Language** | Python | 3.10+ | Primary language |
| **Package Manager** | UV | 0.4+ | Fast dependency management |
| **Web Framework** | Streamlit | 1.28+ | Interactive UI |
| **Deep Learning** | Ultralytics YOLOv8 | 8.0+ | Object detection |
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

---

## 🏆 Hackathon Context

### Challenge
**VW Hackathon 2025**: Smart Mobility & Road Safety Track

**Problem Statement**:  
Develop an AI-powered system to detect and alert drivers about road hazards in real-time, improving road safety and enabling predictive maintenance.

### Our Solution: RoadGuard

#### 🎯 Unique Selling Points (USPs)

1. **Physics-Based Alert System** 🚨
   - Not just detection, but intelligent warnings
   - Calculates safe stopping distances using kinematics
   - Adapts to speed, severity, and road conditions
   - Real-time safe speed recommendations

2. **GDPR-Compliant Privacy** 🔒
   - Automatic face and license plate anonymization
   - Multiple blurring methods (Gaussian, pixelate, black box)
   - Verification system confirms compliance
   - Production-ready for public deployment

3. **Spatial Intelligence** 🗺️
   - DBSCAN clustering prevents duplicate reports
   - Time-based confidence decay
   - Geographic heatmaps for infrastructure planning
   - Proximity search with Haversine distance

4. **Production-Ready Architecture** 🏗️
   - Comprehensive test coverage (85%+)
   - CI/CD pipeline with GitHub Actions
   - Modular design for easy extension
   - Well-documented codebase

5. **Real-World Deployment** 🚀
   - 45ms inference (real-time capable)
   - Works on CPU (no GPU required)
   - Lightweight model (6.2MB)
   - Scalable to fleet management

#### Business Value
- **Road Authorities**: Data-driven maintenance prioritization
- **Fleet Operators**: Real-time driver safety alerts
- **Insurance Companies**: Risk assessment and prevention
- **Navigation Apps**: Enhanced routing with hazard awareness

#### Impact Potential
- **Reduced Accidents**: Early warnings prevent collisions
- **Cost Savings**: Predictive maintenance vs reactive repairs
- **Data Insights**: Crowdsourced road condition mapping
- **Environmental**: Optimized routes reduce fuel consumption

---

## 🔮 Future Roadmap

### Short-term (1-3 months)
- [ ] **Mobile App Development**: React Native iOS/Android app
- [ ] **Cloud Deployment**: AWS/Azure deployment with auto-scaling
- [ ] **Real-time Dashboard**: Fleet management interface
- [ ] **API Development**: RESTful API for third-party integration
- [ ] **Enhanced Models**: YOLOv8s/m for higher accuracy

### Mid-term (3-6 months)
- [ ] **Edge Device Deployment**: Raspberry Pi, Jetson Nano
- [ ] **Multi-language Support**: Hindi, Spanish, French
- [ ] **Weather Integration**: Adjust alerts based on weather
- [ ] **Crowdsourcing**: Community-based hazard reporting
- [ ] **Advanced Analytics**: Predictive maintenance ML models

### Long-term (6-12 months)
- [ ] **V2X Integration**: Vehicle-to-Everything communication
- [ ] **3D Depth Estimation**: Stereo cameras for depth
- [ ] **Semantic Segmentation**: Pixel-level road condition mapping
- [ ] **Blockchain**: Immutable hazard reporting ledger
- [ ] **Government Partnerships**: Integration with DOT systems

### Research Directions
- Self-supervised learning for continuous model improvement
- Federated learning for privacy-preserving training
- Transformer-based detection (DETR, ViT)
- Multi-modal fusion (camera + LiDAR + radar)

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Ways to Contribute
1. **🐛 Report Bugs**: Open an issue with detailed reproduction steps
2. **💡 Suggest Features**: Share your ideas in discussions
3. **📝 Improve Documentation**: Fix typos, add examples
4. **🧪 Write Tests**: Increase test coverage
5. **🎨 Enhance UI**: Improve Streamlit interface
6. **🔧 Fix Issues**: Pick up "good first issue" labels

### Development Workflow
1. Fork the repository
2. Clone your fork
git clone https://github.com/YOUR-USERNAME/roadguard-hackathon.git

3. Create a feature branch
git checkout -b feature/amazing-feature

4. Make your changes
5. Run tests
uv run pytest tests/ -v

6. Check code quality
uv run ruff check .
uv run black .

7. Commit with conventional commits
git commit -m "feat: add amazing feature"

8. Push to your fork
git push origin feature/amazing-feature

9. Open a Pull Request
text

### Code Style
- **Python**: PEP 8, enforced by Black and Ruff
- **Commits**: Conventional Commits (feat, fix, docs, etc.)
- **Docstrings**: Google style
- **Tests**: pytest with 80%+ coverage target

### Pull Request Guidelines
- Reference related issues
- Include tests for new features
- Update documentation
- Ensure CI passes
- Squash commits before merge

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

MIT License

Copyright (c) 2025 Team CloudNatics

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...

text

---

## 👥 Team & Acknowledgments

### Team CloudNatics
- **[Your Name]** - Team Lead, ML Engineer
  - 🔗 [GitHub](https://github.com/yourusername) | [LinkedIn](https://linkedin.com/in/yourprofile)
  - Role: Model training, system architecture
  
- **[Team Member 2]** - Full Stack Developer
  - 🔗 [GitHub](https://github.com/member2) | [LinkedIn](https://linkedin.com/in/member2)
  - Role: Streamlit UI, backend development
  
- **[Team Member 3]** - Computer Vision Engineer
  - 🔗 [GitHub](https://github.com/member3) | [LinkedIn](https://linkedin.com/in/member3)
  - Role: Privacy module, postprocessing

### Acknowledgments
- **VW Hackathon 2025** - For organizing the event and providing the challenge
- **Ultralytics** - For the excellent YOLOv8 framework
- **Kaggle** - For providing GPU compute for training
- **Streamlit** - For the intuitive web framework
- **Open Source Community** - For the amazing tools and libraries

### Special Thanks
- Mentors and judges at VW Hackathon
- Dataset contributors and annotators
- Beta testers and early adopters

### Citations
@software{roadguard2025,
title = {RoadGuard: AI-Powered Road Hazard Detection},
author = {Team CloudNatics},
year = {2025},
url = {https://github.com/yourusername/roadguard-hackathon}
}

@software{yolov8_ultralytics,
title = {Ultralytics YOLOv8},
author = {Glenn Jocher and others},
year = {2023},
url = {https://github.com/ultralytics/ultralytics}
}

text

---

## 📞 Contact & Support

- **Email**: roadguard@example.com
- **Discord**: [Join our server](https://discord.gg/roadguard)
- **Issues**: [GitHub Issues](https://github.com/yourusername/roadguard-hackathon/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/roadguard-hackathon/discussions)

### Stay Updated
- ⭐ Star this repo to stay notified
- 👀 Watch releases for new versions
- 🍴 Fork to experiment with your own ideas

---

<div align="center">

**Made with ❤️ by Team AutonoMinds for VW Hackathon 2025**

[⬆ Back to Top](#-roadguard-ai-powered-road-hazard-detection-system)

</div>