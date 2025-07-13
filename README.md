# 🔍 PCB Defect Detection AI (Streamlit App)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**A modern, AI-powered web application for PCB defect detection using YOLOv8 and Streamlit**

*Developed by [Raj Patel](https://www.linkedin.com/in/raj-patel5)*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Raj%20Patel-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/raj-patel5)

</div>

---

## 🚀 Features

- **🔍 Advanced Defect Detection**: Upload single or multiple PCB images for AI-powered defect analysis
- **📊 Batch Processing**: Process multiple images simultaneously with comprehensive analytics
- **📈 Interactive Analytics**: Visualize defect distributions with interactive charts and statistics
- **💾 Download Results**: Export all defect-annotated images as a ZIP archive
- **🎨 Modern UI**: Clean, glassmorphism-inspired responsive interface
- **⚡ High Performance**: Powered by finetuned YOLOv8 model for accurate detection
- **🌐 Cross-Platform**: Works seamlessly on Windows, Linux (Ubuntu), and Streamlit Cloud
- **🖥️ GPU Support**: Automatic GPU acceleration when available (CUDA compatible)

---

## 📁 Project Structure

```
clean_project/
├── 📂 src/
│   ├── 🐍 web_interface.py      # Main Streamlit web interface
│   ├── 🧠 model.py             # YOLOv8 model and detection logic
│   └── 📄 __init__.py          # Package initialization
├── 🤖 model/
│   └── pcb_defect_detection_finetuned.pt  # Trained YOLOv8 model
├── 📊 data/
│   └── data.yaml               # Dataset configuration
├── 🚀 main.py                  # Application entry point
├── 📋 requirements.txt         # Python dependencies
├── ⚙️ config.yaml             # Application configuration
└── 📖 README.md               # Project documentation
```

---

## 🛠️ Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| **Frontend** | Streamlit | 1.28+ |
| **AI Model** | YOLOv8 (Ultralytics) | Latest |
| **Deep Learning** | PyTorch | 2.0+ |
| **Image Processing** | OpenCV | 4.8+ |
| **Data Visualization** | Plotly | 5.0+ |
| **Data Handling** | Pandas, NumPy | Latest |
| **Configuration** | PyYAML | Latest |

---

## 📦 Dependencies

### Core Dependencies (Required for Deployment)
```bash
streamlit          # Web application framework
ultralytics        # YOLOv8 model and utilities
opencv-python      # Image processing and computer vision
numpy              # Numerical computing
pillow             # Image I/O operations
matplotlib         # Plotting and visualization
plotly             # Interactive charts and graphs
pyyaml             # Configuration file parsing
pandas             # Data manipulation and analysis
```

### Development Dependencies (Optional)
```bash
torch              # Deep learning framework (GPU/CPU)
torchvision        # Computer vision utilities
supervision        # Advanced annotation tools
tqdm               # Progress bars for training
```

---

## 🖥️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/R204570/PCB_DEFECT_DETECTION.git
```

### 2. Set Up Python Environment

## Windows

```bash
cd PCB_DEFECT_DETECTION
python -m venv venv
venv\Scripts\activate
```
## Linux / Unubtu 

```bash
cd PCB_DEFECT_DETECTION
python3 -m venv venv
source venv/bin/activate
```
### 3. Install Dependencies

#### **Option A: CPU Installation (Recommended for Streamlit Cloud)**

```bash
pip install -r requirements.txt
```

#### **Option B: GPU Installation (For Local Development with NVIDIA GPU)**
```bash

# Install PyTorch with CUDA support (adjust version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

```

---

### Run Application

```bash
# Run the application
python main.py web
```

The application will open in your browser at `http://localhost:8501`

### Streamlit Cloud Deployment
1. Push your repository to GitHub
2. Connect your GitHub repo to [Streamlit Cloud](https://streamlit.io/cloud)
3. Set the entry point to `PCB_DEFECT_DETECTION/main.py`
4. Deploy and share your app!

---

## 🧠 Model Information

- **Model Type**: YOLOv8 (You Only Look Once v8)
- **Training Data**: Custom PCB defect dataset
- **Model File**: `model/pcb_defect_detection_finetuned.pt`
- **Input Size**: Configurable (default: 640x640)
- **Supported Defects**: Multiple PCB manufacturing defects
- **Performance**: Optimized for real-time inference

---

## 💡 Usage Guide

### Single Image Analysis
1. Navigate to the "Single Analysis" tab
2. Upload a PCB image
3. Adjust detection parameters if needed
4. View detection results with bounding boxes
5. Download annotated image

### Batch Processing
1. Go to the "Batch Processing" tab
2. Upload multiple PCB images (max 10)
3. Monitor processing progress
4. View defect analytics and distribution charts
5. Download all annotated images as ZIP

### Live Capture
1. Use the "Live Capture" tab
2. Take photos using your device camera
3. Get real-time defect detection results

---

## 🛠️ Configuration

The application can be customized through `config.yaml`:

```yaml
model:
  confidence_threshold: 0.25
  nms_threshold: 0.4
  input_size: [640, 640]

data:
  batch_size: 16
  output_path: "output"

training:
  epochs: 100
```

---

## 🚨 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Missing model file** | Ensure `model/pcb_defect_detection_finetuned.pt` exists |
| **CUDA not available** | App automatically falls back to CPU |
| **Slow inference** | Use GPU-enabled installation for faster results |
| **ModuleNotFoundError** | Verify virtual environment and requirements installation |
| **Streamlit Cloud errors** | Check repository size and file structure |

---

## 📊 Performance Metrics

- **Inference Speed**: ~50ms per image (GPU), ~200ms per image (CPU)
- **Accuracy**: mAP@0.5 > 85% on test dataset
- **Supported Formats**: JPG, PNG, BMP
- **Max Image Size**: 4096x4096 pixels
- **Batch Processing**: Up to 10 images simultaneously

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Developer

**Raj Patel**
- 🔗 [LinkedIn Profile](https://www.linkedin.com/in/raj-patel5)
- 🎯 AI/ML Learner
- 🚀 Passionate about AI 

---

## 🙏 Acknowledgments

- **Ultralytics** for the YOLOv8 framework
- **Streamlit** for the amazing web app framework
- **OpenCV** for computer vision capabilities
- **PyTorch** for deep learning infrastructure

---

<div align="center">

**Made with ❤️ using Streamlit and YOLOv8**

[![Star on GitHub](https://img.shields.io/github/stars/yourusername/yourrepo?style=social)](https://github.com/yourusername/yourrepo)
[![Follow on LinkedIn](https://img.shields.io/badge/Follow-Raj%20Patel-blue?style=social&logo=linkedin)](https://www.linkedin.com/in/raj-patel5)

</div> 
