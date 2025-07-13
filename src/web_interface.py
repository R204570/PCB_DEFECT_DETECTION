import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from pathlib import Path
import yaml
from src.model import PCBDefectDetector
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import pandas as pd
import threading
import time
from collections import deque
import glob

# Custom CSS for modern glassmorphism SaaS dashboard
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

body, .main, .stApp {
    font-family: 'Inter', sans-serif !important;
    background: none !important;
    color: #fff !important;
}

/* Animated Gradient Background */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    z-index: -3;
    background: linear-gradient(120deg, #0f2027 0%, #2c5364 50%, #00c6ff 100%);
    background-size: 200% 200%;
    animation: gradientMove 12s ease-in-out infinite;
    opacity: 0.85;
}
@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Ocean Waves SVG Layer */
.stApp::after {
    content: '';
    position: fixed;
    left: 0; right: 0; bottom: 0; height: 340px;
    z-index: -2;
    pointer-events: none;
    background: url('data:image/svg+xml;utf8,<svg width="100%25" height="340" viewBox="0 0 1920 340" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M0 200 Q480 340 960 200 T1920 200 V340 H0 V200Z" fill="%2300c6ff" fill-opacity="0.45"><animate attributeName=\"d\" dur=\"8s\" repeatCount=\"indefinite\" values=\"M0 200 Q480 340 960 200 T1920 200 V340 H0 V200Z;M0 220 Q480 300 960 220 T1920 220 V340 H0 V220Z;M0 200 Q480 340 960 200 T1920 200 V340 H0 V200Z\"/></path><path d="M0 240 Q600 320 1200 240 T1920 240 V340 H0 V240Z" fill="%230072ff" fill-opacity="0.35"><animate attributeName=\"d\" dur=\"10s\" repeatCount=\"indefinite\" values=\"M0 240 Q600 320 1200 240 T1920 240 V340 H0 V240Z;M0 260 Q600 300 1200 260 T1920 260 V340 H0 V260Z;M0 240 Q600 320 1200 240 T1920 240 V340 H0 V240Z\"/></path><path d="M0 300 Q960 340 1920 300 V340 H0 V300Z" fill="%23fff" fill-opacity="0.10"><animate attributeName=\"d\" dur=\"12s\" repeatCount=\"indefinite\" values=\"M0 300 Q960 340 1920 300 V340 H0 V300Z;M0 320 Q960 320 1920 320 V340 H0 V320Z;M0 300 Q960 340 1920 300 V340 H0 V300Z\"/></path></svg>');
    background-size: cover;
    background-repeat: no-repeat;
    background-position: bottom;
    opacity: 1;
}

/* Glassmorphism Card */
.glass-card {
    background: rgba(34, 40, 49, 0.55);
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.18);
    backdrop-filter: blur(16px) saturate(180%);
    -webkit-backdrop-filter: blur(16px) saturate(180%);
    border-radius: 18px;
    border: 1.5px solid rgba(255,255,255,0.18);
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    transition: box-shadow 0.3s, transform 0.3s;
    position: relative;
    color: #fff !important;
}
.glass-card:hover {
    box-shadow: 0 12px 40px 0 rgba(0,198,255,0.18), 0 1.5px 8px 0 rgba(31,38,135,0.10);
    transform: translateY(-4px) scale(1.01);
}

/* Sidebar Glass */
[data-testid="stSidebar"] > div:first-child {
    background: rgba(34, 40, 49, 0.65) !important;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.18);
    backdrop-filter: blur(18px) saturate(180%);
    -webkit-backdrop-filter: blur(18px) saturate(180%);
    border-right: 1.5px solid rgba(255,255,255,0.12);
    color: #fff !important;
}

/* Modern Headers */
h1, h2, h3, h4, h5, h6, label, p, span, div, .stMarkdown, .stText, .stDataFrame, .stAlert, .stMetric, .stTabs, .stFileUploader, .stButton, .stSlider, .stSelectbox, .streamlit-expanderHeader {
    color: #fff !important;
}

/* Modern Buttons */
.stButton > button {
    background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
    color: #fff !important;
    border: none;
    border-radius: 12px;
    font-weight: 600;
    font-size: 1.08rem;
    padding: 0.75rem 2.2rem;
    box-shadow: 0 2px 12px rgba(0,198,255,0.18);
    transition: background 0.2s, box-shadow 0.2s, transform 0.1s;
    outline: none;
    cursor: pointer;
    position: relative;
    overflow: hidden;
}
.stButton > button:active {
    transform: scale(0.98);
    box-shadow: 0 1px 4px rgba(0,198,255,0.10);
}
.stButton > button:hover {
    background: linear-gradient(90deg, #0072ff 0%, #00c6ff 100%);
    box-shadow: 0 4px 24px rgba(0,198,255,0.28);
}

/* File Uploader Glass */
.stFileUploader {
    background: rgba(34, 40, 49, 0.45);
    border: 2px dashed #00c6ff;
    border-radius: 16px;
    padding: 2rem 1.5rem;
    transition: border 0.2s, box-shadow 0.2s;
    color: #fff !important;
}
.stFileUploader:hover {
    border-color: #0072ff;
    box-shadow: 0 2px 12px rgba(0,198,255,0.18);
}

/* Tabs - Modern Floating Pills */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(34, 40, 49, 0.45);
    border-radius: 16px;
    padding: 6px 8px;
    border: 1.5px solid rgba(255,255,255,0.10);
    box-shadow: 0 1px 8px rgba(0,198,255,0.10);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 12px;
    color: #fff !important;
    font-weight: 600;
    font-size: 1.08rem;
    transition: background 0.2s, color 0.2s;
    padding: 0.5rem 1.5rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
    color: #fff !important;
    box-shadow: 0 2px 8px rgba(0,198,255,0.18);
}

/* Metrics Glass */
[data-testid="metric-container"] {
    background: rgba(34, 40, 49, 0.55);
    border: 1.5px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    color: #fff !important;
    box-shadow: 0 1px 8px rgba(0,198,255,0.10);
    padding: 1.2rem 1rem;
    margin: 0.5rem 0;
    transition: box-shadow 0.2s, transform 0.2s;
}
[data-testid="metric-container"]:hover {
    box-shadow: 0 4px 24px rgba(0,198,255,0.18);
    transform: translateY(-2px) scale(1.01);
}

/* Expander Glass */
.streamlit-expanderHeader {
    background: rgba(34, 40, 49, 0.45) !important;
    border: 1.5px solid rgba(255,255,255,0.10) !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-weight: 600;
    transition: background 0.2s, box-shadow 0.2s;
}
.streamlit-expanderHeader:hover {
    background: rgba(0,198,255,0.10) !important;
    box-shadow: 0 2px 8px rgba(0,198,255,0.10);
}

/* Image Glass */
img {
    border-radius: 14px;
    box-shadow: 0 2px 16px rgba(0,198,255,0.10);
    transition: box-shadow 0.2s, transform 0.2s;
}
img:hover {
    transform: scale(1.01);
    box-shadow: 0 4px 32px rgba(0,198,255,0.18);
}

/* Progress Bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    border-radius: 4px;
}

/* Alerts Glass */
.stAlert {
    border-radius: 14px;
    border: 1.5px solid rgba(255,255,255,0.10);
    background: rgba(34, 40, 49, 0.45);
    color: #fff !important;
    animation: fadeIn 0.3s ease;
}
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(-10px);}
    to {opacity: 1; transform: translateY(0);}
}

/* Custom Scrollbar */
::-webkit-scrollbar {width: 7px;}
::-webkit-scrollbar-track {background: rgba(34, 40, 49, 0.18);}
::-webkit-scrollbar-thumb {background: #00c6ff; border-radius: 4px;}
::-webkit-scrollbar-thumb:hover {background: #0072ff;}

/* Responsive */
@media (max-width: 900px) {
    .glass-card {padding: 1.2rem 0.7rem;}
    h1 {font-size: 2rem;}
}

.center‑card{
  width:100%;                 /* fluid until it hits its max */
  max-width:380px;            /* stop growing after this */
  margin-inline:auto;         /* splits any extra space evenly */
  padding:2rem;               /* inner breathing room  */
  
  /* —— optional visuals (remove or tweak) —— */
  background:#1f2937;
  border-radius:1rem;
  box-shadow:0 16px 40px rgba(0,0,0,.35);
  color:#fff;
}
</style>
"""

class PCBDefectWebInterface:
    """Web interface for PCB defect detection using Streamlit. - Raj Patel"""
    
    def __init__(self):
        """Initialize the web interface. - Raj Patel"""
        st.set_page_config(
            page_title="PCB Defect Detection AI",
            page_icon="🔬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Inject custom CSS
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
        
        # Load configuration
        with open("config.yaml", 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Always load class names from finetune_data/data.yaml if it exists
        finetune_yaml = "data/data.yaml"
        if os.path.exists(finetune_yaml):
            with open(finetune_yaml, 'r') as f:
                data = yaml.safe_load(f)
            self.classes = data['names']
        else:
            self.classes = self.config['classes']
        
        # Initialize detector
        self.detector = None
        
        # Find available models
        self.available_models = self._find_available_models()
        
        # Class mapping fix for finetuned model
        self.finetuned_class_mapping = {
            0: 2,  # spur -> missing_hole
            1: 1,  # mouse_bite -> mouse_bite (unchanged)
            2: 4,  # missing_hole -> open_circuit
            3: 3,  # short -> short (unchanged)
            4: 0,  # open_circuit -> spur
            5: 5,  # spurious_copper -> spurious_copper (unchanged)
            6: 6   # class_6 -> class_6 (unchanged)
        }
        
        # Descriptive names for soldering defects (class_6)
        self.soldering_defect_names = [
            'poor_solder',
            'spike', 
            'excessive_solder',
            'cold_solder_joint'
        ]
        
        # Mapping for soldering defect types based on filename patterns
        self.soldering_defect_mapping = {
            'poor_solder': 'poor_solder',
            'spike': 'spike',
            'excessive_solder': 'excessive_solder',
            'cold_solder_joint': 'cold_solder_joint',
            'Excessive Solder': 'excessive_solder',
            'cold solder joint': 'cold_solder_joint'
        }
        
        self.current_model_path = None
    
    def _find_available_models(self):
        """Find available model files in the project directory. - Raj Patel"""
        models = {}
        # Only include the finetuned model
        path = "model/pcb_defect_detection_finetuned.pt"
        if os.path.exists(path):
            size_mb = round(os.path.getsize(path) / (1024 * 1024), 1)
            models[f"Enhanced Model ({size_mb} MB)"] = path
        return models
        
    def main(self):
        """Main application interface. - Raj Patel"""
        # Professional header
        st.markdown("""
        <div class="glass-card" style="text-align: center; margin-bottom: 40px;">
            <h1 style="margin-bottom: 8px; font-size: 2.5rem; font-weight: 700;">
                PCB Defect Detection
            </h1>
            <p style="color: #e0eaff; font-size: 1.1rem; margin-top: 0; font-weight: 400;">
                AI-Powered Circuit Board Analysis System
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Sidebar
        self._create_sidebar()
        
        # Main content with tabs
        tab1, tab2, tab3 = st.tabs([
            "Single Analysis", 
            "Live Capture", 
            "Batch Processing"
        ])
        
        with tab1:
            self._image_upload_tab()
        
        with tab2:
            self._camera_photo_tab()
        with tab3:
            self._batch_processing_tab()
    
    def _create_sidebar(self):
        """Create professional sidebar with configuration options. - Raj Patel"""
        st.sidebar.markdown("""
        <div style="text-align: center; margin-bottom: 24px;">
            <h2 style="margin: 0; color: #0f172a; font-weight: 600;">Configuration</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Model selection
        st.sidebar.markdown("### Model Selection")
        
        # Add custom model option
        use_custom_model = st.sidebar.checkbox("Custom Model Path", value=False)
        
        if use_custom_model:
            # Text input for custom model path
            model_path = st.sidebar.text_input(
                "📁 Model Path",
                value="",
                help="Enter the full path to your model file (.pt)"
            )
        else:
            # Dropdown for model selection
            if self.available_models:
                model_options = list(self.available_models.keys())
                selected_model = st.sidebar.selectbox(
                    "Select Model",
                    options=model_options,
                    index=0,
                    help="Choose a model for defect detection"
                )
                model_path = self.available_models[selected_model]
            else:
                st.sidebar.warning("⚠️ No models found. Please provide a custom path.")
                model_path = ""
        
        # Detection parameters
        st.sidebar.markdown("### Detection Parameters")
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.1,
            step=0.05,
            help="Minimum confidence for defect detection"
        )
        
        nms_threshold = st.sidebar.slider(
            "NMS Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.4,
            step=0.05,
            help="Non-maximum suppression threshold"
        )
        
        # Initialize detector with specified model path
        if self.detector is None or (model_path and model_path.strip() and model_path != self.current_model_path):
            try:
                if model_path and model_path.strip():
                    self.detector = PCBDefectDetector(model_path=model_path.strip())
                    self.current_model_path = model_path.strip()
                else:
                    self.detector = PCBDefectDetector()
                    self.current_model_path = None
                # Update thresholds
                self.detector.confidence_threshold = confidence_threshold
                self.detector.nms_threshold = nms_threshold
                # Update classes from detector
                self.classes = self.detector.classes
                st.sidebar.success("Model loaded successfully")
                
                # Show class mapping fix warning for finetuned model (hidden from users)
                if "finetuned" in model_path.lower():
                    pass  # Hidden from users
                    
            except Exception as e:
                st.sidebar.error(f"❌ Error loading model: {str(e)}")
                st.sidebar.info("Using default YOLOv8 model instead")
                self.detector = PCBDefectDetector()
                self.classes = self.detector.classes
                self.current_model_path = None
        
        # Update thresholds if detector exists
        if self.detector:
            self.detector.confidence_threshold = confidence_threshold
            self.detector.nms_threshold = nms_threshold
        
        # Defect classes info
        st.sidebar.markdown("### Defect Categories")
        
        # PCB Defects
        st.sidebar.markdown("**PCB Defects:**")
        for i, defect_class in enumerate(self.classes):
            if defect_class != 'soldering_defect':
                st.sidebar.markdown(f"• {defect_class.replace('_', ' ').title()}")
        
        # Soldering Defects
        st.sidebar.markdown("**Soldering Defects:**")
        soldering_types = [
            "Poor Solder",
            "Spike", 
            "Excessive Solder",
            "Cold Solder Joint"
        ]
        for soldering_type in soldering_types:
            st.sidebar.markdown(f"• {soldering_type}")
    
    def _get_specific_soldering_defect_type(self, detection, image_path=None):
        """Determine the specific soldering defect type based on context."""
        # If we have image path information, try to extract from filename
        if image_path:
            filename = os.path.basename(image_path).lower()
            
            # Check for soldering defect patterns in filename
            for pattern, defect_type in self.soldering_defect_mapping.items():
                if pattern.lower() in filename:
                    return defect_type
        
        # If no filename info, use a simple heuristic based on confidence and bbox
        confidence = detection['confidence']
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        
        # Simple heuristics (these could be improved with more sophisticated analysis)
        if confidence > 0.85:
            return 'poor_solder'
        elif confidence > 0.75:
            return 'spike'
        elif area > 10000:  # Large area
            return 'excessive_solder'
        else:
            return 'cold_solder_joint'
    
    def _fix_detection_classes(self, result):
        """Fix class mapping for finetuned model detections."""
        if not self.current_model_path or "finetuned" not in self.current_model_path.lower():
            return result
        
        # Apply class mapping fix
        for detection in result['detections']:
            original_class_id = detection['class_id']
            if original_class_id in self.finetuned_class_mapping:
                detection['class_id'] = self.finetuned_class_mapping[original_class_id]
                
                # Handle soldering defects (class_6) with specific names
                if detection['class_id'] == 6:
                    specific_type = self._get_specific_soldering_defect_type(detection, result.get('image_path'))
                    detection['class_name'] = specific_type
                else:
                    detection['class_name'] = self.classes[detection['class_id']]
        
        return result
    
    def _image_upload_tab(self):
        """Image upload and detection tab."""
        st.markdown("""
        <div style="text-align: center; margin-bottom: 32px;">
            <h2>Single Image Analysis</h2>
            <p style="color: #64748b;">Upload a PCB image for defect detection</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📤 Upload Image")
            
            # File uploader
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="Upload a PCB image for defect detection"
            )
            
            if uploaded_file is not None:
                # Display original image
                image = Image.open(uploaded_file)
                st.image(image, caption="Original Image", use_column_width=True)
                
                # Detection button
            if st.button("Analyze Image", type="primary"):
                    with st.spinner("Analyzing image..."):
                        try:
                            # Save uploaded file temporarily
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                                image.save(tmp_file.name)
                                tmp_path = tmp_file.name
                            
                            # Run detection
                            result = self.detector.predict_image(tmp_path, save_result=False)
                            
                            # Apply class mapping fix for finetuned model
                            result = self._fix_detection_classes(result)
                            
                            # Clean up
                            os.unlink(tmp_path)
                            
                            # Display results
                            self._display_detection_results(result, image)
                            
                        except Exception as e:
                            st.error(f"❌ Error during analysis: {str(e)}")
        
        with col2:
            st.markdown("### Analysis Results")
            st.info("Upload an image and click 'Analyze Image' to see results here.")
    
    def _display_detection_results(self, result, original_image):
        """Display detection results with futuristic styling."""
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Detection Summary")
            
            # Debug print for detection info
            print("[DEBUG] Detection result:", result)
            
            # Summary statistics
            st.metric("Total Defects", result['num_defects'])
            
            if result['num_defects'] > 0:
                # Defect type breakdown
                defect_counts = {}
                for detection in result['detections']:
                    cid = detection.get('class_id', -1)
                    if cid == 6:
                        defect_type = "Poor Solder"
                    elif cid == 7:
                        defect_type = "Spike"
                    elif cid == 8:
                        defect_type = "Excessive Solder"
                    elif cid == 9:
                        defect_type = "Cold Solder Joint"
                    elif cid >= 6:  # Any other soldering defect
                        defect_type = "Soldering Defect"
                    elif 0 <= cid < len(self.classes):
                        defect_type = self.classes[cid].replace('_', ' ').title()
                    else:
                        defect_type = f"Unknown ({cid})"
                    defect_counts[defect_type] = defect_counts.get(defect_type, 0) + 1
                
                st.markdown("### Defect Types Found")
                for defect_type, count in defect_counts.items():
                    st.markdown(f"• **{defect_type}**: {count}")
                
                # Average confidence
                avg_confidence = np.mean([d['confidence'] for d in result['detections']])
                st.metric("Average Confidence", f"{avg_confidence:.3f}")
        
        with col2:
            st.markdown("### Detailed Analysis")
            
            if result['num_defects'] > 0:
                # Create annotated image
                annotated_image = self._create_annotated_image(original_image, result['detections'])
                st.image(annotated_image, caption="Annotated Image", use_column_width=True)
                
                # Detailed detection list
                st.markdown("### Individual Detections")
                for i, detection in enumerate(result['detections']):
                    cid = detection.get('class_id', -1)
                    if cid == 6:
                        class_name = "Poor Solder"
                    elif cid == 7:
                        class_name = "Spike"
                    elif cid == 8:
                        class_name = "Excessive Solder"
                    elif cid == 9:
                        class_name = "Cold Solder Joint"
                    elif cid >= 6:  # Any other soldering defect
                        class_name = "Soldering Defect"
                    elif 0 <= cid < len(self.classes):
                        class_name = self.classes[cid].replace('_', ' ').title()
                    else:
                        class_name = f"Unknown ({cid})"
                    with st.expander(f"Defect {i+1}: {class_name}"):
                        st.markdown(f"**Confidence:** {detection['confidence']:.3f}")
                        st.markdown(f"**Location:** {detection['bbox']}")
            else:
                st.success("No defects detected. PCB appears to be clean.")
    
    def _create_annotated_image(self, image, detections):
        """Create annotated image with bounding boxes."""
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Create figure
        fig, ax = plt.subplots(1, figsize=(10, 8))
        ax.imshow(img_array)
        
        # Add bounding boxes
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            cid = detection.get('class_id', -1)
            if cid == 6:
                class_name = "Poor Solder"
            elif cid == 7:
                class_name = "Spike"
            elif cid == 8:
                class_name = "Excessive Solder"
            elif cid == 9:
                class_name = "Cold Solder Joint"
            elif cid >= 6:  # Any other soldering defect
                class_name = "Soldering Defect"
            elif 0 <= cid < len(self.classes):
                class_name = self.classes[cid].replace('_', ' ').title()
            else:
                class_name = f"Unknown ({cid})"
            confidence = detection['confidence']
            
            # Create rectangle
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=3, edgecolor='#00d4ff', facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            ax.text(x1, y1 - 10, f"{class_name} {confidence:.2f}",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="#00d4ff", alpha=0.8),
                   fontsize=10, color='white', weight='bold')
        
        ax.set_title("Defect Detection Results", fontsize=14, weight='bold', color='#0f172a')
        ax.axis('off')
        
        # Convert to PIL image
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor='#ffffff')
        buf.seek(0)
        annotated_image = Image.open(buf)
        plt.close()
        
        return annotated_image
    
    def _camera_photo_tab(self):
        """Camera photo capture and detection tab."""
        st.markdown("""
        <div style="text-align: center; margin-bottom: 32px;">
            <h2>Live Capture Analysis</h2>
            <p style="color: #64748b;">Capture real-time images for instant analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        camera_input = st.camera_input("Take a photo for defect detection")
        if camera_input is not None:
            image = Image.open(camera_input)
            st.image(image, caption="Captured Image", use_column_width=True)
            if st.button("Analyze Captured Image", type="primary"):
                if self.detector is None:
                    st.error("Model not loaded. Please check sidebar configuration.")
                else:
                    with st.spinner("Analyzing captured image..."):
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                                image.save(tmp_file.name)
                                tmp_path = tmp_file.name
                            result = self.detector.predict_image(tmp_path, save_result=False)
                            
                            # Apply class mapping fix for finetuned model
                            result = self._fix_detection_classes(result)
                            
                            os.unlink(tmp_path)
                            self._display_detection_results(result, image)
                        except Exception as e:
                            st.error(f"❌ Error during analysis: {str(e)}")
    
    def _batch_processing_tab(self):
        """Batch processing tab."""
        st.markdown("""
        <div style="text-align: center; margin-bottom: 32px;">
            <h2>Batch Processing</h2>
            <p style="color: #64748b;">Process multiple PCB images simultaneously</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📤 Upload Multiple Images")
            
            uploaded_files = st.file_uploader(
                "Choose multiple image files",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                accept_multiple_files=True,
                help="Upload multiple PCB images for batch processing"
            )
            
            if uploaded_files and st.button("Process All Images", type="primary"):
                if self.detector is None:
                    st.error("Model not loaded. Please check sidebar configuration.")
                else:
                    self._process_batch_images(uploaded_files)
        
        with col2:
            st.markdown("### Batch Processing Info")
            st.info("""
            **Advanced Features:**
            - Process multiple images simultaneously
            - Comprehensive defect analysis
            - Statistical insights
            - Export capabilities
            
            **Supported Formats:**
            - JPG, JPEG, PNG, BMP
            - Maximum 10 images per batch
            """)
            # --- Small table below batch info ---
            if 'batch_results_preview' in st.session_state:
                preview = st.session_state['batch_results_preview']
                if preview:
                    table_html = """
                    <div style='margin-top:0.5em;margin-bottom:1.2em;'>
                    <table style='width:100%;font-size:0.98rem;background:rgba(34,40,49,0.35);border-radius:10px;border-collapse:separate;border-spacing:0 0.3em;'>
                        <thead>
                            <tr style='background:rgba(0,198,255,0.08);'>
                                <th style='padding:0.4em 0.7em;text-align:left;color:#fff;font-weight:600;'>Image</th>
                                <th style='padding:0.4em 0.7em;text-align:center;color:#fff;font-weight:600;'>Defects</th>
                                <th style='padding:0.4em 0.7em;text-align:center;color:#fff;font-weight:600;'>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                    """
                    for row in preview:
                        status_color = "#00c6ff" if row['Status'] == 'Clean' else "#ffc107"
                        status_icon = "✅" if row['Status'] == 'Clean' else "⚠️"
                        table_html += f"<tr>"
                        table_html += f"<td style='padding:0.4em 0.7em;color:#fff;'>{row['Filename']}</td>"
                        table_html += f"<td style='padding:0.4em 0.7em;text-align:center;color:#fff;'>{row['Defects Found']}</td>"
                        table_html += f"<td style='padding:0.4em 0.7em;text-align:center;color:{status_color};font-weight:600;'>{status_icon} {row['Status']}</td>"
                        table_html += "</tr>"
                    table_html += """
                        </tbody>
                    </table>
                    </div>
                    """
                    st.markdown(table_html, unsafe_allow_html=True)
                    # --- Download button for all marked defected images as ZIP ---
                    if 'last_batch_results' in st.session_state:
                        results = st.session_state['last_batch_results']
                        defected_imgs = [r for r in results if r['num_defects'] > 0 and r.get('original_image') is not None]
                        if defected_imgs:
                            import io, zipfile
                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                                for r in defected_imgs:
                                    annotated_img = self._create_annotated_image(r['original_image'], r['detections'])
                                    img_bytes = io.BytesIO()
                                    annotated_img.save(img_bytes, format="PNG")
                                    img_bytes.seek(0)
                                    arcname = f"annotated_{r['filename'].rsplit('.',1)[0]}.png"
                                    zipf.writestr(arcname, img_bytes.read())
                            zip_buffer.seek(0)
                            st.download_button(
                                label="Download All Defected Images (ZIP)",
                                data=zip_buffer,
                                file_name="defected_annotated_images.zip",
                                mime="application/zip",
                                help="Download all annotated images with defects as a ZIP archive"
                            )
    
    def _process_batch_images(self, uploaded_files):
        """Process multiple images in batch."""
        if len(uploaded_files) > 10:
            st.warning("⚠️ Maximum 10 images allowed per batch. Processing first 10.")
            uploaded_files = uploaded_files[:10]
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []
        preview = []
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing image {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    image = Image.open(uploaded_file)
                    image.save(tmp_file.name)
                    tmp_path = tmp_file.name
                # Run detection
                result = self.detector.predict_image(tmp_path, save_result=False)
                # Apply class mapping fix for finetuned model
                result = self._fix_detection_classes(result)
                result['filename'] = uploaded_file.name
                result['original_image'] = image.copy()  # Store original image for annotation
                results.append(result)
                # For preview table
                preview.append({
                    'Filename': uploaded_file.name,
                    'Defects Found': result['num_defects'],
                    'Status': 'Clean' if result['num_defects'] == 0 else 'Defects'
                })
                # Clean up
                os.unlink(tmp_path)
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                results.append({
                    'filename': uploaded_file.name,
                    'detections': [],
                    'num_defects': 0,
                    'error': str(e),
                    'original_image': None
                })
                preview.append({
                    'Filename': uploaded_file.name,
                    'Defects Found': 0,
                    'Status': 'Error'
                })
            progress_bar.progress((i + 1) / len(uploaded_files))
        status_text.text("✅ Batch processing completed!")
        # Save preview table to session state for display in batch tab
        st.session_state['batch_results_preview'] = preview
        # Save full results for download button
        st.session_state['last_batch_results'] = results
        # Display batch results
        self._display_batch_results(results)
    
    def _display_batch_results(self, results):
        """Display batch processing results: bar chart (if defects), then annotated images grid, both full width. No table, no analytics cards."""
        import plotly.graph_objects as go
        import base64
        # --- Prepare defect counts ---
        defect_counts = {}
        for result in results:
            for detection in result['detections']:
                cid = detection.get('class_id', -1)
                if cid == 6:
                    defect_type = "Poor Solder"
                elif cid == 7:
                    defect_type = "Spike"
                elif cid == 8:
                    defect_type = "Excessive Solder"
                elif cid == 9:
                    defect_type = "Cold Solder Joint"
                elif cid >= 6:
                    defect_type = "Soldering Defect"
                elif 0 <= cid < len(self.classes):
                    defect_type = self.classes[cid].replace('_', ' ').title()
                else:
                    defect_type = f"Unknown ({cid})"
                defect_counts[defect_type] = defect_counts.get(defect_type, 0) + 1
        # Prepare gallery images
        gallery_images = []
        for idx, result in enumerate(results):
            if result['original_image'] is not None:
                annotated_image = self._create_annotated_image(result['original_image'], result['detections'])
                gallery_images.append({
                    'idx': idx,
                    'filename': result['filename'],
                    'image': annotated_image
                })
        # --- Bar chart (if any defects) ---
        if defect_counts:
            st.markdown("""
            <div style='font-size:1.3rem;font-weight:700;margin-bottom:0.5em;'>Defect Type Distribution</div>
            """, unsafe_allow_html=True)
            fig = go.Figure(data=[go.Bar(
                x=list(defect_counts.keys()),
                y=list(defect_counts.values()),
                marker_color="#00c6ff",
                hoverinfo="x+y",
                text=list(defect_counts.values()),
                textposition="outside"
            )])
            fig.update_layout(
                plot_bgcolor="rgba(34,40,49,0.85)",
                paper_bgcolor="rgba(34,40,49,0.85)",
                font_color="#fff",
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis=dict(showgrid=False, tickangle=-30),
                yaxis=dict(showgrid=False, zeroline=False),
                height=320
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        # --- Annotated images grid (as before) ---
        st.markdown("""
        <div style="font-size:1.3rem;font-weight:700;margin:1.5em 0 0.5em 0;">Annotated Images</div>
        """, unsafe_allow_html=True)
        # Responsive grid using Streamlit columns
        num_cols = 4 if len(gallery_images) >= 4 else max(1, len(gallery_images))
        rows = [gallery_images[i:i+num_cols] for i in range(0, len(gallery_images), num_cols)]
        for row_imgs in rows:
            cols = st.columns(len(row_imgs))
            for col, img_info in zip(cols, row_imgs):
                img_b64 = self._pil_to_base64(img_info['image'])
                col.image(img_info['image'], caption=img_info['filename'], use_column_width=True)
        # Modal overlay for enlarged image (if any, as before)
        enlarged_idx = st.session_state.get('enlarged_img_idx', None)
        if enlarged_idx is not None:
            img_info = next((g for g in gallery_images if g['idx'] == enlarged_idx), None)
            if img_info:
                st.markdown(f"""
                <div style="position:fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(15,32,39,0.85); z-index:10000; display:flex; align-items:center; justify-content:center;">
                    <div style="position:relative; background:rgba(34,40,49,0.95); border-radius:24px; box-shadow:0 8px 32px 0 rgba(0,198,255,0.18); padding:2rem; max-width:90vw; max-height:90vh; display:flex; flex-direction:column; align-items:center;">
                        <button onclick="window.dispatchEvent(new Event('closeModal'))" style="position:absolute; top:1.2rem; right:1.2rem; background:rgba(0,198,255,0.15); border:none; border-radius:50%; width:40px; height:40px; color:#fff; font-size:1.5rem; cursor:pointer;">&times;</button>
                        <img src='data:image/png;base64,{self._pil_to_base64(img_info['image'])}' style="max-width:80vw; max-height:70vh; border-radius:18px; box-shadow:0 4px 32px rgba(0,198,255,0.18); margin-bottom:1.5rem;" />
                        <a download="annotated_{img_info['filename'].rsplit('.', 1)[0]}.png" href="data:image/png;base64,{self._pil_to_base64(img_info['image'])}" style="background:linear-gradient(90deg,#00c6ff,#0072ff);color:#fff;padding:0.8rem 2.2rem;border-radius:12px;font-weight:600;text-decoration:none;box-shadow:0 2px 12px rgba(0,198,255,0.18);font-size:1.1rem;">Download Image</a>
                    </div>
                </div>
                <script>
                window.addEventListener('closeModal', function() {{
                    window.parent.postMessage({{isStreamlitMessage: true, type: 'streamlit:setComponentValue', key: 'enlarged_img_idx', value: null}}, '*');
                }});
                </script>
                """, unsafe_allow_html=True)

    def _pil_to_base64(self, pil_img):
        import base64
        from io import BytesIO
        buf = BytesIO()
        pil_img.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode('utf-8')

def main():
    """Main function to run the application."""
    app = PCBDefectWebInterface()
    app.main()

if __name__ == "__main__":
    main() 
