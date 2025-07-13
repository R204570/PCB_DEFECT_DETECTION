import torch
import torch.nn as nn
from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import yaml
from pathlib import Path
import supervision as sv
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

class PCBDefectDetector:
    """PCB Defect Detection Model using YOLOv8. - Raj Patel"""
    
    def __init__(self, config_path: str = "config.yaml", model_path: Optional[str] = None):
        """Initialize the PCB defect detector. - Raj Patel"""
        # Try to load class names from data.yaml if available
        data_yaml_path = "output/yolo_format/data.yaml"
        if os.path.exists(data_yaml_path):
            with open(data_yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            self.classes = data['names']
        else:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            self.classes = self.config['classes']
        
        self.class_to_id = {cls: idx for idx, cls in enumerate(self.classes)}
        self.id_to_class = {idx: cls for idx, cls in enumerate(self.classes)}
        
        # Load config for other settings
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.confidence_threshold = self.config['model']['confidence_threshold']
        self.nms_threshold = self.config['model']['nms_threshold']
        self.input_size = self.config['model']['input_size']
        
        # Initialize model
        if model_path and Path(model_path).exists():
            print(f"Loading custom model: {model_path}")
            self.model = YOLO(model_path)
        else:
            print("Loading default YOLOv8 model")
            self.model = YOLO('yolov8n.pt')  # Start with nano model
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize supervision annotators
        self.box_annotator = sv.BoxAnnotator(
        )
    
    def train(self, data_yaml_path: str, epochs: Optional[int] = None):
        """Train the model on PCB defect dataset. - Raj Patel"""
        if epochs is None:
            epochs = self.config['training']['epochs']
        
        print(f"Starting training for {epochs} epochs...")
        
        # Train the model
        results = self.model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=self.input_size[0],
            batch=self.config['data']['batch_size'],
            device=self.device,
            patience=20,
            save=True,
            project=self.config['data']['output_path'],
            name='pcb_defect_detection'
        )
        
        print("Training completed!")
        return results
    
    def predict_image(self, image_path: str, save_result: bool = True) -> Dict:
        """Predict defects in a single image with robust error handling and validation. - Raj Patel"""
        try:
            # Validate image path
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            if not image_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                raise ValueError(f"Unsupported image format: {image_path}")

            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image (may be corrupted or unreadable): {image_path}")

            # Check image size
            if image.shape[0] < 10 or image.shape[1] < 10:
                raise ValueError(f"Image is too small or invalid: {image_path}")

            # Run prediction
            try:
                results = self.model(image, conf=self.confidence_threshold, iou=self.nms_threshold)
            except Exception as e:
                print(f"[ERROR] Model inference failed: {e}")
                raise RuntimeError(f"Model inference failed: {e}")

            # Process results
            detections = []
            for result in results:
                boxes = getattr(result, 'boxes', None)
                if boxes is not None:
                    for box in boxes:
                        try:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0].cpu().numpy())
                            class_id = int(box.cls[0].cpu().numpy())
                            detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': confidence,
                                'class_id': class_id,
                                'class_name': self.id_to_class.get(class_id, str(class_id))
                            })
                        except Exception as box_e:
                            print(f"[WARNING] Skipping invalid detection box: {box_e}")

            # Save result if requested
            if save_result:
                try:
                    self._save_detection_result(image, detections, image_path)
                except Exception as save_e:
                    print(f"[WARNING] Could not save detection result: {save_e}")

            return {
                'image_path': image_path,
                'detections': detections,
                'num_defects': len(detections)
            }
        except Exception as e:
            print(f"[DETECTION ERROR] {e}")
            return {
                'image_path': image_path,
                'detections': [],
                'num_defects': 0,
                'error': str(e)
            }
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """Predict defects in multiple images. - Raj Patel"""
        results = []
        for image_path in image_paths:
            try:
                result = self.predict_image(image_path, save_result=False)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                results.append({
                    'image_path': image_path,
                    'detections': [],
                    'num_defects': 0,
                    'error': str(e)
                })
        
        return results
    
    def live_detection(self, camera_id: int = 0, save_video: bool = False):
        """Perform live defect detection using webcam. - Raj Patel"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['live_detection']['frame_width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['live_detection']['frame_height'])
        cap.set(cv2.CAP_PROP_FPS, self.config['live_detection']['fps'])
        
        # Video writer for saving
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                self.config['live_detection']['output_path'],
                fourcc,
                self.config['live_detection']['fps'],
                (self.config['live_detection']['frame_width'], 
                 self.config['live_detection']['frame_height'])
            )
        
        print("Starting live detection... Press 'q' to quit.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run prediction
                results = self.model(frame, conf=self.confidence_threshold, iou=self.nms_threshold)
                
                # Process and display results
                annotated_frame = self._annotate_frame(frame, results[0])
                
                # Display frame
                cv2.imshow('PCB Defect Detection', annotated_frame)
                
                # Save frame if requested
                if save_video and video_writer:
                    video_writer.write(annotated_frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
    
    def _annotate_frame(self, frame: np.ndarray, result) -> np.ndarray:
        """Annotate frame with detection results. - Raj Patel"""
        # Convert BGR to RGB for supervision
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract detections
        detections = sv.Detections.from_ultralytics(result)
        
        # Create labels
        labels = []
        for confidence, class_id in zip(detections.confidence, detections.class_id):
            class_name = self.id_to_class[int(class_id)]
            labels.append(f"{class_name} {confidence:.2f}")
        
        # Annotate frame
        annotated_frame = self.box_annotator.annotate(
            scene=frame_rgb,
            detections=detections,
            labels=labels
        )
        
        # Convert back to BGR
        return cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    
    def _save_detection_result(self, image: np.ndarray, detections: List[Dict], 
                             original_path: str):
        """Save detection result with annotations. - Raj Patel"""
        # Create output directory
        output_dir = Path(self.config['data']['output_path']) / 'predictions'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figure
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Add bounding boxes
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Create rectangle
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            ax.text(x1, y1 - 10, f"{class_name} {confidence:.2f}",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                   fontsize=10, color='white')
        
        ax.set_title(f"PCB Defect Detection - {Path(original_path).name}")
        ax.axis('off')
        
        # Save result
        output_path = output_dir / f"result_{Path(original_path).stem}.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Detection result saved to: {output_path}")
    
    def evaluate(self, test_data_yaml: str) -> Dict:
        """Evaluate model performance on test dataset. - Raj Patel"""
        print("Evaluating model...")
        
        # Run validation
        results = self.model.val(data=test_data_yaml)
        
        # Extract metrics
        metrics = {
            'mAP50': results.box.map50,
            'mAP50-95': results.box.map,
            'precision': results.box.mp,
            'recall': results.box.mr
        }
        
        print("Evaluation Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
    
    def export_model(self, format: str = 'onnx'):
        """Export model to different formats. - Raj Patel"""
        print(f"Exporting model to {format} format...")
        
        output_path = Path(self.config['data']['output_path']) / f"pcb_detector.{format}"
        self.model.export(format=format, file=output_path)
        
        print(f"Model exported to: {output_path}")
        return str(output_path)

class PCBDefectClassifier(nn.Module):
    """Additional classifier for defect type classification. - Raj Patel"""
    
    def __init__(self, num_classes: int, input_size: int = 224):
        super(PCBDefectClassifier, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    # Example usage
    detector = PCBDefectDetector()
    
    # Train model
    # detector.train("output/yolo_format/data.yaml")
    
    # Test prediction
    # result = detector.predict_image("path/to/test/image.jpg")
    # print(f"Found {result['num_defects']} defects") 