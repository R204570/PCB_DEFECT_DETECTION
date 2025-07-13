#!/usr/bin/env python3
"""
PCB Defect Detection System
Main script for training, evaluation, and inference

Developed by: Raj Patel
LinkedIn: https://www.linkedin.com/in/raj-patel5
"""

import argparse
import sys
from pathlib import Path
import yaml
from src.data_preparation import PCBDataPreparator
from src.model import PCBDefectDetector

def main():
    """Main function with command-line interface. - Raj Patel"""
    parser = argparse.ArgumentParser(
        description="PCB Defect Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare data from all datasets
  python main.py prepare-data
  
  # Train model
  python main.py train --epochs 100
  
  # Predict on single image
  python main.py predict --image path/to/image.jpg
  
  # Live detection
  python main.py live
  
  # Evaluate model
  python main.py evaluate
  
  # Web interface
  python main.py web
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Prepare data command - Raj Patel
    prep_parser = subparsers.add_parser('prepare-data', help='Prepare and combine all datasets')
    prep_parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    
    # Train command - Raj Patel
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data-yaml', default='output/yolo_format/data.yaml', 
                             help='Path to data.yaml file')
    train_parser.add_argument('--epochs', type=int, default=None, 
                             help='Number of training epochs')
    train_parser.add_argument('--model-path', default=None, 
                             help='Path to pretrained model')
    train_parser.add_argument('--config', default='config.yaml', 
                             help='Configuration file path')
    
    # Predict command - Raj Patel
    predict_parser = subparsers.add_parser('predict', help='Predict defects in image')
    predict_parser.add_argument('--image', required=True, help='Path to input image')
    predict_parser.add_argument('--model-path', default=None, 
                               help='Path to trained model')
    predict_parser.add_argument('--config', default='config.yaml', 
                               help='Configuration file path')
    predict_parser.add_argument('--no-save', action='store_true', 
                               help='Do not save result image')
    
    # Live detection command - Raj Patel
    live_parser = subparsers.add_parser('live', help='Live defect detection')
    live_parser.add_argument('--camera', type=int, default=0, 
                            help='Camera device ID')
    live_parser.add_argument('--model-path', default=None, 
                            help='Path to trained model')
    live_parser.add_argument('--save-video', action='store_true', 
                            help='Save detection video')
    live_parser.add_argument('--config', default='config.yaml', 
                            help='Configuration file path')
    
    # Evaluate command - Raj Patel
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    eval_parser.add_argument('--data-yaml', default='output/yolo_format/data.yaml', 
                            help='Path to data.yaml file')
    eval_parser.add_argument('--model-path', default=None, 
                            help='Path to trained model')
    eval_parser.add_argument('--config', default='config.yaml', 
                            help='Configuration file path')
    
    # Web interface command - Raj Patel
    web_parser = subparsers.add_parser('web', help='Launch web interface')
    web_parser.add_argument('--port', type=int, default=8501, 
                           help='Port for web interface')
    web_parser.add_argument('--host', default='localhost', 
                           help='Host for web interface')
    
    # Export command - Raj Patel
    export_parser = subparsers.add_parser('export', help='Export model to different format')
    export_parser.add_argument('--model-path', required=True, 
                              help='Path to trained model')
    export_parser.add_argument('--format', default='onnx', 
                              choices=['onnx', 'torchscript', 'tflite'], 
                              help='Export format')
    export_parser.add_argument('--config', default='config.yaml', 
                              help='Configuration file path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'prepare-data':
            prepare_data(args)
        elif args.command == 'train':
            train_model(args)
        elif args.command == 'predict':
            predict_image(args)
        elif args.command == 'live':
            live_detection(args)
        elif args.command == 'evaluate':
            evaluate_model(args)
        elif args.command == 'web':
            launch_web_interface(args)
        elif args.command == 'export':
            export_model(args)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

def prepare_data(args):
    """Prepare and combine all datasets. - Raj Patel"""
    print("🔧 Preparing PCB defect datasets...")
    
    preparator = PCBDataPreparator(args.config)
    preparator.prepare_all_datasets()
    
    print("✅ Data preparation completed!")
    print(f"📁 Output directory: {preparator.yolo_dir}")
    print(f"📄 Data config: {preparator.yolo_dir / 'data.yaml'}")

def train_model(args):
    """Train the PCB defect detection model. - Raj Patel"""
    print("🚀 Starting model training...")
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize detector
    detector = PCBDefectDetector(args.config, args.model_path)
    
    # Check if data.yaml exists
    data_yaml_path = Path(args.data_yaml)
    if not data_yaml_path.exists():
        print(f"❌ Data file not found: {data_yaml_path}")
        print("Please run 'python main.py prepare-data' first.")
        return
    
    # Start training
    results = detector.train(args.data_yaml, args.epochs)
    
    print("✅ Training completed!")
    print(f"📁 Model saved to: {config['data']['output_path']}/pcb_defect_detection")

def predict_image(args):
    """Predict defects in a single image. - Raj Patel"""
    print(f"🔍 Predicting defects in: {args.image}")
    
    # Initialize detector
    detector = PCBDefectDetector(args.config, args.model_path)
    
    # Check if image exists
    if not Path(args.image).exists():
        print(f"❌ Image not found: {args.image}")
        return
    
    # Run prediction
    result = detector.predict_image(args.image, save_result=not args.no_save)
    
    # Display results
    print(f"📊 Detection Results:")
    print(f"   Image: {result['image_path']}")
    print(f"   Defects found: {result['num_defects']}")
    
    if result['num_defects'] > 0:
        print("   Defect details:")
        for i, detection in enumerate(result['detections']):
            print(f"     {i+1}. {detection['class_name']} (confidence: {detection['confidence']:.3f})")
    else:
        print("   ✅ No defects detected!")

def live_detection(args):
    """Perform live defect detection. - Raj Patel"""
    print("📹 Starting live detection...")
    
    # Initialize detector
    detector = PCBDefectDetector(args.config, args.model_path)
    
    # Start live detection
    detector.live_detection(args.camera, args.save_video)

def evaluate_model(args):
    """Evaluate model performance. - Raj Patel"""
    print("📈 Evaluating model performance...")
    
    # Initialize detector
    detector = PCBDefectDetector(args.config, args.model_path)
    
    # Check if data.yaml exists
    data_yaml_path = Path(args.data_yaml)
    if not data_yaml_path.exists():
        print(f"❌ Data file not found: {data_yaml_path}")
        print("Please run 'python main.py prepare-data' first.")
        return
    
    # Run evaluation
    metrics = detector.evaluate(args.data_yaml)
    
    print("✅ Evaluation completed!")

def launch_web_interface(args):
    """Launch the web interface. - Raj Patel"""
    print(f"🌐 Launching web interface at http://{args.host}:{args.port}")
    
    try:
        import streamlit.web.cli as stcli
        import sys
        
        # Set up streamlit arguments
        sys.argv = [
            "streamlit", "run", "src/web_interface.py",
            "--server.port", str(args.port),
            "--server.address", args.host
        ]
        
        # Launch streamlit
        stcli.main()
        
    except ImportError:
        print("❌ Streamlit not installed. Please install with: pip install streamlit")
    except Exception as e:
        print(f"❌ Error launching web interface: {str(e)}")

def export_model(args):
    """Export model to different format. - Raj Patel"""
    print(f"📦 Exporting model to {args.format} format...")
    
    # Initialize detector
    detector = PCBDefectDetector(args.config, args.model_path)
    
    # Export model
    output_path = detector.export_model(args.format)
    
    print(f"✅ Model exported to: {output_path}")

if __name__ == "__main__":
    main() 