import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import json
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import shutil
from tqdm import tqdm
import random

class PCBDataPreparator:
    """Comprehensive data preparation for PCB defect detection datasets."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the data preparator with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.classes = self.config['classes']
        self.class_to_id = {cls: idx for idx, cls in enumerate(self.classes)}
        self.id_to_class = {idx: cls for idx, cls in enumerate(self.classes)}
        
        self.output_dir = Path(self.config['data']['output_path'])
        self.output_dir.mkdir(exist_ok=True)
        
        # Create YOLO format directories
        self.yolo_dir = self.output_dir / "yolo_format"
        self.yolo_dir.mkdir(exist_ok=True)
        
        for split in ['train', 'val', 'test']:
            (self.yolo_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.yolo_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    def prepare_all_datasets(self):
        """Prepare all available datasets and combine them."""
        print("Starting comprehensive data preparation...")
        
        data_path = Path(self.config['data']['train_path'])
        all_images = []
        all_annotations = []
        
        # Process each dataset
        datasets = [
            ('pcb-defects-akhatova', self._process_akhatova_dataset),
            ('pcb-aoi', self._process_pcb_aoi_dataset),
            ('deeppcb', self._process_deeppcb_dataset),
            ('DeepPCB-master', self._process_deeppcb_dataset),
            ('deep-pcb-datasets', self._process_deeppcb_dataset),
            ('tiny-defect-pcb', self._process_tiny_defect_dataset),
            ('soldef-ai-pcb', self._process_soldef_dataset),
            ('pcb-defect-yolo', self._process_pcb_defect_yolo_dataset)
        ]
        
        for dataset_name, processor in datasets:
            dataset_path = data_path / dataset_name
            if dataset_path.exists():
                print(f"\nProcessing {dataset_name} dataset...")
                try:
                    images, annotations = processor(dataset_path)
                    all_images.extend(images)
                    all_annotations.extend(annotations)
                    print(f"✓ {dataset_name}: {len(images)} images processed")
                except Exception as e:
                    print(f"✗ Error processing {dataset_name}: {str(e)}")
        
        # Split data and save in YOLO format
        self._split_and_save_data(all_images, all_annotations)
        
        # Create data.yaml for YOLO training
        self._create_data_yaml()
        
        print(f"\nData preparation completed!")
        print(f"Total images: {len(all_images)}")
        print(f"Output directory: {self.yolo_dir}")
    
    def _process_akhatova_dataset(self, dataset_path: Path) -> Tuple[List[str], List[List]]:
        """Process Akhatova PCB defects dataset."""
        images = []
        annotations = []
        
        # Process images and annotations
        for defect_type in self.classes:
            defect_path = dataset_path / "images" / defect_type
            if defect_path.exists():
                for img_file in defect_path.glob("*.jpg"):
                    img_path = str(img_file)
                    images.append(img_path)
                    
                    # Create annotation (center point for classification)
                    img = cv2.imread(img_path)
                    if img is not None:
                        h, w = img.shape[:2]
                        # Place annotation in center of image
                        x_center = w / 2
                        y_center = h / 2
                        class_id = self.class_to_id[defect_type]
                        
                        # Convert to YOLO format (normalized)
                        x_norm = x_center / w
                        y_norm = y_center / h
                        w_norm = 0.1  # Small bounding box
                        h_norm = 0.1
                        
                        annotations.append([[class_id, x_norm, y_norm, w_norm, h_norm]])
        
        return images, annotations
    
    def _process_pcb_aoi_dataset(self, dataset_path: Path) -> Tuple[List[str], List[List]]:
        """Process PCB-AOI dataset with XML annotations."""
        images = []
        annotations = []
        
        # Process training data
        train_annotations = dataset_path / "train_data" / "Annotations"
        train_images = dataset_path / "train_data" / "JPEGImages"
        
        if train_annotations.exists() and train_images.exists():
            for xml_file in train_annotations.glob("*.xml"):
                img_name = xml_file.stem + ".jpeg"
                img_path = train_images / img_name
                
                if img_path.exists():
                    # Parse XML annotation
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                    
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        h, w = img.shape[:2]
                        images.append(str(img_path))
                        
                        img_annotations = []
                        for obj in root.findall('object'):
                            name = obj.find('name').text.lower()
                            if name in self.class_to_id:
                                bbox = obj.find('bndbox')
                                xmin = float(bbox.find('xmin').text)
                                ymin = float(bbox.find('ymin').text)
                                xmax = float(bbox.find('xmax').text)
                                ymax = float(bbox.find('ymax').text)
                                
                                # Convert to YOLO format
                                x_center = (xmin + xmax) / 2
                                y_center = (ymin + ymax) / 2
                                width = xmax - xmin
                                height = ymax - ymin
                                
                                x_norm = x_center / w
                                y_norm = y_center / h
                                w_norm = width / w
                                h_norm = height / h
                                
                                class_id = self.class_to_id[name]
                                img_annotations.append([class_id, x_norm, y_norm, w_norm, h_norm])
                        
                        annotations.append(img_annotations)
        
        return images, annotations
    
    def _process_deeppcb_dataset(self, dataset_path: Path) -> Tuple[List[str], List[List]]:
        """Process DeepPCB dataset with text annotations from all group folders, using _not folders for annotations."""
        images = []
        annotations = []
        pcb_data = dataset_path / "PCBData"
        if pcb_data.exists():
            for group_dir in pcb_data.iterdir():
                if group_dir.is_dir():
                    for pcb_dir in group_dir.iterdir():
                        if pcb_dir.is_dir() and not pcb_dir.name.endswith('_not'):
                            # Find the corresponding _not folder for annotations
                            not_dir = pcb_dir.parent / (pcb_dir.name + '_not')
                            for img_file in pcb_dir.glob("*_test.jpg"):
                                txt_file = not_dir / (img_file.stem.replace('_test', '') + '.txt')
                                if txt_file.exists():
                                    img = cv2.imread(str(img_file))
                                    if img is not None:
                                        h, w = img.shape[:2]
                                        images.append(str(img_file))
                                        img_annotations = []
                                        with open(txt_file, 'r') as f:
                                            for line in f:
                                                parts = line.strip().split(',')
                                                if len(parts) >= 5:
                                                    x1, y1, x2, y2, defect_type = parts[:5]
                                                    defect_mapping = {
                                                        '1': 'open_circuit',
                                                        '2': 'short',
                                                        '3': 'mouse_bite',
                                                        '4': 'spur',
                                                        '5': 'spurious_copper',
                                                        '6': 'pin_hole'
                                                    }
                                                    if defect_type in defect_mapping:
                                                        mapped_class = defect_mapping[defect_type]
                                                        if mapped_class in self.class_to_id:
                                                            x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])
                                                            x_center = (x1 + x2) / 2
                                                            y_center = (y1 + y2) / 2
                                                            width = x2 - x1
                                                            height = y2 - y1
                                                            x_norm = x_center / w
                                                            y_norm = y_center / h
                                                            w_norm = width / w
                                                            h_norm = height / h
                                                            class_id = self.class_to_id[mapped_class]
                                                            img_annotations.append([class_id, x_norm, y_norm, w_norm, h_norm])
                                        annotations.append(img_annotations)
        return images, annotations
    
    def _process_tiny_defect_dataset(self, dataset_path: Path) -> Tuple[List[str], List[List]]:
        """Process Tiny Defect PCB dataset from all relevant subfolders and XMLs."""
        images = []
        annotations = []
        # Gather all images from demos_backup, inference_results, and root
        image_dirs = [
            dataset_path / "tools" / "demos_backup",
            dataset_path / "tools" / "inference_results",
            dataset_path / "tools" / "demos",
            dataset_path
        ]
        # Gather all XML annotation files
        xml_dir = dataset_path / "tools" / "test_annotation"
        xml_map = {}
        if xml_dir.exists():
            for xml_file in xml_dir.glob("*.xml"):
                xml_map[xml_file.stem] = xml_file
        for img_dir in image_dirs:
            if img_dir.exists():
                for img_file in img_dir.glob("*.jpg"):
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        h, w = img.shape[:2]
                        images.append(str(img_file))
                        img_annotations = []
                        # Try to find matching XML annotation
                        xml_file = xml_map.get(img_file.stem)
                        if xml_file:
                            try:
                                tree = ET.parse(xml_file)
                                root = tree.getroot()
                                for obj in root.findall('object'):
                                    name = obj.find('name').text.lower()
                                    if name in self.class_to_id:
                                        bbox = obj.find('bndbox')
                                        xmin = float(bbox.find('xmin').text)
                                        ymin = float(bbox.find('ymin').text)
                                        xmax = float(bbox.find('xmax').text)
                                        ymax = float(bbox.find('ymax').text)
                                        x_center = (xmin + xmax) / 2
                                        y_center = (ymin + ymax) / 2
                                        width = xmax - xmin
                                        height = ymax - ymin
                                        x_norm = x_center / w
                                        y_norm = y_center / h
                                        w_norm = width / w
                                        h_norm = height / h
                                        class_id = self.class_to_id[name]
                                        img_annotations.append([class_id, x_norm, y_norm, w_norm, h_norm])
                            except Exception as e:
                                pass
                        annotations.append(img_annotations)
        return images, annotations
    
    def _process_soldef_dataset(self, dataset_path: Path) -> Tuple[List[str], List[List]]:
        """Process SolDef AI PCB dataset with JSON annotations."""
        images = []
        annotations = []
        
        labeled_dir = dataset_path / "SolDef_AI" / "Labeled"
        if labeled_dir.exists():
            for json_file in labeled_dir.glob("*.json"):
                img_name = json_file.stem + ".jpg"
                img_path = labeled_dir / img_name
                
                if img_path.exists():
                    # Parse JSON annotation
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        h, w = img.shape[:2]
                        images.append(str(img_path))
                        
                        img_annotations = []
                        if 'shapes' in data:
                            for shape in data['shapes']:
                                label = shape['label'].lower()
                                if label in self.class_to_id:
                                    points = shape['points']
                                    if len(points) >= 2:
                                        # Calculate bounding box from points
                                        x_coords = [p[0] for p in points]
                                        y_coords = [p[1] for p in points]
                                        
                                        xmin, xmax = min(x_coords), max(x_coords)
                                        ymin, ymax = min(y_coords), max(y_coords)
                                        
                                        # Convert to YOLO format
                                        x_center = (xmin + xmax) / 2
                                        y_center = (ymin + ymax) / 2
                                        width = xmax - xmin
                                        height = ymax - ymin
                                        
                                        x_norm = x_center / w
                                        y_norm = y_center / h
                                        w_norm = width / w
                                        h_norm = height / h
                                        
                                        class_id = self.class_to_id[label]
                                        img_annotations.append([class_id, x_norm, y_norm, w_norm, h_norm])
                        
                        annotations.append(img_annotations)
        
        return images, annotations
    
    def _process_pcb_defect_yolo_dataset(self, dataset_path: Path) -> Tuple[List[str], List[List]]:
        """Process PCB Defect YOLO dataset."""
        images = []
        annotations = []
        
        dataset_dir = dataset_path / "pcb-defect-dataset"
        if dataset_dir.exists():
            # Process train, val, and test splits
            for split in ['train', 'val', 'test']:
                images_dir = dataset_dir / split / 'images'
                labels_dir = dataset_dir / split / 'labels'
                
                if images_dir.exists() and labels_dir.exists():
                    for img_file in images_dir.glob("*.jpg"):
                        img_name = img_file.stem
                        label_file = labels_dir / f"{img_name}.txt"
                        
                        if label_file.exists():
                            img = cv2.imread(str(img_file))
                            if img is not None:
                                h, w = img.shape[:2]
                                images.append(str(img_file))
                                
                                img_annotations = []
                                with open(label_file, 'r') as f:
                                    for line in f:
                                        parts = line.strip().split()
                                        if len(parts) >= 5:
                                            class_id = int(parts[0])
                                            x_center = float(parts[1])
                                            y_center = float(parts[2])
                                            width = float(parts[3])
                                            height = float(parts[4])
                                            
                                            # Map YOLO class IDs to our classes
                                            # Assuming the YOLO dataset uses similar class mapping
                                            if class_id < len(self.classes):
                                                img_annotations.append([class_id, x_center, y_center, width, height])
                                
                                annotations.append(img_annotations)
        
        return images, annotations
    
    def _split_and_save_data(self, images: List[str], annotations: List[List]):
        """Split data into train/val/test and save in YOLO format."""
        # Create pairs of images and annotations
        data_pairs = list(zip(images, annotations))
        random.shuffle(data_pairs)
        
        # Split data
        total = len(data_pairs)
        train_split = self.config['data']['train_split']
        val_split = self.config['data']['val_split']
        
        train_end = int(total * train_split)
        val_end = train_end + int(total * val_split)
        
        train_data = data_pairs[:train_end]
        val_data = data_pairs[train_end:val_end]
        test_data = data_pairs[val_end:]
        
        # Save data
        splits = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
        for split_name, split_data in splits.items():
            print(f"\nSaving {split_name} split ({len(split_data)} images)...")
            
            for idx, (img_path, img_annotations) in enumerate(tqdm(split_data)):
                # Copy image
                img_name = f"{split_name}_{idx:06d}.jpg"
                dst_img_path = self.yolo_dir / split_name / 'images' / img_name
                shutil.copy2(img_path, dst_img_path)
                
                # Save annotations
                label_name = f"{split_name}_{idx:06d}.txt"
                dst_label_path = self.yolo_dir / split_name / 'labels' / label_name
                
                with open(dst_label_path, 'w') as f:
                    for ann in img_annotations:
                        f.write(f"{' '.join(map(str, ann))}\n")
    
    def _create_data_yaml(self):
        """Create data.yaml file for YOLO training."""
        yaml_content = {
            'path': str(self.yolo_dir),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.classes),
            'names': self.classes
        }
        
        yaml_path = self.yolo_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        print(f"Created data.yaml at {yaml_path}")

if __name__ == "__main__":
    preparator = PCBDataPreparator()
    preparator.prepare_all_datasets() 