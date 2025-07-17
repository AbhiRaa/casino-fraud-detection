#!/usr/bin/env python3
"""
Generate Watchlist for Casino Fraud Detection

This script processes face images from a folder structure and generates
a watchlist JSON file with averaged embeddings for each person.
"""

# %%
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np
import insightface
from ultralytics import YOLO

# %%
class WatchlistGenerator:
    """Generate watchlist from face images."""
    
    def __init__(self):
        """Initialize the watchlist generator."""
        self.face_detector = None
        self.face_recognizer = None
        
    def load_models(self) -> None:
        """Load YOLOv8n and ArcFace models."""
        print("Loading YOLOv8n model...")
        try:
            self.face_detector = YOLO('yolov8n.pt')
        except Exception as e:
            print(f"Error loading YOLOv8n model: {e}")
            # Fallback to downloading from URL
            self.face_detector = YOLO('yolov8n')
        
        print("Loading ArcFace model...")
        self.face_recognizer = insightface.app.FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.face_recognizer.prepare(ctx_id=0, det_size=(640, 640))
    
    def extract_face_from_image(self, image_path: str) -> np.ndarray:
        """
        Extract face from image and return embedding.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Face embedding vector or None if no face found
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            return None
        
        # Detect faces using YOLOv8n (person detection)
        results = self.face_detector(image, verbose=False)
        faces = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Filter for person class (class 0 in COCO)
                    if int(box.cls[0]) == 0:  # Person class
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        w, h = x2 - x1, y2 - y1
                        faces.append((int(x1), int(y1), int(w), int(h)))
        
        if not faces:
            print(f"Warning: No face detected in {image_path}")
            return None
        
        # Use the largest face (assuming it's the main subject)
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        # Ensure valid crop region
        if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
            print(f"Warning: Invalid face crop region in {image_path}")
            return None
        
        face_crop = image[y:y+h, x:x+w]
        
        # Extract embedding using InsightFace
        faces = self.face_recognizer.get(face_crop)
        
        if faces:
            return faces[0].embedding
        else:
            print(f"Warning: Could not extract embedding from {image_path}")
            return None
    
    def process_person_folder(self, person_folder: Path) -> np.ndarray:
        """
        Process all images in a person's folder and return averaged embedding.
        
        Args:
            person_folder: Path to person's image folder
            
        Returns:
            Averaged embedding vector or None if no valid embeddings
        """
        person_name = person_folder.name
        print(f"Processing {person_name}...")
        
        # Supported image extensions
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Find all image files
        image_files = []
        for ext in supported_extensions:
            image_files.extend(person_folder.glob(f'*{ext}'))
            image_files.extend(person_folder.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"Warning: No image files found in {person_folder}")
            return None
        
        print(f"Found {len(image_files)} images for {person_name}")
        
        # Extract embeddings from all images
        embeddings = []
        for image_path in image_files:
            embedding = self.extract_face_from_image(str(image_path))
            if embedding is not None:
                embeddings.append(embedding)
        
        if not embeddings:
            print(f"Warning: No valid embeddings extracted for {person_name}")
            return None
        
        # Average the embeddings
        averaged_embedding = np.mean(embeddings, axis=0)
        print(f"Successfully processed {len(embeddings)} images for {person_name}")
        
        return averaged_embedding
    
    def generate_watchlist(self, img_root: str, output_path: str) -> None:
        """
        Generate watchlist from image folder structure.
        
        Args:
            img_root: Root directory containing person folders
            output_path: Path to output JSON file
        """
        img_root_path = Path(img_root)
        
        if not img_root_path.exists():
            raise FileNotFoundError(f"Image root directory {img_root} does not exist")
        
        # Load models
        self.load_models()
        
        # Process each person folder
        watchlist = {}
        
        for person_folder in img_root_path.iterdir():
            if person_folder.is_dir():
                embedding = self.process_person_folder(person_folder)
                if embedding is not None:
                    # Convert numpy array to list for JSON serialization
                    watchlist[person_folder.name] = embedding.tolist()
        
        if not watchlist:
            print("Warning: No valid embeddings generated. Watchlist will be empty.")
        
        # Save watchlist to JSON
        with open(output_path, 'w') as f:
            json.dump(watchlist, f, indent=2)
        
        print(f"Watchlist saved to {output_path}")
        print(f"Total persons in watchlist: {len(watchlist)}")
        
        # Print summary
        for name in watchlist.keys():
            print(f"  - {name}")

# %%
def create_sample_structure(sample_dir: str) -> None:
    """
    Create a sample folder structure for testing.
    
    Args:
        sample_dir: Directory to create sample structure in
    """
    sample_path = Path(sample_dir)
    sample_path.mkdir(exist_ok=True)
    
    # Create sample person folders
    persons = ["john_doe", "jane_smith", "bob_johnson"]
    
    for person in persons:
        person_dir = sample_path / person
        person_dir.mkdir(exist_ok=True)
        
        # Create a README file with instructions
        readme_path = person_dir / "README.txt"
        with open(readme_path, 'w') as f:
            f.write(f"Add 3-5 clear face images of {person.replace('_', ' ').title()} here.\n")
            f.write("Supported formats: JPG, JPEG, PNG, BMP, TIFF\n")
            f.write("Example files: img1.jpg, img2.png, photo1.jpeg, etc.\n")
    
    print(f"Sample folder structure created in {sample_dir}")
    print("Add face images to each person's folder, then run:")
    print(f"python generate_watchlist.py --img_root {sample_dir} --out watchlist.json")

# %%
def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Generate watchlist for casino fraud detection')
    parser.add_argument('--img_root', type=str, default='./sample_faces',
                       help='Root directory containing person folders (default: ./sample_faces)')
    parser.add_argument('--out', type=str, default='watchlist.json',
                       help='Output JSON file path (default: watchlist.json)')
    parser.add_argument('--create_sample', action='store_true',
                       help='Create sample folder structure')
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_structure(args.img_root)
        return
    
    # Generate watchlist
    generator = WatchlistGenerator()
    generator.generate_watchlist(args.img_root, args.out)

# %%
if __name__ == "__main__":
    main()