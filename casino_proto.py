#!/usr/bin/env python3
"""
Casino Fraud Detection Prototype

A real-time face detection and recognition system for casino fraud detection.
Uses YOLOv8-face for detection and ArcFace for embeddings.
"""

# %%
import argparse
import json
import time
import csv
import os
import threading
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import insightface
from sklearn.metrics.pairwise import cosine_similarity
import pygame

# %%
class CasinoFraudDetector:
    """Main fraud detection system class."""
    
    def __init__(self, 
                 watchlist_path: str = "watchlist.json",
                 threshold: float = 0.55,
                 cam_id: int = 0,
                 show_fps: bool = False):
        """
        Initialize the fraud detection system.
        
        Args:
            watchlist_path: Path to the watchlist JSON file
            threshold: Similarity threshold for face matching
            cam_id: Camera device ID
            show_fps: Whether to show FPS instead of latency
        """
        self.watchlist_path = watchlist_path
        self.threshold = threshold
        self.cam_id = cam_id
        self.show_fps = show_fps
        
        # Initialize models
        self.face_detector = None
        self.face_recognizer = None
        self.watchlist = {}
        self.camera = None
        
        # Performance tracking
        self.frame_count = 0
        self.fps_start_time = time.time()
        
        # CSV logging
        self.csv_file = "detections.csv"
        self._init_csv()
        
        # Initialize pygame mixer for audio
        pygame.mixer.init()
        
    def _init_csv(self) -> None:
        """Initialize CSV file for logging detections."""
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'name', 'score', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'])
    
    def load_models(self) -> None:
        """Load YOLOv8-face and ArcFace models."""
        print("Loading YOLOv8n model...")
        # Use standard YOLOv8n model - can detect persons/faces
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
        
    def load_watchlist(self) -> None:
        """Load the watchlist from JSON file."""
        if os.path.exists(self.watchlist_path):
            with open(self.watchlist_path, 'r') as f:
                data = json.load(f)
                # Convert lists back to numpy arrays
                self.watchlist = {name: np.array(embedding) for name, embedding in data.items()}
            print(f"Loaded watchlist with {len(self.watchlist)} entries")
        else:
            print(f"Warning: Watchlist file {self.watchlist_path} not found")
            self.watchlist = {}
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in frame using YOLOv8n (person detection).
        
        Args:
            frame: Input frame
            
        Returns:
            List of bounding boxes (x, y, w, h)
        """
        results = self.face_detector(frame, verbose=False)
        faces = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Filter for person class (class 0 in COCO)
                    if int(box.cls[0]) == 0:  # Person class
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        w, h = x2 - x1, y2 - y1
                        faces.append((int(x1), int(y1), int(w), int(h)))
        
        return faces
    
    def get_face_embedding(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Extract face embedding using ArcFace.
        
        Args:
            frame: Input frame
            bbox: Face bounding box (x, y, w, h)
            
        Returns:
            512-d embedding vector or None if extraction fails
        """
        x, y, w, h = bbox
        
        # Ensure valid crop region
        if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
            return None
        
        face_crop = frame[y:y+h, x:x+w]
        
        # Use InsightFace for embedding extraction
        faces = self.face_recognizer.get(face_crop)
        
        if faces:
            # Return the embedding of the first (and likely only) face
            return faces[0].embedding
        
        return None
    
    def match_face(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Match face embedding against watchlist.
        
        Args:
            embedding: Face embedding vector
            
        Returns:
            Tuple of (matched_name, similarity_score)
        """
        if not self.watchlist:
            return None, 0.0
        
        best_match = None
        best_score = 0.0
        
        for name, watchlist_embedding in self.watchlist.items():
            similarity = cosine_similarity(
                embedding.reshape(1, -1),
                watchlist_embedding.reshape(1, -1)
            )[0][0]
            
            if similarity > best_score:
                best_score = similarity
                best_match = name
        
        return best_match, best_score
    
    def draw_bbox(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                  name: Optional[str], score: float) -> None:
        """
        Draw bounding box and label on frame.
        
        Args:
            frame: Input frame
            bbox: Bounding box coordinates
            name: Matched name (if any)
            score: Similarity score
        """
        x, y, w, h = bbox
        
        # Choose color based on match
        if name and score > self.threshold:
            color = (0, 0, 255)  # Red for matches
            label = f"{name}: {score:.3f}"
        else:
            color = (0, 255, 0)  # Green for unknown
            label = f"Unknown: {score:.3f}"
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                     (x + label_size[0], y), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def log_detection(self, name: str, score: float, bbox: Tuple[int, int, int, int]) -> None:
        """
        Log detection event to CSV file.
        
        Args:
            name: Matched name
            score: Similarity score
            bbox: Bounding box coordinates
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        x, y, w, h = bbox
        
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, name, score, x, y, w, h])
    
    def play_alert(self) -> None:
        """Play alert sound asynchronously."""
        def play_sound():
            try:
                if os.path.exists("siren-alert-96052.mp3"):
                    pygame.mixer.music.load("siren-alert-96052.mp3")
                    pygame.mixer.music.play()
                elif os.path.exists("alert.mp3"):
                    pygame.mixer.music.load("alert.mp3")
                    pygame.mixer.music.play()
                elif os.path.exists("alert.wav"):
                    pygame.mixer.music.load("alert.wav")
                    pygame.mixer.music.play()
                else:
                    # Fallback: system beep
                    print("\a")  # ASCII bell character
            except Exception as e:
                print(f"Alert sound failed: {e}")
        
        threading.Thread(target=play_sound, daemon=True).start()
    
    def calculate_fps(self) -> float:
        """Calculate current FPS."""
        self.frame_count += 1
        if self.frame_count % 30 == 0:  # Update every 30 frames
            current_time = time.time()
            fps = self.frame_count / (current_time - self.fps_start_time)
            self.fps_start_time = current_time
            self.frame_count = 0
            return fps
        return 0.0
    
    def run(self) -> None:
        """Main detection loop."""
        print("Starting fraud detection system...")
        
        # Load models and watchlist
        self.load_models()
        self.load_watchlist()
        
        # Initialize camera
        self.camera = cv2.VideoCapture(self.cam_id)
        if not self.camera.isOpened():
            raise RuntimeError(f"Cannot open camera {self.cam_id}")
        
        # Set camera resolution for better performance
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Press ESC to exit...")
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = self.camera.read()
                if not ret:
                    break
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                # Process each face
                for bbox in faces:
                    # Get embedding
                    embedding = self.get_face_embedding(frame, bbox)
                    
                    if embedding is not None:
                        # Match against watchlist
                        name, score = self.match_face(embedding)
                        
                        # Draw bounding box
                        self.draw_bbox(frame, bbox, name, score)
                        
                        # Handle matches
                        if name and score > self.threshold:
                            self.log_detection(name, score, bbox)
                            self.play_alert()
                
                # Calculate and display performance metrics
                end_time = time.time()
                processing_time = (end_time - start_time) * 1000  # Convert to ms
                
                if self.show_fps:
                    fps = self.calculate_fps()
                    if fps > 0:
                        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f"Latency: {processing_time:.1f}ms", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Casino Fraud Detection', frame)
                
                # Check for ESC key
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    break
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.camera is not None:
            self.camera.release()
        cv2.destroyAllWindows()
        print("Cleanup complete")

# %%
def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Casino Fraud Detection Prototype')
    parser.add_argument('--watchlist', type=str, default='watchlist.json',
                       help='Path to watchlist JSON file (default: watchlist.json)')
    parser.add_argument('--threshold', type=float, default=0.55,
                       help='Similarity threshold for matching (default: 0.55)')
    parser.add_argument('--cam_id', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--show_fps', action='store_true',
                       help='Show FPS instead of latency')
    
    args = parser.parse_args()
    
    # Create and run detector
    detector = CasinoFraudDetector(
        watchlist_path=args.watchlist,
        threshold=args.threshold,
        cam_id=args.cam_id,
        show_fps=args.show_fps
    )
    
    detector.run()

# %%
if __name__ == "__main__":
    main()