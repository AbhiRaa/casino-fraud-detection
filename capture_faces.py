#!/usr/bin/env python3
"""
Simple face capture script for building watchlist
"""

import cv2
import os
import sys

def capture_person_images(person_name):
    """Capture images for a specific person."""
    # Create person folder
    person_folder = f'sample_faces/{person_name}'
    os.makedirs(person_folder, exist_ok=True)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print(f"\nCapturing images for: {person_name}")
    print("Instructions:")
    print("- Press SPACE to capture image")
    print("- Press ESC to finish with this person")
    print("- Take 3-5 photos from different angles")
    print("- Make sure your face is well-lit and clear")
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Display instructions on frame
        cv2.putText(frame, f"Person: {person_name}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Photos taken: {count}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE = Capture, ESC = Next person", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Face Capture', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Space to capture
            filename = f'{person_folder}/img{count + 1}.jpg'
            cv2.imwrite(filename, frame)
            print(f'✓ Captured: {filename}')
            count += 1
        elif key == 27:  # ESC to exit
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {count} images for {person_name}")
    return count

def main():
    print("=== Face Capture Tool ===")
    print("This will help you capture face images for the watchlist")
    
    # Default person names
    default_people = ["john_doe", "jane_smith", "bob_johnson"]
    
    print("\nDefault people to capture:")
    for i, person in enumerate(default_people, 1):
        print(f"{i}. {person}")
    
    choice = input("\nUse default names? (y/n): ").lower()
    
    if choice == 'y':
        people = default_people
    else:
        people = []
        while True:
            name = input("Enter person name (or press Enter to finish): ").strip()
            if not name:
                break
            # Replace spaces with underscores
            name = name.replace(' ', '_').lower()
            people.append(name)
    
    if not people:
        print("No people specified. Exiting.")
        return
    
    # Capture images for each person
    total_images = 0
    for person in people:
        count = capture_person_images(person)
        total_images += count
        
        if count > 0:
            continue_choice = input(f"\nContinue to next person? (y/n): ").lower()
            if continue_choice != 'y':
                break
    
    print(f"\n=== Summary ===")
    print(f"Total images captured: {total_images}")
    print(f"People: {', '.join(people)}")
    
    if total_images > 0:
        print("\nNext steps:")
        print("1. Generate watchlist: python generate_watchlist.py --img_root ./sample_faces --out watchlist.json")
        print("2. Run detection: python casino_proto.py --watchlist watchlist.json")
    else:
        print("\nNo images captured. Please run the script again.")

if __name__ == "__main__":
    main()