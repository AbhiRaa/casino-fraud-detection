#!/usr/bin/env python3
"""
Simple image capture using system tools
"""

import os
import subprocess
import sys
import time

def capture_with_system_camera():
    """Use system camera tools to capture images."""
    print("=== Simple Face Capture ===")
    print("This will use your system's camera to capture images.")
    print("We'll create a simple script to help organize the images.")
    
    # Create sample faces directory
    os.makedirs('sample_faces', exist_ok=True)
    
    # Default person names
    people = ["john_doe", "jane_smith", "bob_johnson"]
    
    print("\nWe'll capture images for these people:")
    for person in people:
        print(f"- {person}")
    
    print("\nFor each person, we'll:")
    print("1. Create their folder")
    print("2. You'll take 3-5 photos using your system camera")
    print("3. Save them to the person's folder")
    
    for person in people:
        person_folder = f'sample_faces/{person}'
        os.makedirs(person_folder, exist_ok=True)
        
        print(f"\n--- {person.replace('_', ' ').title()} ---")
        print(f"Folder created: {person_folder}")
        
        # Instructions for manual photo capture
        print("\nMANUAL STEPS:")
        print("1. Open your system camera app (Photo Booth on Mac, Camera on Windows)")
        print("2. Take 3-5 clear photos of the person")
        print("3. Save them as JPG files in this folder:")
        print(f"   {os.path.abspath(person_folder)}")
        print("4. Name them: img1.jpg, img2.jpg, img3.jpg, etc.")
        
        # Wait for user to complete
        input(f"\nPress Enter when you've added images for {person}...")
        
        # Check if images were added
        images = [f for f in os.listdir(person_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if images:
            print(f"✓ Found {len(images)} images: {', '.join(images)}")
        else:
            print("⚠ No images found. You can add them later.")
    
    # Final instructions
    print("\n=== Next Steps ===")
    print("Once you have images in the folders, run:")
    print("1. python generate_watchlist.py --img_root ./sample_faces --out watchlist.json")
    print("2. python casino_proto.py --watchlist watchlist.json")
    
    # Show folder structure
    print("\n=== Folder Structure ===")
    for root, dirs, files in os.walk('sample_faces'):
        level = root.replace('sample_faces', '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")

def create_empty_watchlist():
    """Create an empty watchlist for testing without images."""
    print("\n=== Alternative: Empty Watchlist ===")
    print("If you want to test the system without adding images:")
    
    empty_watchlist = "{}"
    with open('watchlist.json', 'w') as f:
        f.write(empty_watchlist)
    
    print("✓ Created empty watchlist.json")
    print("✓ You can now run: python casino_proto.py --watchlist watchlist.json")
    print("✓ All faces will show as 'Unknown' but you can test the detection system")

if __name__ == "__main__":
    try:
        capture_with_system_camera()
        
        print("\n" + "="*50)
        choice = input("Want to create an empty watchlist for testing? (y/n): ").lower()
        if choice == 'y':
            create_empty_watchlist()
    except KeyboardInterrupt:
        print("\nCancelled by user")
    except Exception as e:
        print(f"Error: {e}")
        create_empty_watchlist()