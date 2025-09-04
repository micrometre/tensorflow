#!/usr/bin/env python3
"""
Demo script to test object detection setup
"""

import os
import sys

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'tensorflow',
        'tensorflow_hub', 
        'numpy',
        'matplotlib',
        'PIL'
    ]
    
    print("Checking dependencies...")
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("All dependencies installed!")
    return True

def run_demo():
    """Run a simple demo"""
    if not check_dependencies():
        return
    
    print("\n" + "="*50)
    print("OBJECT DETECTION DEMO")
    print("="*50)
    
    # Step 1: Download images
    print("\n1. Downloading test images...")
    if os.system("python download_images.py") != 0:
        print("Error downloading images!")
        return
    
    # Step 2: Check images directory
    if os.path.exists("images"):
        image_files = [f for f in os.listdir("images") if f.endswith('.jpg')]
        print(f"Found {len(image_files)} images in images/ directory")
    else:
        print("Images directory not found!")
        return
    
    # Step 3: Run object detection
    print("\n2. Running object detection on one test image...")
    print("This will download the TensorFlow Hub model on first run...")
    
    # Simple test with one image
    test_script = """
import tensorflow as tf
import tensorflow_hub as hub
import os

print("Loading model...")
model_url = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
detector = hub.load(model_url).signatures['default']
print("Model loaded successfully!")

# Test with first available image
image_files = [f for f in os.listdir("images") if f.endswith('.jpg') and not f.endswith('_detected.jpg')]
if image_files:
    test_image = os.path.join("images", image_files[0])
    print(f"Testing with: {test_image}")
    
    # Load and process image
    img = tf.io.read_file(test_image)
    img = tf.image.decode_jpeg(img, channels=3)
    converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    
    # Run detection
    result = detector(converted_img)
    result = {key: value.numpy() for key, value in result.items()}
    
    print(f"Detection completed! Found {len(result['detection_scores'])} objects")
    print("Demo successful!")
else:
    print("No test images found!")
"""
    
    # Write and execute test script
    with open("temp_test.py", "w") as f:
        f.write(test_script)
    
    exit_code = os.system("python temp_test.py")
    os.remove("temp_test.py")
    
    if exit_code == 0:
        print("\n✓ Demo completed successfully!")
        print("\nNext steps:")
        print("- Run 'python simple_object_detection.py' for full demo")
        print("- Run 'python object_detection_offline.py --help' for advanced options")
        print("- Check images/ directory for downloaded images")
    else:
        print("\n✗ Demo failed. Check error messages above.")

if __name__ == "__main__":
    run_demo()
