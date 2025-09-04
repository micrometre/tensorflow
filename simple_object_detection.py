#!/usr/bin/env python3
"""
Simple Object Detection Script
A minimal version for quick object detection with offline image storage
"""

import os
import io
import time
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor, ImageOps
from urllib.request import urlopen


def download_image(url, save_dir="images", width=640, height=480):
    """Download and save image from URL"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    filename = f"image_{int(time.time())}.jpg"
    filepath = os.path.join(save_dir, filename)
    
    try:
        response = urlopen(url)
        image_data = response.read()
        
        pil_image = Image.open(io.BytesIO(image_data))
        pil_image = ImageOps.fit(pil_image, (width, height), Image.LANCZOS)
        pil_image.convert("RGB").save(filepath, "JPEG", quality=90)
        
        print(f"Downloaded: {filepath}")
        return filepath
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None


def detect_objects_in_image(detector, image_path):
    """Run object detection on a single image"""
    # Load and preprocess image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    
    # Run detection
    start_time = time.time()
    result = detector(converted_img)
    inference_time = time.time() - start_time
    
    # Convert results to numpy
    result = {key: value.numpy() for key, value in result.items()}
    
    print(f"Objects found: {len(result['detection_scores'])}")
    print(f"Inference time: {inference_time:.3f}s")
    
    return img.numpy(), result, inference_time


def draw_detections(image, boxes, class_names, scores, min_score=0.1):
    """Draw bounding boxes and labels on image"""
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'cyan']
    
    image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
    draw = ImageDraw.Draw(image_pil)
    width, height = image_pil.size
    
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    for i in range(len(scores)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = boxes[i]
            left, right, top, bottom = (xmin * width, xmax * width, 
                                      ymin * height, ymax * height)
            
            color = colors[i % len(colors)]
            
            # Draw box
            draw.rectangle([(left, top), (right, bottom)], outline=color, width=3)
            
            # Draw label
            label = f"{class_names[i].decode('ascii')}: {int(100 * scores[i])}%"
            if font:
                draw.text((left, top - 20), label, fill=color, font=font)
    
    return np.array(image_pil)


def main():
    # Test image URLs
    test_urls = [
        "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg",
        "https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg"
    ]
    
    print("Loading TensorFlow Hub model...")
    model_url = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
    detector = hub.load(model_url).signatures['default']
    print("Model loaded!")
    
    # Process images
    for i, url in enumerate(test_urls):
        print(f"\n--- Processing image {i+1} ---")
        
        # Download image
        image_path = download_image(url)
        if not image_path:
            continue
        
        # Detect objects
        image, results, inference_time = detect_objects_in_image(detector, image_path)
        
        # Draw results
        annotated_image = draw_detections(
            image, 
            results['detection_boxes'],
            results['detection_class_entities'],
            results['detection_scores']
        )
        
        # Save annotated image
        output_path = image_path.replace('.jpg', '_detected.jpg')
        Image.fromarray(annotated_image).save(output_path)
        print(f"Saved annotated image: {output_path}")
        
        # Show results
        plt.figure(figsize=(10, 8))
        plt.imshow(annotated_image)
        plt.title(f"Object Detection Results (Inference: {inference_time:.3f}s)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    print(f"\nAll images saved in: ./images/")


if __name__ == "__main__":
    main()
