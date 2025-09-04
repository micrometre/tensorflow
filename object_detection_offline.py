#!/usr/bin/env python3
"""
Simplified Object Detection Script for Offline Use
Adapted from object_detection.ipynb

This script downloads test images to a local directory and performs
object detection using TensorFlow Hub pre-trained models.
"""

import os
import time
import argparse
from urllib.request import urlopen
from urllib.parse import urlparse
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps


class ObjectDetector:
    def __init__(self, model_url=None, images_dir="images"):
        """
        Initialize the Object Detector
        
        Args:
            model_url: TensorFlow Hub model URL
            images_dir: Directory to store downloaded images
        """
        self.images_dir = images_dir
        self.create_images_directory()
        
        # Default to Faster R-CNN model if not specified
        if model_url is None:
            model_url = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
        
        print(f"Loading model from: {model_url}")
        self.detector = hub.load(model_url).signatures['default']
        print("Model loaded successfully!")
        
    def create_images_directory(self):
        """Create images directory if it doesn't exist"""
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
            print(f"Created directory: {self.images_dir}")
    
    def download_image(self, url, filename=None, width=640, height=480):
        """
        Download and resize image from URL
        
        Args:
            url: Image URL
            filename: Local filename (auto-generated if None)
            width: Target width
            height: Target height
            
        Returns:
            Local file path of downloaded image
        """
        if filename is None:
            # Generate filename from URL
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            if not filename or not filename.endswith(('.jpg', '.jpeg', '.png')):
                filename = f"image_{int(time.time())}.jpg"
        
        filepath = os.path.join(self.images_dir, filename)
        
        # Skip download if file already exists
        if os.path.exists(filepath):
            print(f"Image already exists: {filepath}")
            return filepath
        
        try:
            print(f"Downloading: {url}")
            response = urlopen(url)
            image_data = response.read()
            
            # Open and resize image
            pil_image = Image.open(io.BytesIO(image_data))
            pil_image = ImageOps.fit(pil_image, (width, height), Image.LANCZOS)
            pil_image_rgb = pil_image.convert("RGB")
            
            # Save image
            pil_image_rgb.save(filepath, format="JPEG", quality=90)
            print(f"Image saved to: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return None
    
    def load_image(self, path):
        """Load image from file path"""
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        return img
    
    def draw_bounding_boxes(self, image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
        """Draw bounding boxes on image"""
        colors = list(ImageColor.colormap.values())
        
        # Try to load a better font, fallback to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf", 25)
        except (IOError, OSError):
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 25)  # macOS
            except (IOError, OSError):
                font = ImageFont.load_default()
        
        image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
        draw = ImageDraw.Draw(image_pil)
        im_width, im_height = image_pil.size
        
        for i in range(min(boxes.shape[0], max_boxes)):
            if scores[i] >= min_score:
                ymin, xmin, ymax, xmax = tuple(boxes[i])
                left, right, top, bottom = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)
                
                # Draw bounding box
                color = colors[hash(class_names[i]) % len(colors)]
                draw.rectangle([(left, top), (right, bottom)], outline=color, width=4)
                
                # Draw label
                display_str = f"{class_names[i].decode('ascii')}: {int(100 * scores[i])}%"
                bbox = font.getbbox(display_str)
                text_width, text_height = bbox[2], bbox[3]
                
                # Background for text
                draw.rectangle([(left, top - text_height - 10), 
                              (left + text_width + 10, top)], 
                              fill=color)
                draw.text((left + 5, top - text_height - 5), display_str, 
                         fill="black", font=font)
        
        return np.array(image_pil)
    
    def detect_objects(self, image_path, save_result=True, show_result=True):
        """
        Perform object detection on image
        
        Args:
            image_path: Path to image file
            save_result: Save annotated image
            show_result: Display result using matplotlib
            
        Returns:
            Dictionary with detection results
        """
        print(f"\nProcessing: {image_path}")
        
        # Load image
        img = self.load_image(image_path)
        converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
        
        # Run inference
        start_time = time.time()
        result = self.detector(converted_img)
        end_time = time.time()
        
        # Convert tensors to numpy arrays
        result = {key: value.numpy() for key, value in result.items()}
        
        print(f"Found {len(result['detection_scores'])} objects")
        print(f"Inference time: {end_time - start_time:.3f} seconds")
        
        # Draw bounding boxes
        image_with_boxes = self.draw_bounding_boxes(
            img.numpy(), 
            result["detection_boxes"],
            result["detection_class_entities"], 
            result["detection_scores"]
        )
        
        # Save annotated image
        if save_result:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(self.images_dir, f"{base_name}_detected.jpg")
            Image.fromarray(image_with_boxes).save(output_path)
            print(f"Annotated image saved to: {output_path}")
        
        # Display result
        if show_result:
            plt.figure(figsize=(12, 8))
            plt.imshow(image_with_boxes)
            plt.axis('off')
            plt.title(f"Object Detection Results - {os.path.basename(image_path)}")
            plt.tight_layout()
            plt.show()
        
        return result
    
    def process_urls(self, image_urls):
        """Download and process multiple images from URLs"""
        results = []
        
        for i, url in enumerate(image_urls):
            print(f"\n{'='*50}")
            print(f"Processing image {i+1}/{len(image_urls)}")
            
            # Download image
            image_path = self.download_image(url)
            
            if image_path:
                # Detect objects
                result = self.detect_objects(image_path)
                results.append({
                    'url': url,
                    'path': image_path,
                    'result': result
                })
            else:
                print(f"Skipping image {i+1} due to download error")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Object Detection with Offline Image Storage")
    parser.add_argument("--images-dir", default="images", help="Directory to store images")
    parser.add_argument("--model", default=None, help="TensorFlow Hub model URL")
    parser.add_argument("--urls", nargs="+", help="Image URLs to process")
    parser.add_argument("--local", nargs="+", help="Local image files to process")
    
    args = parser.parse_args()
    
    # Default test images if none provided
    if not args.urls and not args.local:
        args.urls = [
            "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg",
            "https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Biblioteca_Maim%C3%B3nides%2C_Campus_Universitario_de_Rabanales_007.jpg/1024px-Biblioteca_Maim%C3%B3nides%2C_Campus_Universitario_de_Rabanales_007.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/0/09/The_smaller_British_birds_%288053836633%29.jpg"
        ]
    
    # Initialize detector
    detector = ObjectDetector(model_url=args.model, images_dir=args.images_dir)
    
    # Process URLs
    if args.urls:
        print(f"Processing {len(args.urls)} images from URLs...")
        results = detector.process_urls(args.urls)
        
        print(f"\n{'='*50}")
        print("SUMMARY")
        print(f"{'='*50}")
        for i, result in enumerate(results):
            print(f"Image {i+1}: {result['path']}")
            num_detections = len(result['result']['detection_scores'])
            print(f"  Objects detected: {num_detections}")
    
    # Process local files
    if args.local:
        print(f"\nProcessing {len(args.local)} local images...")
        for image_path in args.local:
            if os.path.exists(image_path):
                detector.detect_objects(image_path)
            else:
                print(f"File not found: {image_path}")
    
    print(f"\nAll images and results saved in: {args.images_dir}")


if __name__ == "__main__":
    # Fix missing import
    import io
    main()
