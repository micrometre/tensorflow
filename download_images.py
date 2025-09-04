#!/usr/bin/env python3
"""
Image Downloader for Object Detection
Downloads all test images used in the notebook for offline processing
"""

import os
import io
import time
from urllib.request import urlopen
from PIL import Image, ImageOps


def download_image_with_name(url, filename, save_dir="images", width=640, height=480):
    """Download image with specific filename"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    filepath = os.path.join(save_dir, filename)
    
    # Skip if already exists
    if os.path.exists(filepath):
        print(f"Already exists: {filename}")
        return filepath
    
    try:
        print(f"Downloading: {filename}")
        response = urlopen(url)
        image_data = response.read()
        
        pil_image = Image.open(io.BytesIO(image_data))
        pil_image = ImageOps.fit(pil_image, (width, height), Image.LANCZOS)
        pil_image.convert("RGB").save(filepath, "JPEG", quality=90)
        
        print(f"Saved: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return None


def main():
    """Download all test images from the notebook"""
    
    # Images from the notebook
    images_to_download = [
        {
            "url": "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg",
            "filename": "grace_hopper.jpg",
            "description": "Grace Hopper - computer scientist"
        },
        {
            "url": "https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg", 
            "filename": "yellow_labrador.jpg",
            "description": "Yellow Labrador dog"
        },
        {
            "url": "https://upload.wikimedia.org/wikipedia/commons/1/1b/The_Coleoptera_of_the_British_islands_%28Plate_125%29_%288592917784%29.jpg",
            "filename": "beetles_plate.jpg",
            "description": "British beetles scientific illustration"
        },
        {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Biblioteca_Maim%C3%B3nides%2C_Campus_Universitario_de_Rabanales_007.jpg/1024px-Biblioteca_Maim%C3%B3nides%2C_Campus_Universitario_de_Rabanales_007.jpg",
            "filename": "library_campus.jpg", 
            "description": "University library building"
        },
        {
            "url": "https://upload.wikimedia.org/wikipedia/commons/0/09/The_smaller_British_birds_%288053836633%29.jpg",
            "filename": "british_birds.jpg",
            "description": "British birds scientific illustration"
        }
    ]
    
    print("Downloading images for offline object detection...")
    print(f"Total images: {len(images_to_download)}")
    
    successful_downloads = 0
    
    for i, img_info in enumerate(images_to_download, 1):
        print(f"\n[{i}/{len(images_to_download)}] {img_info['description']}")
        
        result = download_image_with_name(
            img_info["url"], 
            img_info["filename"],
            save_dir="images"
        )
        
        if result:
            successful_downloads += 1
    
    print(f"\n{'='*50}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*50}")
    print(f"Successfully downloaded: {successful_downloads}/{len(images_to_download)} images")
    print(f"Images saved in: ./images/")
    
    # Create a list file for easy reference
    with open("images/image_list.txt", "w") as f:
        f.write("Downloaded Images for Object Detection\n")
        f.write("="*40 + "\n\n")
        for img_info in images_to_download:
            f.write(f"Filename: {img_info['filename']}\n")
            f.write(f"Description: {img_info['description']}\n") 
            f.write(f"Source URL: {img_info['url']}\n")
            f.write("-" * 40 + "\n")
    
    print("Image list saved to: ./images/image_list.txt")


if __name__ == "__main__":
    main()
