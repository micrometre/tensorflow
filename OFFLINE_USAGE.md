# Object Detection Offline Scripts

This directory contains simplified Python scripts extracted from the `object_detection.ipynb` notebook for offline use with pre-downloaded images.

## Scripts Overview

### 1. `download_images.py` 
Downloads all test images used in the notebook to a local `images/` directory.

**Usage:**
```bash
python download_images.py
```

**Output:**
- Creates `images/` directory
- Downloads 5 test images with descriptive names
- Creates `images/image_list.txt` with image descriptions

### 2. `simple_object_detection.py`
Minimal script that downloads and processes images with object detection.

**Usage:**
```bash
python simple_object_detection.py
```

**Features:**
- Downloads test images automatically
- Performs object detection using Faster R-CNN
- Saves annotated images with bounding boxes
- Displays results using matplotlib

### 3. `object_detection_offline.py`
Full-featured script with command-line options for batch processing.

**Usage:**
```bash
# Process default test images
python object_detection_offline.py

# Process specific URLs
python object_detection_offline.py --urls "https://example.com/image1.jpg" "https://example.com/image2.jpg"

# Process local images
python object_detection_offline.py --local "path/to/image1.jpg" "path/to/image2.jpg"

# Specify custom images directory
python object_detection_offline.py --images-dir "my_images"

# Use different TF Hub model
python object_detection_offline.py --model "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
```

## Quick Start

1. **Download images first:**
   ```bash
   python download_images.py
   ```

2. **Run simple detection:**
   ```bash
   python simple_object_detection.py
   ```

3. **For batch processing:**
   ```bash
   python object_detection_offline.py --local images/*.jpg
   ```

## Models Available

The scripts support two pre-trained models from TensorFlow Hub:

1. **Faster R-CNN + Inception ResNet V2** (default)
   - URL: `https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1`
   - High accuracy, slower inference
   - Good for detailed object detection

2. **SSD + MobileNet V2**
   - URL: `https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1`  
   - Lower accuracy, faster inference
   - Good for real-time applications

## Output Files

All scripts save results to the `images/` directory:

- **Original images**: `image_name.jpg`
- **Annotated images**: `image_name_detected.jpg`
- **Image list**: `image_list.txt` (descriptions and sources)

## Dependencies

Make sure you have installed the requirements:
```bash
pip install -r requirements.txt
```

Required packages:
- tensorflow
- tensorflow-hub
- numpy
- matplotlib
- pillow (PIL)

## Offline Usage

1. First run with internet connection to:
   - Download TensorFlow Hub model (cached locally)
   - Download test images

2. Subsequent runs can work offline using:
   - Cached model from TensorFlow Hub
   - Local images in `images/` directory

## Performance Notes

- **First run**: Downloads model (~200MB) and caches it
- **Model loading**: ~5-10 seconds on first load
- **Inference time**: 
  - Faster R-CNN: ~2-5 seconds per image
  - SSD MobileNet: ~0.5-1 second per image

## Customization

You can easily modify the scripts to:
- Add your own images to process
- Change detection confidence thresholds
- Modify bounding box visualization
- Use different pre-trained models
- Batch process entire directories

## Troubleshooting

**Model download issues:**
- Ensure internet connection for first run
- Check TensorFlow Hub cache: `~/.cache/tensorflow_hub/`

**Image download issues:**
- Some URLs may be blocked by firewalls
- Use local images with `--local` option

**Memory issues:**
- Reduce image size in download functions
- Process images one at a time instead of batches
