# TensorFlow Machine Learning Projects

This repository contains a collection of TensorFlow tutorials and machine learning projects covering various aspects of deep learning, including object detection, text classification, image processing, and tensor operations.

## 📁 Project Structure

```
├── requirements.txt                # Project dependencies
└── notbboks/                      # Collection of tutorial notebooks
    ├── classification.ipynb       # Classification examples
    ├── tensor_basics.ipynb        # Introduction to tensors
    ├── tensor_image_example.ipynb # Image processing with tensors
    |── text_classification.ipynb  # Text sentiment analysis
    ├── object_detection.ipynb     # Object detection using TensorFlow Hub models
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. Clone this repository:
```bash
git clone 
cd tensorflow
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate     # On Windows
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies

The project uses the following main libraries:
- **TensorFlow**: Core machine learning framework
- **TensorFlow Hub**: Pre-trained model repository
- **NumPy**: Numerical computations
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Additional ML utilities
- **IPython**: Interactive computing

## 📚 Notebooks Overview

### 1. Object Detection (`object_detection.ipynb`)
- Demonstrates object detection using TensorFlow Hub pre-trained models
- Uses state-of-the-art models like Faster R-CNN with Inception ResNet v2
- Includes practical examples with image processing and bounding box visualization

### 2. Tensor Basics (`notbboks/tensor_basics.ipynb`)
- Introduction to TensorFlow tensors
- Covers tensor creation, manipulation, and operations
- Perfect starting point for TensorFlow beginners
- Explains tensor properties, shapes, and data types

### 3. Text Classification (`notbboks/text_classification.ipynb`)
- Binary sentiment analysis using the IMDB movie reviews dataset
- Implements neural network models for text classification
- Covers text preprocessing, tokenization, and model training
- Includes performance evaluation and visualization

### 4. Image Processing (`notbboks/tensor_image_example.ipynb`)
- Image processing techniques using TensorFlow
- Tensor operations for computer vision tasks
- Image manipulation and transformation examples

### 5. Classification Examples (`notbboks/classification.ipynb`)
- Various classification algorithms and techniques
- Comparative analysis of different approaches
- Best practices for classification tasks

## 📊 Dataset Information

### IMDB Movie Review Dataset
The project includes the Large Movie Review Dataset v1.0:
- **Size**: 50,000 movie reviews total
- **Training Set**: 25,000 reviews (balanced: 12.5k positive, 12.5k negative)
- **Test Set**: 25,000 reviews (balanced: 12.5k positive, 12.5k negative)
- **Additional**: 50,000 unlabeled reviews for unsupervised learning
- **Format**: Text files with sentiment labels
- **Citation**: Maas et al. (2011) - "Learning Word Vectors for Sentiment Analysis"

## 🎯 Use Cases

This repository is perfect for:
- **Learning TensorFlow**: Step-by-step tutorials from basics to advanced topics
- **Computer Vision**: Object detection and image processing examples
- **Natural Language Processing**: Text classification and sentiment analysis
- **Academic Research**: Pre-processed datasets and baseline implementations
- **Prototyping**: Ready-to-use code for various ML tasks

## 🏃‍♂️ Running the Notebooks

1. Make sure you have activated your virtual environment
2. Start Jupyter Notebook or use VS Code with Jupyter extension:
```bash
jupyter notebook
# or use VS Code with the Jupyter extension
```
3. Open any notebook file (`.ipynb`) and run the cells sequentially

## 📈 Model Performance

The notebooks include various pre-trained models and custom implementations:
- **Object Detection**: Uses TensorFlow Hub models with high accuracy on COCO dataset
- **Text Classification**: Achieves competitive performance on IMDB sentiment analysis
- **Image Processing**: Demonstrates efficient tensor operations for computer vision

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Add new tutorial notebooks
- Improve existing code and documentation
- Fix bugs or performance issues
- Add new datasets or examples

## 📄 License

This project includes code licensed under the Apache License 2.0. The IMDB dataset is provided under its original terms and conditions.

## 🔗 References

- [TensorFlow Official Documentation](https://www.tensorflow.org/)
- [TensorFlow Hub](https://tfhub.dev/)
- [IMDB Dataset Paper](http://www.aclweb.org/anthology/P11-1015)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)

## 📞 Support

For questions or issues:
1. Check the notebook documentation and comments
2. Refer to TensorFlow official documentation
3. Open an issue in this repository

---

*This repository serves as a comprehensive introduction to machine learning with TensorFlow, covering essential topics from tensor basics to advanced applications in computer vision and natural language processing.*
