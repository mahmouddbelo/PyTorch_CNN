# Cotton Disease Classification Web App with PyTorch and Flask

This project provides a deep learning solution for classifying images of cotton crops into four categories: healthy cotton and three types of diseased cotton. A Convolutional Neural Network (CNN) built with PyTorch performs the classification, while Flask serves as the backend for deploying a web application where users can upload images for real-time classification.

![Model Architecture](https://github.com/mahmouddbelo/PyTorch_CNN/blob/main/Smart%20Agricultural%20ML%20System%20-%20Google%20Chrome%2010_28_2024%206_46_23%20PM.png)  

*Figure 1. Model Architecture Overview*

---

## Table of Contents
- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Classes](#classes)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview
This repository offers an end-to-end solution for cotton disease classification:
- **Deep Learning Model**: CNN in PyTorch to classify cotton crop images.
- **Web Deployment**: Flask-based app for real-time classification of uploaded cotton crop images.

## Model Architecture
The model uses a Convolutional Neural Network (CNN) with the following characteristics:
1. **Feature Extraction**: Convolutional and pooling layers to capture essential spatial features.
2. **Classification**: Fully connected layers and a softmax output layer for final class prediction.

![Classification Example](https://github.com/mahmouddbelo/PyTorch_CNN/blob/main/Smart%20Agricultural%20ML%20System%20-%20Google%20Chrome%2010_28_2024%206_46_12%20PM.png)  
*Figure 2. Classification Example in Action*

### Classes
The model classifies images into the following categories:
- **Fresh Cotton Plant**
- **Diseased Cotton Leaf**
- **Diseased Cotton Plant**
- **Fresh Cotton Leaf**

## Installation

### Prerequisites
- Python 3.6 or higher
- [PyTorch](https://pytorch.org/get-started/locally/) (GPU support if available)
- Flask
- Other dependencies as listed in `requirements.txt`

### Setup
1. Clone the repository:
    ```bash
    git clone https://github.com/mahmouddbelo/PyTorch_CNN
    cd PyTorch_CNN
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. (Optional) Install PyTorch with GPU support:
    ```bash
    pip install torch torchvision
    ```

## Usage

### Running the Model
To test the model directly:
```bash
python model.py
