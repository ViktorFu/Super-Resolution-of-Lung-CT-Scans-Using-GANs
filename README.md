# Super-Resolution of Lung CT Scans Using GANs

## Overview
This repository presents a **Super-Resolution** model developed for **Lung CT Scans** using **Generative Adversarial Networks (GANs)**. The model aims to enhance the resolution of medical CT images, improving the quality and clarity for better diagnosis and analysis. The approach leverages advanced deep learning techniques to restore high-quality images from low-resolution CT scans.

## Key Features
- **Super-Resolution**: Enhance the resolution of Lung CT scans, providing clearer and more detailed images for analysis.
- **GAN Architecture**: Utilizes a deep learning-based GAN model to generate high-resolution images from low-resolution inputs.
- **Optimized for Medical Imaging**: Specifically designed for improving the quality of CT scans, aiding medical professionals in diagnosis and further research.

## Requirements
Before running this project, ensure that you have the following dependencies installed:
- Python 3.x
- TensorFlow or PyTorch (depending on the implementation)
- NumPy
- OpenCV
- Matplotlib
- Other dependencies listed in the `requirements.txt` file

## Installation
1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/ViktorFu/Super-Resolution-of-Lung-CT-Scans-Using-GANs.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Super-Resolution-of-Lung-CT-Scans-Using-GANs
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To apply the model on your Lung CT scans, follow these steps:

1. Prepare your low-resolution CT scan images in the specified input directory.
2. Run the training or inference script:
   ```bash
   python run_inference.py --input /path/to/low-res-image --output /path/to/save/high-res-image
   ```

## Model Training
If you would like to train the model from scratch:
1. Ensure that you have a dataset of low-resolution and high-resolution Lung CT scans.
2. Start the training process:
   ```bash
   python train_model.py --data /path/to/dataset --epochs 100
   ```

## Results
The model generates high-resolution CT scans from low-resolution inputs, improving the quality of the images while preserving important features. The effectiveness of the model can be visualized through various examples in the `results/` folder.
