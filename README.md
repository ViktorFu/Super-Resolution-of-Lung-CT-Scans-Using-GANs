# Super-Resolution of Lung CT Scans Using GANs

[![Python 3.10](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning implementation of Super-Resolution Generative Adversarial Network (SRGAN) specifically designed for enhancing the resolution of lung CT scans. This project provides 4x upsampling capabilities to improve medical image quality for better diagnosis and analysis.

## 🌟 Features

- **4x Super-Resolution**: Enhance CT scan resolution from 96×96 to 384×384 pixels
- **SRGAN Architecture**: State-of-the-art GAN-based super-resolution with perceptual loss
- **Medical Image Optimized**: Specifically tuned for lung CT scan characteristics
- **CUDA Support**: GPU acceleration for faster training and inference
- **Automatic Dataset Scanning**: Streamlined data pipeline without manual annotation files
- **Multiple Format Support**: Compatible with JPG, PNG, BMP, TIFF, and other common formats
- **Pre-trained Model**: Ready-to-use model weights for immediate inference

## 🏗️ Architecture

The project implements a complete SRGAN framework consisting of:

### Generator Network
- **Residual Blocks**: Deep residual learning for feature extraction
- **Sub-pixel Convolution**: Efficient upsampling with learnable filters
- **Skip Connections**: Preserve fine-grained details during upsampling

### Discriminator Network
- **Adversarial Training**: Ensures realistic texture generation
- **Patch-based Discrimination**: Focus on local image patches for better quality

### Loss Functions
- **Perceptual Loss**: VGG19-based feature matching for realistic textures
- **Adversarial Loss**: GAN loss for photorealistic results
- **Content Loss**: MSE loss for pixel-wise accuracy

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.10+
PyTorch 2.0+
CUDA 11.0+ (optional, for GPU acceleration)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Super-Resolution-of-Lung-CT-Scans-Using-GANs.git
cd Super-Resolution-of-Lung-CT-Scans-Using-GANs
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify CUDA installation** (optional)
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Usage

#### 🔮 Inference (Using Pre-trained Model)

```bash
python predict.py
```

The script will prompt you to:
1. Enter the path to your low-resolution CT image
2. Specify the output path for the enhanced image

#### 🏋️ Training (Custom Dataset)

1. **Prepare your dataset**
```bash
mkdir datasets
# Place your high-resolution CT images in the datasets/ folder
# Supported formats: .jpg, .png, .bmp, .tiff, .jpeg
```

2. **Start training**
```bash
python train.py
```

The training script will:
- Automatically scan the `datasets/` folder
- Generate low-resolution pairs from high-resolution images
- Train the SRGAN model with default parameters

#### ⚙️ Configuration

Key training parameters in `train.py`:

```python
# Training Configuration
Cuda = True                    # Enable CUDA acceleration
scale_factor = 4              # 4x upsampling ratio
lr_shape = [96, 96]          # Low-resolution input size
Epoch = 200                  # Total training epochs
batch_size = 4               # Training batch size
lr = 0.0002                  # Learning rate
save_interval = 50           # Image saving frequency
```

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Upsampling Factor** | 4x |
| **Input Resolution** | 96×96 |
| **Output Resolution** | 384×384 |
| **Model Size** | ~6.3MB |
| **GPU Memory** | ~4GB (training) |
| **Inference Time** | ~50ms (RTX 4060) |

## 🗂️ Project Structure

```
Super-Resolution-of-Lung-CT-Scans-Using-GANs/
├── 📄 train.py              # Training script with auto dataset scanning
├── 📄 predict.py            # Inference script for single images
├── 📄 srgan.py              # Main SRGAN class for inference
├── 📁 nets/
│   └── 📄 srgan.py         # Generator and Discriminator networks
├── 📁 utils/
│   ├── 📄 dataloader.py    # Data loading and augmentation
│   ├── 📄 utils_fit.py     # Training loop implementation
│   ├── 📄 utils_metrics.py # PSNR and SSIM evaluation metrics
│   └── 📄 utils.py         # Utility functions
├── 📁 model_data/
│   └── 📄 Generator_SRGAN.pth  # Pre-trained model weights
└── 📄 README.md            # Project documentation
```

## 💡 Key Improvements

This implementation includes several enhancements over standard SRGAN:

1. **Streamlined Data Pipeline**: Automatic dataset scanning eliminates manual annotation
2. **Medical Image Optimization**: Tuned hyperparameters for CT scan characteristics  
3. **CUDA Integration**: Seamless GPU acceleration support
4. **Error Handling**: Comprehensive error messages and solutions
5. **International Support**: Full English codebase for global collaboration

## 🔧 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size in train.py
batch_size = 2  # or even 1
```

**No Images Found**
```bash
# Ensure your dataset folder structure is correct
datasets/
├── image001.jpg
├── image002.png
└── ...
```

**Module Import Errors**
```bash
# Reinstall dependencies
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 📈 Results

The model demonstrates significant improvements in:
- **Image Sharpness**: Enhanced edge definition and detail preservation
- **Texture Quality**: Realistic tissue texture reconstruction
- **Diagnostic Value**: Improved visibility of anatomical structures

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original SRGAN paper: [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)
- PyTorch team for the excellent deep learning framework
- Medical imaging community for validation and feedback

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please open an issue or contact the development team.

---

**Note**: This model is intended for research and educational purposes. For clinical applications, please ensure proper validation and regulatory compliance.
