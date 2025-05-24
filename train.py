import os
import glob
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.vgg import vgg19

from nets.srgan import Discriminator, Generator
from utils.dataloader import SRGAN_dataset_collate, SRGANDataset
from utils.utils_fit import fit_one_epoch

def get_dataset_lines(datasets_path="datasets/"):
    """
    Automatically scan the dataset folder and return image path list.
    Integrated functionality from txt_annotation.py, no need to run separately.
    
    Args:
        datasets_path: Path to the dataset folder
    
    Returns:
        list: Formatted path list (each element ends with \n, compatible with SRGANDataset)
    """
    if not os.path.exists(datasets_path):
        raise ValueError(f"❌ Dataset folder does not exist: {datasets_path}")
    
    # Supported image formats (consistent with txt_annotation.py)
    image_extensions = ['.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff']
    
    # Use glob to scan all image files
    image_files = []
    for ext in image_extensions:
        pattern = os.path.join(datasets_path, f"*{ext}")
        image_files.extend(glob.glob(pattern))
        # Also match uppercase extensions
        pattern = os.path.join(datasets_path, f"*{ext.upper()}")
        image_files.extend(glob.glob(pattern))
    
    # Sort and convert to absolute paths (consistent with txt_annotation.py)
    image_files = sorted([os.path.abspath(f) for f in image_files])
    
    if len(image_files) == 0:
        raise ValueError(f"❌ No image files found in {datasets_path}!\n"
                        f"💡 Please ensure:\n"
                        f"   1. The folder contains image files\n" 
                        f"   2. Supported formats: {', '.join(image_extensions)}")
    
    print(f"✅ Successfully scanned {len(image_files)} training images")
    print(f"📁 Dataset path: {os.path.abspath(datasets_path)}")
    
    # Convert to format expected by SRGANDataset (one path per line)
    lines = [f"{path}\n" for path in image_files]
    return lines

if __name__ == "__main__":
    #-------------------------------#
    #   Whether to use CUDA
    #   Set to False if no GPU available
    #-------------------------------#
    Cuda            = True
    #-----------------------------------#
    #   Represents 4x upsampling
    #-----------------------------------#
    scale_factor    = 4
    #-----------------------------------#
    #   Get input and output image shapes
    #-----------------------------------#
    lr_shape        = [96, 96]
    hr_shape        = [lr_shape[0] * scale_factor, lr_shape[1] * scale_factor]
    #--------------------------------------------------------------------------#
    #   For resuming training, set model_path to the trained weights file in logs folder.
    #   When model_path = '', no model weights will be loaded.
    #
    #   Here we use the complete model weights, so they are loaded in train.py.
    #   To train the model from scratch, set model_path = ''.
    #--------------------------------------------------------------------------#
    G_model_path    = ""
    D_model_path    = ""

    #------------------------------#
    #   Training parameter settings
    #------------------------------#
    Init_epoch      = 0
    Epoch           = 200
    batch_size      = 4
    lr              = 0.0002
    #------------------------------#
    #   Save images every 50 steps
    #------------------------------#
    save_interval   = 50
    #------------------------------#
    #   Dataset path (New: Directly specify dataset folder)
    #   Replaces the original annotation_path and train_lines.txt
    #------------------------------#
    datasets_path   = "datasets/"

    print("🚀 Starting training preparation...")
    print("📊 Scanning dataset...")
    
    #------------------------------#
    #   Automatic dataset scanning (New feature!)
    #   Replaces the txt_annotation.py + train_lines.txt workflow
    #------------------------------#
    try:
        lines = get_dataset_lines(datasets_path)
        num_train = len(lines)
        print(f"📈 Number of training samples: {num_train}")
    except ValueError as e:
        print(f"{e}")
        print("\n🔧 Solutions:")
        print(f"   1. Create dataset folder: mkdir {datasets_path}")
        print(f"   2. Place your CT scan images into '{datasets_path}' folder")
        print("   3. Ensure correct image formats (JPG, PNG, BMP, TIFF, etc.)")
        exit(1)

    #---------------------------#
    #   Generator and Discriminator networks
    #---------------------------#
    G_model = Generator(scale_factor)
    D_model = Discriminator()
    #-----------------------------------#
    #   Create VGG model for feature extraction
    #-----------------------------------#
    VGG_model = vgg19(pretrained=True)
    VGG_feature_model = nn.Sequential(*list(VGG_model.features)[:-1]).eval()
    for param in VGG_feature_model.parameters():
        param.requires_grad = False

    #------------------------------------------#
    #   Load pre-trained model weights
    #------------------------------------------#
    if G_model_path != '':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = G_model.state_dict()
        pretrained_dict = torch.load(G_model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
        model_dict.update(pretrained_dict)
        G_model.load_state_dict(model_dict)
    if D_model_path != '':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = D_model.state_dict()
        pretrained_dict = torch.load(D_model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
        model_dict.update(pretrained_dict)
        D_model.load_state_dict(model_dict)

    G_model_train = G_model.train()
    D_model_train = D_model.train()
    
    if Cuda:
        cudnn.benchmark = True
        G_model_train = torch.nn.DataParallel(G_model)
        G_model_train = G_model_train.cuda()

        D_model_train = torch.nn.DataParallel(D_model)
        D_model_train = D_model_train.cuda()

        VGG_feature_model = torch.nn.DataParallel(VGG_feature_model)
        VGG_feature_model = VGG_feature_model.cuda()

    # Binary Cross Entropy loss
    BCE_loss = nn.BCELoss()
    MSE_loss = nn.MSELoss()

    #------------------------------------------------------#
    #   Init_Epoch is the starting epoch
    #   Epoch is the total training epochs
    #------------------------------------------------------#
    if True:
        epoch_step      = min(num_train // batch_size, 2000)
        if epoch_step == 0:
            raise ValueError("Dataset too small for training, please expand the dataset.")
        #------------------------------#
        #   Adam optimizer
        #------------------------------#
        G_optimizer     = optim.Adam(G_model_train.parameters(), lr=lr, betas=(0.9, 0.999))
        D_optimizer     = optim.Adam(D_model_train.parameters(), lr=lr, betas=(0.9, 0.999))
        G_lr_scheduler  = optim.lr_scheduler.StepLR(G_optimizer,step_size=1,gamma=0.98)
        D_lr_scheduler  = optim.lr_scheduler.StepLR(D_optimizer,step_size=1,gamma=0.98)

        train_dataset   = SRGANDataset(lines, lr_shape, hr_shape)
        gen             = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True,
                                    drop_last=True, collate_fn=SRGAN_dataset_collate)

        print("🎯 Starting training...")
        for epoch in range(Init_epoch, Epoch):
            fit_one_epoch(G_model_train, D_model_train, G_model, D_model, VGG_feature_model, G_optimizer, D_optimizer, BCE_loss, MSE_loss, epoch, epoch_step, gen, Epoch, Cuda, batch_size, save_interval)
            G_lr_scheduler.step()
            D_lr_scheduler.step()
