import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

from nets.srgan import Generator
from utils.utils import cvtColor, preprocess_input


class SRGAN(object):
    #-----------------------------------------#
    #   Note: modify model_path as needed
    #-----------------------------------------#
    _defaults = {
        #-----------------------------------------------#
        #   model_path points to weights file in logs folder
        #-----------------------------------------------#
        "model_path"        : 'model_data/Generator_SRGAN.pth',
        #-----------------------------------------------#
        #   Upsampling factor, same as during training
        #-----------------------------------------------#
        "scale_factor"      : 4, 
        #-------------------------------#
        #   Whether to use CUDA
        #   Set to False if no GPU available
        #-------------------------------#
        "cuda"              : True,
    }

    #---------------------------------------------------#
    #   Initialize SRGAN
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)  
        self.generate()

    def generate(self):
        self.net    = Generator(self.scale_factor)

        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()
        
    def generate_1x1_image(self, image):
        #---------------------------------------------------------#
        #   Convert image to RGB format to prevent grayscale prediction errors.
        #   The code only supports RGB image prediction, all other image types will be converted to RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   Add batch_size dimension and perform normalization
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image, dtype='float32'), [0.5,0.5,0.5], [0.5,0.5,0.5]), [2,0,1]), 0)
        
        with torch.no_grad():
            image_data = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                image_data = image_data.cuda()

            #---------------------------------------------------------#
            #   Input image into network for prediction!
            #---------------------------------------------------------#
            hr_image = self.net(image_data)[0]
            #---------------------------------------------------------#
            #   Convert normalized result back to RGB format
            #---------------------------------------------------------#
            hr_image = (hr_image.cpu().data.numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255
            hr_image = Image.fromarray(np.uint8(hr_image))

            return hr_image
