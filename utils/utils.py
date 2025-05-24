import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch


def show_result(num_epoch, G_net, imgs_lr, imgs_hr):
    with torch.no_grad():
        test_images = G_net(imgs_lr)

        fig, ax = plt.subplots(1, 2)

        for j in itertools.product(range(2)):
            ax[j].get_xaxis().set_visible(False)
            ax[j].get_yaxis().set_visible(False)
        
        ax[0].cla()
        ax[0].imshow(np.transpose(test_images.cpu().numpy()[0] * 0.5 + 0.5, [1,2,0]))

        ax[1].cla()
        ax[1].imshow(np.transpose(imgs_hr.cpu().numpy()[0] * 0.5 + 0.5, [1,2,0]))

        label = 'Epoch {0}'.format(num_epoch)
        fig.text(0.5, 0.04, label, ha='center')
        plt.savefig("results/train_out/epoch_" + str(num_epoch) + "_results.png")
        plt.close('all')  # Avoid memory leaks

#---------------------------------------------------------#
#   Convert image to RGB format to prevent grayscale prediction errors.
#   The code only supports RGB image prediction, all other image types will be converted to RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

def preprocess_input(image, mean, std):
    image = (image/255 - mean)/std
    return image

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']