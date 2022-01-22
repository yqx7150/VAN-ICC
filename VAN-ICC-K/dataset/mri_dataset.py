from __future__ import print_function, division
import os, random, time
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils
from glob import glob
from PIL import Image as PILImage
import numbers
from scipy.misc import imread
from scipy.io import loadmat
import os
IMG_EXTENSIONS = ['.jpg', 'JPG', '.mat']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_image_paths(path):
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    return sorted(images)


def loadmri(path):
    mri_images = loadmat(path)['Img']
    img = np.real(mri_images)
    img = np.array(img, dtype=np.float32)
    
    return img

    
def loadmri_real_imag_cross_mask(path):
    # cross real and imag
    mri_images_temp = loadmat(path)['Img']
    mask = loadmat('./mask/cart_R4_256_256.mat')['mask']
    mri_images = np.zeros(mri_images_temp.shape, dtype=np.complex128)
    for i in range(mri_images_temp.shape[-1]):
        mri_images[:, :, i] = np.fft.fftshift(np.fft.fft2(mri_images_temp[:, :, i]))*mask
    img_real = np.real(mri_images)
    img_real = np.array(img_real, dtype=np.float32)
    img_imag = np.imag(mri_images)
    img_imag = np.array(img_imag, dtype=np.float32)
    img = np.zeros([256, 256, 2*(img_real.shape[-1])], dtype=np.float32)
    for i in range(0, 2*(img_real.shape[-1]), 2):
        img[:, :, i] = img_real[:, :, int(i/2)]
        img[:, :, i+1] = img_imag[:, :, int(i/2)]

    return img


class mriDataset12and4_real_imag_cross(Dataset):
    def __init__(self, root1, root):
        self.path_channel12_mri = get_image_paths(root1)
        self.datanames = np.array([root+"/"+x for x in os.listdir(root)])

    def np2tensor(self, array):
        return torch.Tensor(array).permute(2,0,1)
        
    def __len__(self):
        return len(self.datanames)
    
    def __getitem__(self, index):
        # need 3, 256, 256
        # 加载12线圈图
        path_channel12 = self.path_channel12_mri[index]
        # 取出其中名字
        channel12_name = path_channel12.split('/')[-1].split('.')[0]
        # 12 real 12 imag
        # 加载24通道的实虚交叉原图
        channel24_mri = loadmri_real_imag_cross_mask(path_channel12)
        ###############################################################################################
        # 复制
        output_channel24_mri = channel24_mri.copy()
            
        # 将以上图像均转换为tensor形式，变为need的形式
        channel24_mri = self.np2tensor(channel24_mri)
        output_channel24_mri = self.np2tensor(output_channel24_mri)

        sample = {  'input_channel24_mri':channel24_mri, 
                    'target_channel24_mri':output_channel24_mri,
                    'channelname':channel12_name}
        
        return sample
