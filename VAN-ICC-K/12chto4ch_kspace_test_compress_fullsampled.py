import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np
import os, time, random
import argparse
from torch.utils.data import Dataset, DataLoader
from PIL import Image as PILImage
from model.model import InvISPNet
# from dataset.mri_dataset import mriDataset, mriDataset12, mriDataset12_real_imag, mriDataset12_real_imag_cross
from dataset.mri_dataset_full import mriDataset12and4_real_imag_cross
from config.config import get_arguments
from utils.commons import denorm, preprocess_test_patch
from tqdm import tqdm
from skimage.measure import compare_psnr, compare_ssim
from scipy.io import savemat
from matplotlib import pyplot as plt


os.environ['CUDA_VISIBLE_DEVICES'] = "1"
parser = get_arguments()
parser.add_argument("--ckpt", type=str, default='./exps/2sccto12gcc_cross_weight12_smooth_l1_to26epoch/checkpoint/0025.pth', help="Checkpoint path.")
parser.add_argument("--out_path", type=str, default="./exps/", help="Path to save results. ")
parser.add_argument("--split_to_patch", dest='split_to_patch', action='store_true', help="Test on patch. ")
args = parser.parse_args()

ckpt_name = args.ckpt.split("/")[-1].split(".")[0]
if args.split_to_patch:
    os.makedirs(args.out_path+"%s/results_metric_%s/"%(args.task, ckpt_name), exist_ok=True)
    out_path = args.out_path+"%s/results_metric_%s/"%(args.task, ckpt_name)
else:
    os.makedirs(args.out_path+"%s/results_%s/"%(args.task, ckpt_name), exist_ok=True)
    out_path = args.out_path+"%s/results_%s/"%(args.task, ckpt_name)


def main(args):
    net = InvISPNet(channel_in=24, channel_out=24, block_num=8)
    device = torch.device("cuda:0")
    net.to(device)
    net.eval()
    if os.path.isfile(args.ckpt):
        net.load_state_dict(torch.load(args.ckpt), strict=False)
        print("[INFO] Loaded checkpoint: {}".format(args.ckpt))
    
    print("[INFO] Start data load and preprocessing") 
    mri_dataset = mriDataset12and4_real_imag_cross(root1='./data', root='./data')
    dataloader = DataLoader(mri_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
    
    print("[INFO] Start test...")
    PSNR_COMPRESS = []
    SSIM_COMPRESS = []
    for i_batch, sample_batched in enumerate(tqdm(dataloader)):
        iiii = i_batch + 1
        step_time = time.time()
        
        input_channel24_mri_real_imag = sample_batched['input_channel24_mri'].to(device)
        target_channel24_mri_real_imag = sample_batched['target_channel24_mri'].to(device)

        with torch.no_grad():
            reconstruct_4_real_imag = net(input_channel24_mri_real_imag)


        pred_12_real_imag = reconstruct_4_real_imag.detach().permute(0,2,3,1).squeeze()
        input_channel24_real_imag = input_channel24_mri_real_imag.detach().permute(0,2,3,1).squeeze()
        target_channel24_mri_real_imag = target_channel24_mri_real_imag.detach().permute(0,2,3,1).squeeze()
        
        pred_12_real_imag = pred_12_real_imag.cpu().numpy()
        input_channel24_real_imag = input_channel24_real_imag.cpu().numpy()
        target_channel24_mri_real_imag = target_channel24_mri_real_imag.cpu().numpy()
        
        # input
        input_12coil_complex = np.zeros([256, 256, 12], dtype=np.complex64)
        for i in range(0, 24, 2):
            input_12coil_complex[:, :, int(i/2)] = input_channel24_real_imag[:, :, i] + 1j*input_channel24_real_imag[:, :, i+1]
        # input: from kspace to image
        input_12coil_complex_img = np.zeros(input_12coil_complex.shape, dtype=np.complex64)
        for i in range(input_12coil_complex.shape[-1]):
            input_12coil_complex_img[:, :, i] = np.fft.ifft2(np.fft.fftshift(input_12coil_complex[:, :, i]))
        input_12coil_complex_sos = np.sqrt(np.sum(np.abs((input_12coil_complex_img)**2), axis=2))
        
        # ori
        ori_complex = np.zeros([256, 256, 12], dtype=np.complex64)
        for i in range(0, 24, 2):
            ori_complex[:, :, int(i/2)] = target_channel24_mri_real_imag[:, :, i] + 1j*target_channel24_mri_real_imag[:, :, i+1]
        # kspace to image
        ori_complex_img = np.zeros(ori_complex.shape, dtype=np.complex64)
        for i in range(ori_complex.shape[-1]):
            ori_complex_img[:, :, i] = np.fft.ifft2(np.fft.fftshift(ori_complex[:, :, i]))
        ori_complex_sos = np.sqrt(np.sum(np.abs((ori_complex_img)**2), axis=2))
        
        # for
        pred_12_complex = np.zeros([256, 256, 12], dtype=np.complex64)
        for i in range(0, 24, 2):
            pred_12_complex[:, :, int(i/2)] = pred_12_real_imag[:, :, i] + 1j*pred_12_real_imag[:, :, i+1]
        # kspace to image
        pred_12_complex_img = np.zeros(pred_12_complex.shape, dtype=np.complex64)
        for i in range(pred_12_complex.shape[-1]):
            pred_12_complex_img[:, :, i] = np.fft.ifft2(np.fft.fftshift(pred_12_complex[:, :, i]))
            
        pred_12_complex_img_1 = np.zeros([256, 256, 4], dtype=np.complex64)
        pred_12_complex_img_1 = pred_12_complex_img[:, :, 0:4]
        pred_12_complex_img_2 = np.zeros([256, 256, 4], dtype=np.complex64)
        pred_12_complex_img_2 = pred_12_complex_img[:, :, 4:8]
        pred_12_complex_img_3 = np.zeros([256, 256, 4], dtype=np.complex64)
        pred_12_complex_img_3 = pred_12_complex_img[:, :, 8:12]
        
        pred_12_complex_sos_1 = np.sqrt(np.sum(np.abs((pred_12_complex_img_1)**2), axis=2))
        pred_12_complex_sos_2 = np.sqrt(np.sum(np.abs((pred_12_complex_img_2)**2), axis=2))
        pred_12_complex_sos_3 = np.sqrt(np.sum(np.abs((pred_12_complex_img_3)**2), axis=2))
        pred_12_complex_sos = (pred_12_complex_sos_1 + pred_12_complex_sos_2 + pred_12_complex_sos_3) / 3
        
        plt.subplot(1 ,3, 1)
        plt.imshow(255*abs(np.rot90(ori_complex_sos, -1)) ,cmap='gray')
        plt.title("ori_sos")
        plt.subplot(1 ,3, 2)
        plt.imshow(255*abs(np.rot90(input_12coil_complex_sos, -1)) ,cmap='gray')
        plt.title("input_sos")
        plt.subplot(1 ,3, 3)
        plt.imshow(255*abs(np.rot90(pred_12_complex_sos, -1)) ,cmap='gray')
        plt.title("pred_sos")
        plt.show()
        
        psnr_compress = compare_psnr(255*abs(ori_complex_sos), 255*abs(pred_12_complex_sos), data_range=255)
        ssim_compress = compare_ssim(abs(ori_complex_sos), abs(pred_12_complex_sos), data_range=1)
        #print('psnr_forward:',psnr_compress,'    ssim_forward:',ssim_compress)
        PSNR_COMPRESS.append(psnr_compress)
        SSIM_COMPRESS.append(ssim_compress)

        del reconstruct_4_real_imag
    ave_psnr_compress = sum(PSNR_COMPRESS) / len(PSNR_COMPRESS)
    ave_ssim_compress = sum(SSIM_COMPRESS) / len(SSIM_COMPRESS)
    print("ave_psnr_forward: %.10f || ave_ssim_forward:%.10f"%(ave_psnr_compress, ave_ssim_compress))
    

if __name__ == '__main__':
    torch.set_num_threads(4)
    main(args)

