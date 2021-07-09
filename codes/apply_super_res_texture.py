'''
Test Vid4 (SR) and REDS4 (SR-clean, SR-blur, deblur-clean, deblur-compression) datasets
'''

import time
t0 = time.perf_counter()
import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch
import zmq
import argparse

import utils.util as util
import data.util as data_util
import models.archs.EDVR_arch as EDVR_arch


def eval(images, select_idx, model, device):
    # debug
    print("++++++++++++++++++++")
    images = images.astype(np.float32) / 255.
    images = torch.from_numpy(np.ascontiguousarray(np.transpose(images, (0, 3, 1, 2)))).float()
    print(torch.cuda.memory_allocated(0)/(1024*1024*1024))
    imgs_in = images.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)
    output = util.single_forward(model, imgs_in)
    print(torch.cuda.memory_allocated(0)/(1024*1024*1024))
    output = util.tensor2img(output.squeeze(0))
    output = np.ascontiguousarray(output)
    return output

def main():

    #################
    # configurations
    #################
    device = torch.device('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    save_imgs = True

    #### model
    model_path = "/home/weilunhuang/dermAPP/EDVR/experiments/pretrained_models/EDVR_Vimeo90K_SR_L.pth"
    N_in = 7  # use N_in images to restore one HR image
    N_f = 128 # 64
    predeblur, HR_in = False, False
    back_RBs = 40 # 10
    t0 = time.perf_counter()
    model = EDVR_arch.EDVR(N_f, N_in, 8, 5, back_RBs, predeblur=predeblur, HR_in=HR_in)

    #### set up the models
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()
    model = model.to(device)
    print("MODEL TO DEVICE~~~~~~~~~~~~`")
    print(torch.cuda.memory_allocated(0)/(1024*1024*1024))

    #### evaluation
    home_dir = "/home/weilunhuang/lumo_dermatoscope_app/temp_data"
    lesion_folders = os.listdir(home_dir)
    lesion_folders.sort()
    # crop_sizes = [(200, 150), (160, 120), (120, 120)] # radius of crop # (400, 300) is too large
    crop_sizes = [(160, 120)] # radius of crop # (400, 300) is too large

    for lesion_folder in lesion_folders:

        lesion_folder = os.path.join(home_dir, lesion_folder)
        for crop_size in crop_sizes:
            images = []
            crop_name = "crop_" + str(crop_size[0]*2) + "_" + str(crop_size[1]*2) + "_enhanced"
            crop_folder = os.path.join(lesion_folder, crop_name)
            save_folder_name = crop_name + "_sr_t"
            save_folder = os.path.join(lesion_folder, save_folder_name)
            util.mkdirs(save_folder)
            fnames = glob.glob(crop_folder + '/*.png')
            fnames.sort()

            for fname in fnames:
                img = cv2.imread(fname)
                images.append(img)
            images = np.array(images)
            max_idx = len(images)

            for img_idx in range(max_idx):
                select_idx = data_util.index_generation(img_idx, max_idx, N_in, padding="reflection")
                img_out = eval(images, select_idx, model, device)
                if save_imgs:
                    img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(osp.join(save_folder, '{}.png'.format(str(img_idx).zfill(3))), img_out)
                    # debug: check image downsampled back
                    width = int(img_out.shape[1] // 4)
                    height = int(img_out.shape[0] // 4)
                    dim = (width, height)
                    img_out_d4 = cv2.resize(img_out, dim, interpolation=cv2.INTER_LANCZOS4) # INTER_AREA, INTER_CUBIC, INTER_LANCZOS4
                    cv2.imwrite(osp.join(save_folder, '{}.png'.format(str(img_idx).zfill(3)+ "_d4")), img_out_d4)
                    # debug: check image downsampled back
                    width = int(img_out.shape[1] // 2)
                    height = int(img_out.shape[0] // 2)
                    dim = (width, height)
                    img_out_d2 = cv2.resize(img_out, dim, interpolation=cv2.INTER_LANCZOS4) # INTER_AREA, INTER_CUBIC, INTER_LANCZOS4
                    cv2.imwrite(osp.join(save_folder, '{}.png'.format(str(img_idx).zfill(3)+ "_d2")), img_out_d2)
        break
    print("MAX!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(torch.cuda.max_memory_allocated(0)/(1024*1024*1024))

if __name__ == '__main__':
    main()