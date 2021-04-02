'''
Test Vid4 (SR) and REDS4 (SR-clean, SR-blur, deblur-clean, deblur-compression) datasets
'''

import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch
from collections import OrderedDict

import utils.util as util
import data.util as data_util
import models.archs.EDVR_arch as EDVR_arch

import argparse


def main():

    # Create object for parsing command-line options
    parser = argparse.ArgumentParser(description="Test with EDVR, require path to test dataset folder.")
    parser.add_argument("-i", "--input", type=str, help="Path to test folder")
    parser.add_argument("-m", "--model_mode", type=str, default="reds_m", help="model mode")
    parser.add_argument("-p", "--model_path", type=str,  help="model path")
    # Parse the command line arguments to an object
    args = parser.parse_args()

    # Safety if no parameter have been given
    if args.input is None or args.model_path is None:
        print("No input paramater have been given.")
        print("For help type --help")
        exit()

    folder_name = args.input.split("/")[-1]
    # if folder_name == '':
    #     index = len(args.input.split("/")) - 2
    #     folder_name = args.input.split("/")[index]
    model_mode = args.model_mode # reds_m | reds_l | vimeo
    model_path = args.model_path

    #################
    # configurations
    #################
    device = torch.device('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
     
    stage = 1  # 1 or 2, use two stage strategy for REDS dataset.
    flip_test = False
    ############################################################################
    #### model
    # hyper params, default for reds_m model
    N_f = 64
    N_in = 5
    predeblur, HR_in = False, False
    back_RBs = 10

    # change for large model
    if model_mode == "vimeo" or model_mode == "reds_l":
        N_f = 128
        back_RBs = 40
        if model_mode == 'vimeo':
            N_in = 7  # use N_in images to restore one HR image
        else:
            N_in = 5

    if stage == 2:
        HR_in = True
        back_RBs = 20

    # create model    
    model = EDVR_arch.EDVR(N_f, N_in, 8, 5, back_RBs, predeblur=predeblur, HR_in=HR_in)

    #### dataset
    test_dataset_folder = os.path.join(args.input, 'val_d4')
    GT_dataset_folder = os.path.join(args.input, 'val_gt')

    #### evaluation
    crop_border = 0
    border_frame = N_in // 2  # border frames when evaluate
    # temporal padding mode
    padding = 'new_info'

    save_imgs = True

    save_folder = '../results/{}'.format(folder_name)
    util.mkdirs(save_folder)
    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')

    #### log info
    logger.info('Data: {} - {}'.format(folder_name, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip test: {}'.format(flip_test))

    #### set up the models
    print(model)
    model.load_state_dict(torch.load(model_path), strict=False)
    # load_net = torch.load(model_path)
    # load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    # for k, v in load_net.items():
    #     if k.startswith('module.'):
    #         load_net_clean[k[7:]] = v
    #     else:
    #         load_net_clean[k] = v
    # model.load_state_dict(load_net_clean, strict=True)

    model.eval()
    model = model.to(device)

    avg_psnr_l, avg_psnr_center_l, avg_psnr_border_l = [], [], []
    subfolder_name_l = []

    subfolder_l = sorted(glob.glob(osp.join(test_dataset_folder, '*')))
    subfolder_GT_l = sorted(glob.glob(osp.join(GT_dataset_folder, '*')))
    # for each subfolder
    for subfolder, subfolder_GT in zip(subfolder_l, subfolder_GT_l):

        subfolder_name = osp.basename(subfolder)
        subfolder_name_l.append(subfolder_name)
        save_subfolder = osp.join(save_folder, subfolder_name)

        img_path_l = sorted(glob.glob(osp.join(subfolder, '*')))
        max_idx = len(img_path_l)
        if save_imgs:
            util.mkdirs(save_subfolder)

        #### read LQ and GT images
        imgs_LQ = data_util.read_img_seq(subfolder)
        img_GT_l = []
        for img_GT_path in sorted(glob.glob(osp.join(subfolder_GT, '*'))):
            img_GT_l.append(data_util.read_img(None, img_GT_path))

        avg_psnr, avg_psnr_border, avg_psnr_center, N_border, N_center = 0, 0, 0, 0, 0

        # process each image
        for img_idx, img_path in enumerate(img_path_l):
            img_name = osp.splitext(osp.basename(img_path))[0]
            select_idx = data_util.index_generation(img_idx, max_idx, N_in, padding=padding)
            imgs_in = imgs_LQ.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)
            # imgs_in = imgs_LQ.index_select(0, torch.LongTensor(select_idx))
            # print(imgs_in.shape)

            if flip_test:
                output = util.flipx4_forward(model, imgs_in)
            else:
                output = util.single_forward(model, imgs_in)
            output = util.tensor2img(output.squeeze(0))

            if save_imgs:
                cv2.imwrite(osp.join(save_subfolder, '{}.png'.format(img_name)), output)

            # calculate PSNR
            output = output / 255.
            GT = np.copy(img_GT_l[img_idx])
            # For REDS, evaluate on RGB channels; for Vid4, evaluate on the Y channel
            if model_mode == 'vimeo':  # bgr2y, [0, 1]
                GT = data_util.bgr2ycbcr(GT, only_y=True)
                output = data_util.bgr2ycbcr(output, only_y=True)

            output, GT = util.crop_border([output, GT], crop_border)
            crt_psnr = util.calculate_psnr(output * 255, GT * 255)
            logger.info('{:3d} - {:25} \tPSNR: {:.6f} dB'.format(img_idx + 1, img_name, crt_psnr))

            if img_idx >= border_frame and img_idx < max_idx - border_frame:  # center frames
                avg_psnr_center += crt_psnr
                N_center += 1
            else:  # border frames
                avg_psnr_border += crt_psnr
                N_border += 1

        avg_psnr = (avg_psnr_center + avg_psnr_border) / (N_center + N_border)
        avg_psnr_center = avg_psnr_center / N_center
        avg_psnr_border = 0 if N_border == 0 else avg_psnr_border / N_border
        avg_psnr_l.append(avg_psnr)
        avg_psnr_center_l.append(avg_psnr_center)
        avg_psnr_border_l.append(avg_psnr_border)

        logger.info('Folder {} - Average PSNR: {:.6f} dB for {} frames; '
                    'Center PSNR: {:.6f} dB for {} frames; '
                    'Border PSNR: {:.6f} dB for {} frames.'.format(subfolder_name, avg_psnr,
                                                                   (N_center + N_border),
                                                                   avg_psnr_center, N_center,
                                                                   avg_psnr_border, N_border))

    logger.info('################ Tidy Outputs ################')
    for subfolder_name, psnr, psnr_center, psnr_border in zip(subfolder_name_l, avg_psnr_l,
                                                              avg_psnr_center_l, avg_psnr_border_l):
        logger.info('Folder {} - Average PSNR: {:.6f} dB. '
                    'Center PSNR: {:.6f} dB. '
                    'Border PSNR: {:.6f} dB.'.format(subfolder_name, psnr, psnr_center,
                                                     psnr_border))
    logger.info('################ Final Results ################')
    logger.info('Data: {} - {}'.format(folder_name, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip test: {}'.format(flip_test))
    logger.info('Total Average PSNR: {:.6f} dB for {} clips. '
                'Center PSNR: {:.6f} dB. Border PSNR: {:.6f} dB.'.format(
                    sum(avg_psnr_l) / len(avg_psnr_l), len(subfolder_l),
                    sum(avg_psnr_center_l) / len(avg_psnr_center_l),
                    sum(avg_psnr_border_l) / len(avg_psnr_border_l)))

if __name__ == '__main__':
    main()
