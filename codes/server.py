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

t1 = time.perf_counter()
print("time for import: " + str(t1-t0))

def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)

def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    A = np.frombuffer(buf, dtype=md['dtype'])
    return A.reshape(md['shape'])

def eval(images, N_in, model, device):
    images = images.astype(np.float32) / 255.
    images = torch.from_numpy(np.ascontiguousarray(np.transpose(images, (0, 3, 1, 2)))).float()
    imgs_in = images.index_select(0, torch.LongTensor(list(range(N_in)))).unsqueeze(0).to(device)
    output = util.single_forward(model, imgs_in)
    output = util.tensor2img(output.squeeze(0))
    output = np.ascontiguousarray(output)
    return output

def main():

    # Create object for parsing command-line options
    parser = argparse.ArgumentParser(description="Test with EDVR, require path to the pretrained model and the save_dir.")
    # Add argument which takes path to a bag file as an input
    parser.add_argument("-m", "--model", type=str, default='../experiments/pretrained_models/EDVR_Vimeo90K_SR_L.pth', help="Path to pretrained model")
    parser.add_argument("-s", "--save_dir", type=str, default='../results/LumoImg', help="Path to save_dir")

    # Parse the command line arguments to an object
    args = parser.parse_args()

    #################
    # configurations
    #################
    device = torch.device('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    ############################################################################
    #### model
    model_path = args.model
    N_in = 7  # use N_in images to restore one HR image
    predeblur, HR_in = False, False
    back_RBs = 40
    t0 = time.perf_counter()
    model = EDVR_arch.EDVR(128, N_in, 8, 5, back_RBs, predeblur=predeblur, HR_in=HR_in)
    t1 = time.perf_counter()
    print("time for initiating model: " + str(t1-t0))

    #### evaluation
    save_imgs = True
    img_name = "super_res"
    save_folder = args.save_dir
    util.mkdirs(save_folder)

    t0 = time.perf_counter()
    #### set up the models
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)
    t1 = time.perf_counter()
    print("time for loading model: " + str(t1-t0))

    ### server setting: if Address already in use => netstat -ltnp
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    print("Super Res Server is Listening:")
    while True:
        images = recv_array(socket)
        img_out = eval(images, N_in, model, device)
        send_array(socket, img_out)
        if save_imgs:
            cv2.imwrite(osp.join(save_folder, '{}.png'.format(img_name)), img_out)

if __name__ == '__main__':
    t_main_0 = time.perf_counter()
    main()
    t_main_1 = time.perf_counter()
    print("time for main: " + str(t_main_1-t_main_0))
