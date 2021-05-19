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
    N_in = 5  # use N_in images to restore one HR image
    N_f = 64 # 64
    predeblur, HR_in = False, False
    back_RBs = 10 # 19
    t0 = time.perf_counter()
    model = EDVR_arch.EDVR(N_f, N_in, 8, 5, back_RBs, predeblur=predeblur, HR_in=HR_in)
    t1 = time.perf_counter()
    print("time for initiating model M: " + str(t1-t0))

    #### evaluation
    save_imgs = True
    img_name = "super_res"
    save_folder = args.save_dir
    util.mkdirs(save_folder)

    t0 = time.perf_counter()
    #### set up the models
    # model.load_state_dict(torch.load(model_path), strict=True)
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()
    model = model.to(device)
    t1 = time.perf_counter()
    print("time for loading model: " + str(t1-t0))

    ### server setting: if Address already in use => netstat -ltnp
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    print("==============================")
    print("Super Res Server M is Listening:")
    flag = 'c' # c stands for continue
    while True:
        flag = socket.recv_string()
        if flag != 'c':
            socket.send_string("Closing model M is done")
            break
        else:
            socket.send_string("ok")
        images = recv_array(socket)
        img_out = eval(images, N_in, model, device)
        send_array(socket, img_out)
        if save_imgs:
            img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
            cv2.imwrite(osp.join(save_folder, '{}.png'.format(img_name + "_fidelity")), img_out)
            # debug: check image downsampled back
            width = int(img_out.shape[1] // 4)
            height = int(img_out.shape[0] // 4)
            dim = (width, height)
            img_out_d4 = cv2.resize(img_out, dim, interpolation=cv2.INTER_LANCZOS4) # INTER_AREA, INTER_CUBIC, INTER_LANCZOS4
            cv2.imwrite(osp.join(save_folder, '{}.png'.format(img_name + "_d_fidelity")), img_out_d4)

if __name__ == '__main__':
    main()