import cv2
import os
import glob
import multiprocessing
from multiprocessing import Pool, Manager

def downsample_worker(fname, save_dir, downby):
    img = cv2.imread(fname)
    w,h,channel = img.shape
    res = cv2.resize(img, (h//(downby), w//(downby)), interpolation=cv2.INTER_CUBIC)
    base_name = os.path.basename(fname)
    out_name = os.path.join(save_dir, base_name)
    cv2.imwrite(out_name, res)

def main():
    root_folder = "/data/huangw9/EDVR/datasets/REDS/"
    HR_folder = "val_sharp"
    LR_folder = "val_sharp_bicubic/X2"
    start_id = 0
    end_id = 30
    folder_ids = list(range(start_id, end_id))
    num_worker = multiprocessing.cpu_count()

    HR_dirs = []
    LR_dirs = []
    for folder_id in folder_ids:
        LR_dir = os.path.join(root_folder, LR_folder, str(folder_id).zfill(3))
        HR_dir = os.path.join(root_folder, HR_folder, str(folder_id).zfill(3))
        LR_dirs.append(LR_dir)
        HR_dirs.append(HR_dir)
        if not os.path.isdir(LR_dir):
            os.makedirs(LR_dir)
    for LR_dir, HR_dir in zip(LR_dirs, HR_dirs):
        fnames = glob.glob(HR_dir + "/*.png")
        pool = Pool(num_worker)
        for fname in fnames:
            pool.apply_async(downsample_worker, args=(fname, LR_dir, 2))
        pool.close()
        pool.join()

if __name__ == "__main__":
	main()
