"""
Renames files from anyname to IMG_<>.jpg
"""
import os
import glob
import utils

def rename(img_fpath, img_dpath, gt_fpath, gt_dpath, scount, im_ext, gt_ext):

    img_path_list = [file for file in glob.glob(os.path.join(img_fpath,'*'+im_ext))]
    gt_path_list = []

    for img_path in img_path_list:
        fid = utils.get_file_id(img_path)
        gt_file_path = os.path.join(gt_fpath, fid + gt_ext)
        gt_path_list.append(gt_file_path)

    for i in range(0,len(img_path_list)):
        img_path = img_path_list[i]
        gt_path = gt_path_list[i]
        d_img_name = 'IMG_'+str(scount)+im_ext
        d_gt_name = 'GT_IMG_'+str(scount)+gt_ext
        scount += 1
        os.rename(img_path, os.path.join(img_dpath,d_img_name))
        os.rename(gt_path, os.path.join(gt_dpath, d_gt_name))

if __name__ == "__main__":
    img_fpath = '../data/ST_part_A/test_data/images'
    img_dpath = '../data/stech_2/images/'

    gt_fpath = '../data/ST_part_A/test_data/ground-truth'
    gt_dpath = '../data/stech_2/ground-truth'

    scount = 4579

    im_ext = '.jpg'
    gt_ext = '.mat'

    rename(img_fpath, img_dpath, gt_fpath, gt_dpath, scount, im_ext, gt_ext)