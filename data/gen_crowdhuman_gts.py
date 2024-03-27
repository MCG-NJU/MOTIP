# @Author       : Ruopeng Gao
# @Date         : 2022/12/19
# @Description  :
# @Reference    :

import os.path as osp
import os
import cv2
import json


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


def load_func(fpath):
    print('fpath', fpath)
    assert os.path.exists(fpath)
    with open(fpath, 'r') as fid:
        lines = fid.readlines()
    records =[json.loads(line.strip('\n')) for line in lines]
    return records


def gen_labels_crowd(data_root, label_root, ann_root):
    mkdirs(label_root)
    anns_data = load_func(ann_root)

    tid_curr = 0
    for i, ann_data in enumerate(anns_data):
        print(i)
        image_name = '{}.jpg'.format(ann_data['ID'])
        img_path = os.path.join(data_root, image_name)
        anns = ann_data['gtboxes']
        img = cv2.imread(
            img_path,
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        img_height, img_width = img.shape[0:2]
        for i in range(len(anns)):
            if 'extra' in anns[i] and 'ignore' in anns[i]['extra'] and anns[i]['extra']['ignore'] == 1:
                continue
            x, y, w, h = anns[i]['fbox']
            # x += w / 2    # 保持 xywh 的格式，对齐 DanceTrack
            # y += h / 2
            label_fpath = img_path.replace('images', 'gts').replace('.png', '.txt').replace('.jpg', '.txt')
            label_str = '0 {:d} {:d} {:d} {:d} {:d}\n'.format(
                tid_curr, int(x), int(y), int(w), int(h))
            with open(label_fpath, 'a') as f:
                f.write(label_str)
            tid_curr += 1


if __name__ == '__main__':
    data_val = "./datasets/CrowdHuman/val/images"
    label_val = "./datasets/CrowdHuman/val/gts"
    ann_val = "./datasets/CrowdHuman/annotation_val.odgt"
    data_train = "./datasets/CrowdHuman/train/images"
    label_train = "./datasets/CrowdHuman/train/gts"
    ann_train = "./datasets/CrowdHuman/annotation_train.odgt"
    gen_labels_crowd(data_train, label_train, ann_train)
    gen_labels_crowd(data_val, label_val, ann_val)
