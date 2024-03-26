# Copyright (c) RuopengGao. All Rights Reserved.
# About: Split the MOT17 dataset into train and val splits.

import os


def split_train_seq(src_dir: str, tgt_dir: str, seq_name: str):
    frame_paths = os.listdir(os.path.join(src_dir, "train", seq_name, "img1"))
    frame_paths.sort()
    frame_names = [_[:-4] for _ in frame_paths]
    train_frame_names = frame_names[:len(frame_names)//2]
    val_frame_names = frame_names[len(frame_names)//2:]

    # 创建需要输出的文件夹
    os.makedirs(os.path.join(tgt_dir, "train", seq_name, "img1"), exist_ok=True)
    os.makedirs(os.path.join(tgt_dir, "train", seq_name, "gt"), exist_ok=True)
    os.makedirs(os.path.join(tgt_dir, "train", seq_name, "img1"), exist_ok=True)
    os.makedirs(os.path.join(tgt_dir, "val", seq_name, "img1"), exist_ok=True)
    os.makedirs(os.path.join(tgt_dir, "val", seq_name, "gt"), exist_ok=True)
    os.makedirs(os.path.join(tgt_dir, "val", seq_name, "img1"), exist_ok=True)

    split = int(train_frame_names[-1])

    os.system(f"cp {os.path.join(src_dir, 'train', seq_name, 'seqinfo.ini')} "
              f"{os.path.join(tgt_dir, 'train', seq_name, 'seqinfo.ini')}")
    os.system(f"cp {os.path.join(src_dir, 'train', seq_name, 'seqinfo.ini')} "
              f"{os.path.join(tgt_dir, 'val', seq_name, 'seqinfo.ini')}")
    for frame in train_frame_names:
        os.system(f"cp {os.path.join(src_dir, 'train', seq_name, 'img1', frame + '.jpg')} "
                  f"{os.path.join(tgt_dir, 'train', seq_name, 'img1', frame + '.jpg')}")

    for frame in val_frame_names:
        val_frame = f"{str(int(frame) - split).zfill(6)}"
        os.system(f"cp {os.path.join(src_dir, 'train', seq_name, 'img1', frame + '.jpg')} "
                  f"{os.path.join(tgt_dir, 'val', seq_name, 'img1', val_frame + '.jpg')}")

    train_gts = []
    val_gts = []
    with open(os.path.join(src_dir, "train", seq_name, "gt", "gt.txt"), "r") as gt_file:
        lines = gt_file.readlines()
        for line in lines:
            line = (line[:-1].split(","))
            frame_idx = int(line[0])
            if frame_idx > split:   # in val split
                frame_idx = frame_idx - split
                line[0] = str(frame_idx)
                val_gts.append(line)
            else:                   # in train split
                train_gts.append(line)

    with open(os.path.join(tgt_dir, "train", seq_name, "gt", "gt.txt"), "a") as train_gt_file:
        for line in train_gts:
            gt = ""
            for i in range(len(line) - 1):
                gt += line[i]
                gt += ","
            gt += line[-1] + "\n"
            train_gt_file.write(gt)
    with open(os.path.join(tgt_dir, "val", seq_name, "gt", "gt.txt"), "a") as val_gt_file:
        for line in val_gts:
            gt = ""
            for i in range(len(line) - 1):
                gt += line[i]
                gt += ","
            gt += line[-1] + "\n"
            val_gt_file.write(gt)
    return


if __name__ == '__main__':
    mot17_dir = "/data0/DataForMeMOTRv2/MOT17"
    split_mot17_dir = "/data0/DataForMeMOTRv2/MOT17_SPLIT/"

    os.makedirs(split_mot17_dir, exist_ok=True)

    train_seq_names = os.listdir(os.path.join(mot17_dir, "train"))
    test_seq_names = os.listdir(os.path.join(mot17_dir, "test"))

    os.makedirs(os.path.join(split_mot17_dir), exist_ok=True)
    # os.makedirs(os.path.join(split_mot17_dir, "gts"), exist_ok=True)
    os.system(f"cp -r {os.path.join(mot17_dir, 'test')} {os.path.join(split_mot17_dir)}")
    # os.system(f"cp -r {os.path.join(mot17_dir, 'gts', 'test')} {os.path.join(split_mot17_dir, 'gts')}")

    for train_seq in train_seq_names:
        if "DPM" in train_seq:
            print(f"Processing Seq: '{train_seq}'.")
            split_train_seq(src_dir=mot17_dir, tgt_dir=split_mot17_dir, seq_name=train_seq)
