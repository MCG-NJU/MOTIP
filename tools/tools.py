# Copyright (c) Ruopeng Gao. All Rights Reserved.

import matplotlib.pyplot as plt

from torchvision import transforms


def visualize_a_batch(images, annotations):
    for i in range(len(images)):
        image = transforms.ToPILImage()(images[i])
        fig, ax = plt.subplots(1)
        boxes = annotations[i]["bbox"]
        for box in boxes:
            img_w, img_h = image.size
            w, h = box[2].item(), box[3].item()
            x, y = (box[0].item() - w / 2), (box[1].item() - h / 2)
            x, y, w, h = x * img_w, y * img_h, w * img_w, h * img_h
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=1)
            ax.add_patch(rect)
        plt.imshow(image)
        plt.show()
    pass