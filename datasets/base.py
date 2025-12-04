import os
from PIL import Image


def resize_max(img, boxes, max_size):
    w, h = img.size
    s = min(max_size / w, max_size / h)
    if s != 1.0:
        ow, oh = int(round(w * s)), int(round(h * s))
        img = img.resize((ow, oh), Image.BILINEAR)
        boxes = boxes * s
    return img, boxes


def norm_label(s: str):
    s = str(s).strip()
    if s in ("-1", "119", "", "NA"):
        return None
    if s in ("23", "X", "x"):
        return "X"
    if s in ("24", "Y", "y"):
        return "Y"
    return s


def collate_bb(batch):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)
