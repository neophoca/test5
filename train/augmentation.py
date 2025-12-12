import random
import torch
import torchvision.transforms.functional as F

HFLIP_P   = 0.5
VFLIP_P   = 0.5
BRIGHTNESS = 0.2
CONTRAST   = 0.2


def _update_target(target, boxes, masks):
    new_target = dict(target)
    new_target["boxes"] = boxes
    if masks is not None:
        new_target["masks"] = masks
    return new_target


def _flip_boxes_h(boxes, w):
    boxes = boxes.clone()
    x1 = boxes[:, 0].clone()
    x2 = boxes[:, 2].clone()
    boxes[:, 0] = w - x2
    boxes[:, 2] = w - x1
    return boxes


def _flip_boxes_v(boxes, h):
    boxes = boxes.clone()
    y1 = boxes[:, 1].clone()
    y2 = boxes[:, 3].clone()
    boxes[:, 1] = h - y2
    boxes[:, 3] = h - y1
    return boxes

def _hflip(img, target):
    _, h, w = img.shape
    masks = target.get("masks")

    img = torch.flip(img, dims=[2])
    boxes = _flip_boxes_h(target["boxes"], w)

    if masks is not None:
        masks = masks.flip(-1)

    return img, _update_target(target, boxes, masks)


def _vflip(img, target):
    _, h, w = img.shape
    masks = target.get("masks")

    img = torch.flip(img, dims=[1])
    boxes = _flip_boxes_v(target["boxes"], h)

    if masks is not None:
        masks = masks.flip(-2)

    return img, _update_target(target, boxes, masks)


def _rand_factor(delta):
    return 1.0 + (2.0 * random.random() - 1.0) * delta


def _random_brightness_contrast(img):
    b = _rand_factor(BRIGHTNESS)
    c = _rand_factor(CONTRAST)
    img = F.adjust_brightness(img, b)
    img = F.adjust_contrast(img, c)
    return img


def augment_sample(img, target):
    if random.random() < HFLIP_P:
        img, target = _hflip(img, target)

    if random.random() < VFLIP_P:
        img, target = _vflip(img, target)

    img = _random_brightness_contrast(img)
    return img, target


def augment_batch(images, targets):
    out_images, out_targets = [], []
    for img, tgt in zip(images, targets):
        img2, tgt2 = augment_sample(img, tgt)
        out_images.append(img2)
        out_targets.append(tgt2)
    return out_images, out_targets
