import random
import torch
import torchvision.transforms.functional as F

HFLIP_P = 0.5
VFLIP_P = 0.5
ROT90_P = 0.35

BRIGHTNESS = 0.2
CONTRAST = 0.2
GAMMA_P = 0.25
GAMMA_RANGE = (0.8, 1.25)

NOISE_P = 0.25
NOISE_STD = 0.03

BLUR_P = 0.15
BLUR_KERNEL = 3


def _update_target(target, boxes, masks):
    new_target = dict(target)
    new_target["boxes"] = boxes
    if masks is not None:
        new_target["masks"] = masks
    return new_target


def _hflip(img, target):
    _, H, W = img.shape
    boxes = target["boxes"].clone()
    x1 = boxes[:, 0].clone()
    x2 = boxes[:, 2].clone()
    boxes[:, 0] = W - x2
    boxes[:, 2] = W - x1
    masks = target.get("masks", None)
    if masks is not None:
        masks = torch.flip(masks, dims=[2])
    return torch.flip(img, dims=[2]), _update_target(target, boxes, masks)


def _vflip(img, target):
    _, H, W = img.shape
    boxes = target["boxes"].clone()
    y1 = boxes[:, 1].clone()
    y2 = boxes[:, 3].clone()
    boxes[:, 1] = H - y2
    boxes[:, 3] = H - y1
    masks = target.get("masks", None)
    if masks is not None:
        masks = torch.flip(masks, dims=[1])
    return torch.flip(img, dims=[1]), _update_target(target, boxes, masks)


def _rot90(img, target, k: int):
    k = int(k) % 4
    if k == 0:
        return img, target

    _, H, W = img.shape
    boxes = target["boxes"].clone()
    masks = target.get("masks", None)

    if masks is not None:
        masks = torch.rot90(masks, k=k, dims=[1, 2])
    img2 = torch.rot90(img, k=k, dims=[1, 2])

    def tf(x, y):
        if k == 1:
            return y, W - x
        if k == 2:
            return W - x, H - y
        return H - y, x

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    xs = torch.stack([x1, x2, x1, x2], dim=1)
    ys = torch.stack([y1, y1, y2, y2], dim=1)

    xsp = torch.empty_like(xs)
    ysp = torch.empty_like(ys)
    for j in range(4):
        xsp[:, j], ysp[:, j] = tf(xs[:, j], ys[:, j])

    boxes2 = torch.stack(
        [xsp.min(dim=1).values, ysp.min(dim=1).values, xsp.max(dim=1).values, ysp.max(dim=1).values],
        dim=1,
    )

    return img2, _update_target(target, boxes2, masks)


def _to_01(img):
    return (img + 1.0) * 0.5


def _to_m11(img01):
    return img01 * 2.0 - 1.0


def _photometric(img):
    img01 = _to_01(img).clamp(0.0, 1.0)
    if BRIGHTNESS and BRIGHTNESS > 0:
        b = 1.0 + random.uniform(-BRIGHTNESS, BRIGHTNESS)
        img01 = F.adjust_brightness(img01, b)
    if CONTRAST and CONTRAST > 0:
        c = 1.0 + random.uniform(-CONTRAST, CONTRAST)
        img01 = F.adjust_contrast(img01, c)
    if random.random() < GAMMA_P:
        g = random.uniform(GAMMA_RANGE[0], GAMMA_RANGE[1])
        img01 = img01.clamp(0.0, 1.0).pow(g)
    if random.random() < NOISE_P:
        img01 = (img01 + torch.randn_like(img01) * NOISE_STD).clamp(0.0, 1.0)
    if random.random() < BLUR_P:
        k = int(BLUR_KERNEL)
        if k % 2 == 0:
            k += 1
        img01 = F.gaussian_blur(img01, kernel_size=[k, k])
    return _to_m11(img01)


def augment_sample(img, target):
    if random.random() < HFLIP_P:
        img, target = _hflip(img, target)
    if random.random() < VFLIP_P:
        img, target = _vflip(img, target)
    if random.random() < ROT90_P:
        k = random.choice([1, 2, 3])
        img, target = _rot90(img, target, k)
    img = _photometric(img)
    return img, target


def augment_batch(images, targets):
    out_images, out_targets = [], []
    for img, tgt in zip(images, targets):
        img2, tgt2 = augment_sample(img, tgt)
        out_images.append(img2)
        out_targets.append(tgt2)
    return out_images, out_targets
