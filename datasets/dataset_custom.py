import glob
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as F


def _resize_max(img_pil, boxes_t, masks_t, max_size: int):
    w, h = img_pil.size
    m = max(w, h)
    if max_size <= 0 or m <= max_size:
        return img_pil, boxes_t, masks_t

    s = float(max_size) / float(m)
    nw, nh = int(round(w * s)), int(round(h * s))

    img2 = img_pil.resize((nw, nh), resample=Image.BILINEAR)

    boxes2 = boxes_t.clone()
    if boxes2.numel():
        boxes2[:, 0::2] *= s
        boxes2[:, 1::2] *= s

    if masks_t.numel():
        masks2 = torch.nn.functional.interpolate(
            masks_t[:, None].float(),
            size=(nh, nw),
            mode="nearest",
        )[:, 0].byte()
    else:
        masks2 = masks_t

    return img2, boxes2, masks2

#Small custom labeled dataset only contains instance masks, no chromosome IDs
class DatasetCustom(torch.utils.data.Dataset):
    def __init__(self, root, max_size=640, num_channels=3, normalize_minus1_1=True):
        self.root = Path(root)
        self.max_size = int(max_size)
        self.num_channels = int(num_channels)
        self.normalize_minus1_1 = bool(normalize_minus1_1)

        self.img_dir = self.root / "images"
        self.ann_dir = self.root / "ann"

        self.ids = sorted(Path(p).stem for p in glob.glob(str(self.ann_dir / "*.json")))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img_id = self.ids[i]

        ann = json.loads((self.ann_dir / f"{img_id}.json").read_text(encoding="utf-8"))
        H = int(ann["height"])
        W = int(ann["width"])

        img = Image.open(self.img_dir / f"{img_id}.png").convert("L")

        boxes = []
        masks = []

        for inst in ann.get("instances", []):
            rel = inst["mask"]
            bx, by, bw, bh = map(int, inst["bbox_xywh"])

            crop = np.array(Image.open(self.root / rel).convert("L"), dtype=np.uint8)
            crop = (crop > 0).astype(np.uint8)

            x0 = max(0, bx)
            y0 = max(0, by)
            x1 = min(W, bx + bw)
            y1 = min(H, by + bh)
            if x1 <= x0 or y1 <= y0:
                continue

            cw = x1 - x0
            ch = y1 - y0
            crop = crop[:ch, :cw]
            if crop.size == 0 or crop.max() == 0:
                continue

            m = np.zeros((H, W), dtype=np.uint8)
            m[y0:y1, x0:x1] = crop

            ys, xs = np.where(m > 0)
            if xs.size == 0:
                continue

            x1b, y1b = float(xs.min()), float(ys.min())
            x2b, y2b = float(xs.max() + 1), float(ys.max() + 1)

            boxes.append([x1b, y1b, x2b, y2b])
            masks.append(m)

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            masks_t = torch.tensor(np.stack(masks, 0), dtype=torch.uint8)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            masks_t = torch.zeros((0, H, W), dtype=torch.uint8)

        labels_t = torch.ones((boxes_t.shape[0],), dtype=torch.int64)

        img, boxes_t, masks_t = _resize_max(img, boxes_t, masks_t, self.max_size)

        img_t = F.to_tensor(img)
        if self.normalize_minus1_1:
            img_t = img_t * 2 - 1
        if self.num_channels == 3:
            img_t = img_t.repeat(3, 1, 1)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks_t,
            "image_id": torch.tensor([i], dtype=torch.int64),
            "image_id_str": img_id,
        }
        return img_t, target
