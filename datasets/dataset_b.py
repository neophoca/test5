import os, glob, json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
from .base import resize_max, norm_label


class DatasetB(Dataset):
    def __init__(self, root, split, label_map, max_size, num_channels):
        assert split in {"train", "test", "all"}
        self.root = root
        self.split = split
        self.label_map = label_map
        self.max_size = max_size
        self.num_channels = num_channels

        json_files = []
        if split in {"train", "all"}:
            json_files += glob.glob(
                os.path.join(root, "train_labelme", "**", "*.json"), recursive=True
            )
        if split in {"test", "all"}:
            json_files += glob.glob(
                os.path.join(root, "test_labelme", "**", "*.json"), recursive=True
            )

        self.json_files = sorted(json_files)

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, i):
        jpath = self.json_files[i]
        ann = json.load(open(jpath, encoding="utf-8"))

        img_name = ann["imagePath"]
        img_dir = os.path.dirname(jpath)
        img_path = os.path.join(img_dir, img_name)

        with Image.open(img_path) as im:
            img = im.convert("L")
        w0, h0 = img.size

        boxes, labels, polys = [], [], []
        for sh in ann.get("shapes", []):
            lab = norm_label(sh.get("label", ""))
            if lab is None:
                continue
            lab_id = self.label_map[lab]
            pts = [(float(x), float(y)) for x, y in sh["points"]]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            boxes.append([min(xs), min(ys), max(xs), max(ys)])
            labels.append(lab_id)
            polys.append(pts)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        img, boxes = resize_max(img, boxes, self.max_size)
        img_t = F.to_tensor(img) * 2 - 1
        if self.num_channels == 3:
            img_t = img_t.repeat(3, 1, 1)     
        _, H, W = img_t.shape

        # scale polygons with same factor as resize_max
        s = min(self.max_size / w0, self.max_size / h0)

        import numpy as np
        from PIL import ImageDraw

        masks = torch.zeros((len(polys), H, W), dtype=torch.uint8)
        for j, pts in enumerate(polys):
            pts_arr = np.array(pts, dtype=np.float32) * s
            mask_img = Image.new("L", (W, H), 0)
            ImageDraw.Draw(mask_img).polygon(
                [tuple(p) for p in pts_arr], outline=1, fill=1
            )
            masks[j] = torch.from_numpy(np.array(mask_img, dtype=np.uint8))

        target = {
            "boxes": boxes,
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": masks,
            "image_id": torch.tensor([i]),
        }
        return img_t, target

