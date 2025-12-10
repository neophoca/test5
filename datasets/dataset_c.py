import os, glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F

from .base import resize_max


class DatasetC(Dataset): #ChromoCRCN
    """
    Dataset for data/Chromo-CRCN with YOLO-style labels.

    Folder structure:
      root/
        train/
          images/
          labels/
        val/
          images/
          labels/
        test/
          images/
          labels/

    Each label file:
      class cx cy w h
    in normalized image coordinates, class is always 0 (chromosome).
    """

    def __init__(self, root, split, max_size, num_channels):
        assert split in {"train", "val", "test"}
        self.root = root
        self.split = split
        self.max_size = max_size
        self.num_channels = num_channels

        self.img_dir = os.path.join(root, split, "images")
        self.lbl_dir = os.path.join(root, split, "labels")

        self.label_files = sorted(glob.glob(os.path.join(self.lbl_dir, "*.txt")))

    def __len__(self):
        return len(self.label_files)

    def _find_image_path(self, stem):
        exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
        for ext in exts:
            p = os.path.join(self.img_dir, stem + ext)
            if os.path.exists(p):
                return p
        return os.path.join(self.img_dir, stem + ".png")

    def __getitem__(self, i):
        lbl_path = self.label_files[i]
        stem = os.path.splitext(os.path.basename(lbl_path))[0]
        img_path = self._find_image_path(stem)

        img = Image.open(img_path).convert("L")
        w, h = img.size

        boxes = []
        labels = []

        with open(lbl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cls_id, cx, cy, bw, bh = map(float, line.split())

                cx *= w
                cy *= h
                bw *= w
                bh *= h

                x1 = cx - bw / 2.0
                y1 = cy - bh / 2.0
                x2 = cx + bw / 2.0
                y2 = cy + bh / 2.0

                boxes.append([x1, y1, x2, y2])

                labels.append(1)

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        img, boxes = resize_max(img, boxes, self.max_size)
        img_t = F.to_tensor(img) * 2 - 1
        if self.num_channels == 3:
            img_t = img_t.repeat(3, 1, 1)

        _, H, W = img_t.shape
        masks = torch.zeros((boxes.shape[0], H, W), dtype=torch.uint8)

        for j, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            x1 = int(max(min(x1.item(), W - 1), 0))
            x2 = int(max(min(x2.item(), W), 0))
            y1 = int(max(min(y1.item(), H - 1), 0))
            y2 = int(max(min(y2.item(), H), 0))
            if x2 > x1 and y2 > y1:
                masks[j, y1:y2, x1:x2] = 1

        target = {
            "boxes": boxes,
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": masks,
            "image_id": torch.tensor([i]),
        }
        return img_t, target
