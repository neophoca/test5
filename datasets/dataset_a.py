import os
import glob
import csv
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F

from .base import resize_max


def _read_txt_ids(txt_path):
    with open(txt_path, "r") as f:
        return {line.strip() for line in f if line.strip()}


def _read_csv_ids(csv_path):
    ids = set()
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ids.add(os.path.splitext(row["filename"].strip())[0])
    return ids


class DatasetA(Dataset):
    def __init__(self, xml_dir, img_dir, label_map, max_size, num_channels, with_masks=True):
        self.ann_files = sorted(glob.glob(os.path.join(xml_dir, "*.xml")))
        self.img_dir = img_dir
        self.label_map = label_map
        self.max_size = max_size
        self.num_channels = num_channels
        self.with_masks = with_masks

        self.file_ids = []
        for p in self.ann_files:
            root = ET.parse(p).getroot()
            fname = root.find("filename").text.strip()
            self.file_ids.append(os.path.splitext(fname)[0])

    def indices_from_txt(self, txt_path):
        ids = _read_txt_ids(txt_path)
        return [i for i, fid in enumerate(self.file_ids) if fid in ids]

    def difficult_indices(self, diff_txt_path):
        return self.indices_from_txt(diff_txt_path)

    def abnormal_indices(self, kind, normal_csv, number_csv, structural_csv):
        if kind == "normal":
            ids = _read_csv_ids(normal_csv)
        elif kind == "number":
            ids = _read_csv_ids(number_csv)
        elif kind == "structural":
            ids = _read_csv_ids(structural_csv)
        else:
            ids = set()
        return [i for i, fid in enumerate(self.file_ids) if fid in ids]

    def __len__(self):
        return len(self.ann_files)

    def __getitem__(self, i):
        xml_path = self.ann_files[i]
        root = ET.parse(xml_path).getroot()
        fname = root.find("filename").text.strip()

        boxes, labels = [], []
        for obj in root.findall("object"):
            name = obj.find("name").text.strip()
            lab = self.label_map[name]
            bnd = obj.find("bndbox")
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(lab)

        boxes = torch.tensor(boxes, dtype=torch.float32)

        with Image.open(os.path.join(self.img_dir, fname)) as im:
            img = im.convert("L")

        img, boxes = resize_max(img, boxes, self.max_size)
        img_t = F.to_tensor(img) * 2 - 1
        if self.num_channels == 3:
            img_t = img_t.repeat(3, 1, 1)

        _, H, W = img_t.shape
        target = {"boxes": boxes, "labels": torch.tensor(labels, dtype=torch.int64), "image_id": torch.tensor([i])}

        if self.with_masks:
            # A is boxes-only; rectangle masks are the minimal mask target. They are not needed for training even just in case. 
            masks = torch.zeros((boxes.shape[0], H, W), dtype=torch.uint8)
            for j, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                x1 = int(max(min(x1.item(), W - 1), 0))
                x2 = int(max(min(x2.item(), W), 0))
                y1 = int(max(min(y1.item(), H - 1), 0))
                y2 = int(max(min(y2.item(), H), 0))
                if x2 > x1 and y2 > y1:
                    masks[j, y1:y2, x1:x2] = 1
            target["masks"] = masks

        return img_t, target
