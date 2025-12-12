import os, glob, xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F

from .base import resize_max


class DatasetA(Dataset):
    def __init__(self, xml_dir, img_dir, label_map, max_size, num_channels, with_masks=True):
        self.ann_files = sorted(glob.glob(os.path.join(xml_dir, "*.xml")))
        self.img_dir = img_dir
        self.label_map = label_map
        self.max_size = max_size
        self.num_channels = num_channels
        self.with_masks = with_masks

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


        target = {
            "boxes": boxes,
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([i]),
        }
        
        if self.with_masks:
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
