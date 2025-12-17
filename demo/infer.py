import os
import urllib.request
from io import BytesIO
import numpy as np
import torch
from PIL import Image
from demo.viz import render_overlay_pil, render_karyogram_pil
from models.models import build_model
from train.unsupervised_metrics import UnlabeledSanity
from datasets.cfg import num_classes as _NUM_CLASSES

def _resize_max_gray(pil_img: Image.Image, max_size: int) -> Image.Image:
    w, h = pil_img.size
    s = min(max_size / w, max_size / h)
    if s ==1:
        return pil_img
    ow, oh = int(round(w * s)), int(round(h * s))
    return pil_img.resize((ow, oh), Image.BILINEAR)

def _to_model_tensor(pil_img: Image.Image, num_channels: int) -> torch.Tensor:
    arr = np.array(pil_img, dtype=np.float32) / 255.0  
    t = torch.from_numpy(arr)[None, ...]          
    t = t * 2.0 - 1.0         
    if num_channels == 3: #i mean ofc 3
        t = t.repeat(3, 1, 1) #because maskrcnn takes 3
    return t

def _load_checkpoint_into(model: torch.nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
        else:
            sd = ckpt
    else:
        sd = ckpt
    model.load_state_dict(sd, strict=False)

class Predictor:
    def __init__(
        self,
        model_arch: str,
        weights_path: str,
        score_thresh: float,
    ):
        self.model_arch = model_arch
        self.weights_path = weights_path
        self.max_size = 640
        self.num_channels = 3
        self.score_thresh = score_thresh

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(model_arch, _NUM_CLASSES).to(self.device)
        _load_checkpoint_into(self.model, self.weights_path)
        self.model.eval()
        self.sanity = UnlabeledSanity(num_classes=24, x_id=23, y_id=24)



    @torch.no_grad()
    def predict(self, img_bytes: bytes):
        pil = Image.open(BytesIO(img_bytes)).convert("L")
        pil = _resize_max_gray(pil, self.max_size)

        img_t = _to_model_tensor(pil, self.num_channels).to(self.device)
        out = self.model([img_t])[0]

        scores = out["scores"].detach().cpu().numpy()
        keep = scores >= self.score_thresh

        boxes = out["boxes"][keep].detach().cpu().numpy().astype(np.float32)
        labels = out["labels"][keep].detach().cpu().numpy().astype(np.int64)
        masks = (out["masks"][keep, 0] > 0.5).detach().cpu().numpy().astype(bool)

        n_pred = int(labels.shape[0])

        sanity_res = self.sanity.score_image(boxes, labels)
        sex = sanity_res.sex_pattern or "?"

        x_count = int((labels == 23).sum())
        y_count = int((labels == 24).sum())

        overlay = render_overlay_pil(pil, boxes, labels, masks)
        karyogram = render_karyogram_pil(pil, boxes, labels, masks)

        return {
            "n_pred": n_pred,
            "sex": sex,
            "x_count": x_count,
            "y_count": y_count,
            "overlay_img": overlay,
            "karyogram_img": karyogram,
        }
