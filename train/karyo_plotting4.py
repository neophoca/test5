import math
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from datasets.cfg import class_colors


def extract_chromosome_instances(img_t, out, target, score_thresh=0.5, pad=5, iou_thresh=0.3):
    img = img_t * 0.5 + 0.5
    img = img.clamp(0, 1)

    _, H, W = img.shape

    boxes  = out["boxes"].detach().cpu()
    labels = out["labels"].detach().cpu()
    scores = out["scores"].detach().cpu()
    pmasks = (out["masks"][:, 0] > 0.5).detach().cpu()  # [N,H,W] bool

    keep = scores >= score_thresh
    boxes, labels, scores, pmasks = boxes[keep], labels[keep], scores[keep], pmasks[keep]

    if "masks" in target and target["masks"].numel() > 0:
        gt_masks  = target["masks"].cpu().bool()
        gt_labels = target["labels"].cpu()
    else:
        gt_masks  = torch.empty(0, H, W, dtype=torch.bool)
        gt_labels = torch.empty(0, dtype=torch.long)

    instances = []
    for box, plab, sc, p_mask in zip(boxes, labels, scores, pmasks):
        x1, y1, x2, y2 = box.tolist()
        x1 = max(int(x1) - pad, 0)
        y1 = max(int(y1) - pad, 0)
        x2 = min(int(x2) + pad, W)
        y2 = min(int(y2) + pad, H)
        if x2 <= x1 or y2 <= y1:
            continue

        crop_img = TF.crop(img, top=y1, left=x1, height=y2 - y1, width=x2 - x1)

        m = p_mask.float().unsqueeze(0)  # [1,H,W]
        m_crop = TF.crop(m, top=y1, left=x1, height=y2 - y1, width=x2 - x1)[0] > 0.5

        gt_lab = None
        if gt_masks.numel() > 0:
            inter = (gt_masks & p_mask).sum(dim=(1, 2)).float()
            union = (gt_masks | p_mask).sum(dim=(1, 2)).float()
            ious  = inter / (union + 1e-6)
            best_iou, best_idx = ious.max(dim=0)
            if best_iou.item() >= iou_thresh:
                gt_lab = int(gt_labels[best_idx])

        instances.append(
            {
                "img": crop_img,          # [C,h,w]
                "mask": m_crop,          # [h,w] bool
                "plab": int(plab),
                "gt_lab": gt_lab,
                "score": float(sc),
            }
        )

    return instances


def pad_instances(instances):
    hs = [x["img"].shape[1] for x in instances]
    ws = [x["img"].shape[2] for x in instances]
    max_h = max(hs)
    max_w = max(ws)

    padded = []
    for inst in instances:
        img  = inst["img"]
        mask = inst["mask"].float().unsqueeze(0)  # [1,h,w]

        _, h, w = img.shape
        dh = max_h - h
        dw = max_w - w
        top = dh // 2
        bottom = dh - top
        left = dw // 2
        right = dw - left

        pad_tuple = (left, top, right, bottom)  # L,T,R,B
        img_p  = TF.pad(img, pad_tuple)
        mask_p = TF.pad(mask, pad_tuple)[0] > 0.5

        inst_p = dict(inst)
        inst_p["img"] = img_p
        inst_p["mask"] = mask_p
        padded.append(inst_p)

    return padded


def plot_karyogram_from_output(img_t, out, target, idx=0, score_thresh=0.5, pad=5):
    instances = extract_chromosome_instances(img_t.cpu(), out, target, score_thresh, pad)
    if not instances:
        print("No chromosomes found.")
        return

    instances = pad_instances(instances)

    groups = defaultdict(list)
    for inst in instances:
        groups[inst["plab"]].append(inst)

    for lab in groups:
        groups[lab].sort(key=lambda x: -x["score"])

    labels_sorted = sorted(groups.keys())
    rows = len(labels_sorted)
    cols = 2  # pairs

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    if rows == 1:
        axes = np.array([axes])
    axes = axes.reshape(rows, cols)

    for ax in axes.ravel():
        ax.axis("off")

    for r, lab in enumerate(labels_sorted):
        pair = groups[lab][:2]
        for c in range(cols):
            ax = axes[r, c]
            if c >= len(pair):
                ax.axis("off")
                continue

            inst = pair[c]
            img  = inst["img"].permute(1, 2, 0).cpu().numpy()
            mask = inst["mask"].cpu().numpy().astype(bool)

            ax.imshow(img)

            overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
            color = class_colors.get(inst["plab"], (1.0, 0.0, 0.0))
            overlay[mask, 0] = color[0]
            overlay[mask, 1] = color[1]
            overlay[mask, 2] = color[2]
            overlay[mask, 3] = 0.4
            ax.imshow(overlay)

            gt_str = "-" if inst["gt_lab"] is None else str(inst["gt_lab"])
            ax.text(
                3,
                3,
                f"P:{inst['plab']}  GT:{gt_str}\nS:{inst['score']:.2f}",
                color="white",
                fontsize=8,
                ha="left",
                va="top",
                bbox=dict(facecolor="black", alpha=0.7, linewidth=0),
            )

            ax.set_title(f"chr {lab}", fontsize=10)
            ax.axis("off")

    plt.tight_layout()
    plt.show()

