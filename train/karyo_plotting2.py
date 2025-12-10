import math
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF


def extract_chromosome_crops(img_t, out, score_thresh=0.5, pad=5):
    img = img_t * 0.5 + 0.5
    img = img.clamp(0, 1)

    _, H, W = img.shape
    boxes = out["boxes"].detach().cpu().numpy()
    labels = out["labels"].detach().cpu().numpy()
    scores = out["scores"].detach().cpu().numpy()

    keep = scores >= score_thresh
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    crops = []
    for box, lab, sc in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        x1 = max(int(x1) - pad, 0)
        y1 = max(int(y1) - pad, 0)
        x2 = min(int(x2) + pad, W)
        y2 = min(int(y2) + pad, H)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = TF.crop(img, top=y1, left=x1, height=y2 - y1, width=x2 - x1)
        crops.append((crop, int(lab), float(sc)))

    return crops


def plot_karyogram_from_output(img_t, out, idx=0, score_thresh=0.5, pad=5, cols=8):
    crops = extract_chromosome_crops(img_t.cpu(), out, score_thresh, pad)
    if not crops:
        print("No chromosomes found.")
        return

    # sort by label
    crops.sort(key=lambda x: x[1])

    # find max H/W
    hs = [c[0].shape[1] for c in crops]
    ws = [c[0].shape[2] for c in crops]
    max_h = max(hs)
    max_w = max(ws)

    # pad all crops to (max_h, max_w), centered
    padded = []
    for crop, lab, sc in crops:
        _, h, w = crop.shape
        dh = max_h - h
        dw = max_w - w
        top = dh // 2
        bottom = dh - top
        left = dw // 2
        right = dw - left
        crop_pad = TF.pad(crop, (left, top, right, bottom))  # (L, T, R, B)
        padded.append((crop_pad, lab, sc))

    n = len(padded)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 3))
    if rows == 1:
        axes = np.array([axes])
    axes = axes.reshape(rows, cols)

    for ax in axes.ravel():
        ax.axis("off")

    for i, (crop, lab, sc) in enumerate(padded):
        r = i // cols
        c = i % cols
        ax = axes[r, c]

        img = crop.permute(1, 2, 0).cpu().numpy()
        ax.imshow(img)
        ax.set_title(f"{lab} ({sc:.2f})", fontsize=8)
        ax.axis("off")

    fig.suptitle(f"Karyogram-like layout, image idx {idx}", fontsize=12)
    plt.tight_layout()
    plt.show()
