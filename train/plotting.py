import random
import numpy as np
import matplotlib.pyplot as plt
import torch

from datasets.cfg import class_colors


def plot_predictions(dataset, model, device, n=4, score_thresh=0.5, use_random=True):
    model.eval()
    if use_random:
        idxs = random.sample(range(len(dataset)), n)
    else:
        idxs = list(range(min(n, len(dataset))))

    fig, axes = plt.subplots(1, n, figsize=(15 * n, 15))
    if n == 1:
        axes = [axes]

    for ax, idx in zip(axes, idxs):
        img_t, target = dataset[idx]

        img = img_t * 0.5 + 0.5
        img = img.permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        H, W = img.shape[:2]
        ax.imshow(img)

        gt_masks = target["masks"].cpu().numpy().astype(bool)
        gt_labels = target["labels"].cpu().numpy()

        gt_overlay = np.zeros((H, W, 4), dtype=np.float32)
        for m, lab in zip(gt_masks, gt_labels):
            gt_overlay[m, 0:3] = 1.0
            gt_overlay[m, 3] = 0.3
            ys, xs = np.where(m)
            if len(xs) == 0:
                continue
            cx = xs.mean()
            cy = ys.mean()
            ax.text(
                cx,
                cy,
                f"GT {int(lab)}",
                color="black",
                fontsize=8,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.8, linewidth=0),
            )

        ax.imshow(gt_overlay)

        out = model([img_t.to(device)])[0]
        scores = out["scores"].detach().cpu()
        keep = scores >= score_thresh

        if keep.sum().item() > 0:
            pmasks = (out["masks"][keep, 0] > 0.5).detach().cpu().numpy()
            plabels = out["labels"][keep].detach().cpu().numpy()
            pscores = scores[keep].numpy()

            pred_overlay = np.zeros((H, W, 4), dtype=np.float32)

            for m, lab, sc in zip(pmasks, plabels, pscores):
                r, g, b = class_colors.get(int(lab), (1.0, 0.0, 0.0))
                pred_overlay[m, 0] = r
                pred_overlay[m, 1] = g
                pred_overlay[m, 2] = b
                pred_overlay[m, 3] = 0.35

                ys, xs = np.where(m)
                if len(xs) == 0:
                    continue
                cx = xs.mean()
                cy = ys.mean()
                ax.text(
                    cx,
                    cy,
                    f"P {int(lab)}\n{sc:.2f}",
                    color="white",
                    fontsize=8,
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="black", alpha=0.7, linewidth=0),
                )

            ax.imshow(pred_overlay)

        ax.set_title(f"idx {idx}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()
