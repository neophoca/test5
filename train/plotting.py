import random
import numpy as np
import matplotlib.pyplot as plt

def plot_predictions(dataset, model, device, n=4, score_thresh=0.5):
    model.eval()
    idxs = random.sample(range(len(dataset)), n)
    fig, axes = plt.subplots(1, n, figsize=(15 * n, 15))
    if n == 1:
        axes = [axes]

    cmap = plt.cm.get_cmap("tab20")

    for ax, idx in zip(axes, idxs):
        img_t, target = dataset[idx]

        # base image
        img = img_t * 0.5 + 0.5
        img = img.permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        H, W = img.shape[:2]
        ax.imshow(img)

        # -------- GT masks (white) + IDs ----------
        gt_masks = target["masks"].numpy().astype(bool)
        gt_labels = target["labels"].numpy()

        gt_overlay = np.zeros((H, W, 4), dtype=np.float32)
        for j, (m, lab) in enumerate(zip(gt_masks, gt_labels)):
            gt_overlay[m, 0:3] = 1.0  # white
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

        # -------- predictions: colored masks + IDs + scores ----------
        out = model([img_t.to(device)])[0]
        scores = out["scores"]
        keep = scores >= score_thresh

        if keep.sum() > 0:
            pmasks = (out["masks"][keep, 0] > 0.5).detach().cpu().numpy()
            plabels = out["labels"][keep].detach().cpu().numpy()
            pscores = scores[keep].detach().cpu().numpy()

            pred_overlay = np.zeros((H, W, 4), dtype=np.float32)

            for k, (m, lab, sc) in enumerate(zip(pmasks, plabels, pscores)):
                color = np.array(cmap(k % cmap.N))
                pred_overlay[m, 0:3] = color[:3]
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
