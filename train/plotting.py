import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage import measure
from train.metrics import pairwise_iou_masks


def _mask_to_box(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.array([0, 0, 0, 0], dtype=np.float32)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def _center(mask, box):
    if mask is not None and mask.any():
        ys, xs = np.where(mask)
        return float(xs.mean()), float(ys.mean())
    x1, y1, x2, y2 = box
    return float((x1 + x2) * 0.5), float((y1 + y2) * 0.5)


def _overlay_green(ax, mask, alpha=0.25):
    if mask is None:
        return
    rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
    rgba[mask, 0] = 0.0
    rgba[mask, 1] = 1.0
    rgba[mask, 2] = 0.0
    rgba[mask, 3] = alpha
    ax.imshow(rgba)


def _draw_contours(ax, mask, lw=2.0):
    if mask is None or not mask.any():
        return
    cs = measure.find_contours(mask.astype(np.uint8), 0.5)
    for c in cs:
        ax.plot(c[:, 1], c[:, 0], linewidth=lw)


def plot_predictions_no_gt(dataset, model, device, n=4, score_thresh=0.5, use_random=True):
    model.eval()
    n = min(n, len(dataset))
    idxs = random.sample(range(len(dataset)), n) if use_random else list(range(n))

    fig, axes = plt.subplots(1, n, figsize=(14 * n, 14))
    if n == 1:
        axes = [axes]

    for ax, idx in zip(axes, idxs):
        img_t, _ = dataset[idx]
        img = (img_t * 0.5 + 0.5).permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        H, W = img.shape[:2]
        ax.imshow(img)

        out = model([img_t.to(device)])[0]
        scores = out["scores"].detach().cpu().numpy()
        keep = scores >= score_thresh

        if keep.sum() > 0:
            pmasks = (out["masks"][keep, 0] > 0.5).detach().cpu().numpy().astype(bool)
            pboxes = out["boxes"][keep].detach().cpu().numpy().astype(np.float32)
            plabels = out["labels"][keep].detach().cpu().numpy().astype(int)
        else:
            pmasks = np.zeros((0, H, W), dtype=bool)
            pboxes = np.zeros((0, 4), dtype=np.float32)
            plabels = np.zeros((0,), dtype=int)

        for p in range(len(pmasks)):
            _overlay_green(ax, pmasks[p], alpha=0.20)
            _draw_contours(ax, pmasks[p], lw=2.0)

            x1, y1, x2, y2 = pboxes[p]
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2))

            ys, xs = np.where(pmasks[p])
            if len(xs) > 0:
                cx, cy = float(xs.mean()), float(ys.mean())
            else:
                cx, cy = float((x1 + x2) * 0.5), float((y1 + y2) * 0.5)

            ax.text(
                cx, cy, f"P{plabels[p]} {scores[keep][p]:.2f}",
                color="white", fontsize=7, ha="center", va="center",
                bbox=dict(facecolor="black", alpha=0.6, linewidth=0),
            )

        ax.set_title(f"idx {idx}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_predictions(dataset, model, device, n=4, score_thresh=0.5, iou_thresh=0.5, use_random=True):
    model.eval()
    n = min(n, len(dataset))
    idxs = random.sample(range(len(dataset)), n) if use_random else list(range(n))

    fig, axes = plt.subplots(1, n, figsize=(14 * n, 14))
    if n == 1:
        axes = [axes]

    for ax, idx in zip(axes, idxs):
        img_t, target = dataset[idx]
        img = (img_t * 0.5 + 0.5).permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        H, W = img.shape[:2]
        ax.imshow(img)

        gt_masks = target["masks"].cpu().numpy().astype(bool)
        gt_labels = target["labels"].cpu().numpy().astype(int)
        if "boxes" in target:
            gt_boxes = target["boxes"].cpu().numpy().astype(np.float32)
        else:
            gt_boxes = np.stack([_mask_to_box(m) for m in gt_masks], axis=0) if len(gt_masks) else np.zeros((0, 4), np.float32)

        out = model([img_t.to(device)])[0]
        scores = out["scores"].detach().cpu().numpy()
        keep = scores >= score_thresh

        pmasks = np.zeros((0, H, W), dtype=bool)
        pboxes = np.zeros((0, 4), dtype=np.float32)
        plabels = np.zeros((0,), dtype=int)

        if keep.sum() > 0:
            pmasks = (out["masks"][keep, 0] > 0.5).detach().cpu().numpy().astype(bool)
            pboxes = out["boxes"][keep].detach().cpu().numpy().astype(np.float32)
            plabels = out["labels"][keep].detach().cpu().numpy().astype(int)

        G, P = len(gt_masks), len(pmasks)

        gt_to_p = -np.ones(G, dtype=int)
        p_to_gt = -np.ones(P, dtype=int)

        if G > 0 and P > 0:
            iou, _, _ = pairwise_iou_masks(gt_masks, pmasks)
            pairs = [(float(iou[g, p]), g, p) for g in range(G) for p in range(P)]
            pairs.sort(reverse=True, key=lambda x: x[0])

            used_g, used_p = set(), set()
            for v, g, p in pairs:
                if v < iou_thresh:
                    break
                if g in used_g or p in used_p:
                    continue
                used_g.add(g)
                used_p.add(p)
                gt_to_p[g] = p
                p_to_gt[p] = g

        for g in range(G):
            p = gt_to_p[g]
            if p >= 0:
                _overlay_green(ax, pmasks[p], alpha=0.25)
                cx, cy = _center(pmasks[p], pboxes[p])
                ax.text(cx, cy, f"GT{gt_labels[g]}/P{plabels[p]}",color="white", fontsize=7, ha="center", va="center",bbox=dict(facecolor="black", alpha=0.6, linewidth=0))
            else:
                #_overlay_green(ax, gt_masks[g], alpha=0.0)
                x1, y1, x2, y2 = gt_boxes[g]
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2))
                cx, cy = _center(gt_masks[g], gt_boxes[g])
                ax.text(cx, cy, f"FN",color="red", fontsize=9, ha="center", va="center",bbox=dict(facecolor="black", alpha=0.6, linewidth=0),)

        for p in range(P):
            if p_to_gt[p] >= 0:
                continue
            _overlay_green(ax, pmasks[p], alpha=0.25)
            x1, y1, x2, y2 = pboxes[p]
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="blue", linewidth=2))
            cx, cy = _center(pmasks[p], pboxes[p])
            ax.text(cx, cy, f"FP",color="red", fontsize=9, ha="center", va="center",bbox=dict(facecolor="black", alpha=0.6, linewidth=0),)

        ax.set_title(f"idx {idx}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()
