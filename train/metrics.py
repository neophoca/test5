import numpy as np
from torchmetrics.detection import MeanAveragePrecision

__all__ = [
    "pairwise_iou_masks",
    "compute_map50",
    "compute_pq",
    "compute_aji",
    "eval_instance_seg",
]


def pairwise_iou_masks(g_masks, p_masks):
    """
    g_masks: (G,H,W) bool/0-1
    p_masks: (P,H,W) bool/0-1
    Returns:
      iou:   (G,P)
      inter: (G,P)
      union: (G,P)
    """
    if len(g_masks) == 0 or len(p_masks) == 0:
        shape = (len(g_masks), len(p_masks))
        zero = np.zeros(shape, dtype=np.float32)
        return zero, zero, zero

    g = g_masks.astype(bool).reshape(len(g_masks), -1).astype(np.uint8)
    p = p_masks.astype(bool).reshape(len(p_masks), -1).astype(np.uint8)

    inter = (g[:, None, :] & p[None, :, :]).sum(-1).astype(np.float32)
    union = (g[:, None, :] | p[None, :, :]).sum(-1).astype(np.float32)
    iou = inter / (union + 1e-7)
    return iou, inter, union


def compute_map50(preds, targets):
    """
    mAP@0.5
      pred: {"boxes": Tensor[N,4], "labels": Tensor[N], "scores": Tensor[N], ...}
      targ: {"boxes": Tensor[M,4], "labels": Tensor[M], ...}
    """
    metric = MeanAveragePrecision(
        iou_type="bbox",
        box_format="xyxy",
        iou_thresholds=[0.5],
    )

    preds_det = [
        {"boxes": p["boxes"], "scores": p["scores"], "labels": p["labels"]}
        for p in preds
    ]
    targets_det = [
        {"boxes": t["boxes"], "labels": t["labels"]}
        for t in targets
    ]

    metric.update(preds_det, targets_det)
    res = metric.compute()
    return float(res["map"])


def compute_pq(preds, targets, num_classes, iou_thresh=0.5):
    """
    Panoptic Quality (PQ) and mean PQ over classes (mPQ).

      pred["masks"]: np.uint8[N,H,W], pred["labels"]: Tensor[N]
      gt["masks"]:   np.uint8[M,H,W], gt["labels"]:   Tensor[M]
    """
    C = num_classes
    iou_sum = np.zeros(C + 1, dtype=np.float64)
    tp = np.zeros(C + 1, dtype=np.float64)
    fp = np.zeros(C + 1, dtype=np.float64)
    fn = np.zeros(C + 1, dtype=np.float64)

    for pred, gt in zip(preds, targets):
        if "masks" not in pred or "masks" not in gt:
            continue

        g_masks = gt["masks"].astype(bool)
        p_masks = pred["masks"].astype(bool)
        g_labels = gt["labels"].numpy()
        p_labels = pred["labels"].numpy()

        for c in range(1, C + 1):
            g_idx = np.where(g_labels == c)[0]
            p_idx = np.where(p_labels == c)[0]
            if len(g_idx) == 0 and len(p_idx) == 0:
                continue

            Gc = g_masks[g_idx] if len(g_idx) > 0 else np.zeros((0,) + g_masks.shape[1:], bool)
            Pc = p_masks[p_idx] if len(p_idx) > 0 else np.zeros((0,) + p_masks.shape[1:], bool)

            if len(Gc) == 0:
                fp[c] += len(Pc)
                continue
            if len(Pc) == 0:
                fn[c] += len(Gc)
                continue

            iou_mat, _, _ = pairwise_iou_masks(Gc, Pc)

            pairs = [
                (iou_mat[gi, pj], gi, pj)
                for gi in range(len(Gc))
                for pj in range(len(Pc))
                if iou_mat[gi, pj] >= iou_thresh
            ]
            pairs.sort(reverse=True, key=lambda x: x[0])

            used_g, used_p = set(), set()
            iou_sum_c, tp_c = 0.0, 0

            for iou_val, gi, pj in pairs:
                if gi in used_g or pj in used_p:
                    continue
                used_g.add(gi)
                used_p.add(pj)
                iou_sum_c += float(iou_val)
                tp_c += 1

            iou_sum[c] += iou_sum_c
            tp[c] += tp_c
            fp[c] += len(Pc) - tp_c
            fn[c] += len(Gc) - tp_c

    pq_per_class = np.zeros(C + 1, dtype=np.float64)
    for c in range(1, C + 1):
        denom = tp[c] + 0.5 * fp[c] + 0.5 * fn[c]
        if denom > 0:
            pq_per_class[c] = iou_sum[c] / (denom + 1e-7)
        else:
            pq_per_class[c] = np.nan

    valid = ~np.isnan(pq_per_class[1:])
    mPQ = float(pq_per_class[1:][valid].mean()) if valid.any() else float("nan")

    denom_all = tp.sum() + 0.5 * fp.sum() + 0.5 * fn.sum()
    PQ_all = float(iou_sum.sum() / (denom_all + 1e-7)) if denom_all > 0 else float("nan")

    return PQ_all, mPQ, pq_per_class[1:] 


def compute_aji(preds, targets):
    """
    Aggregated Jaccard Index (AJI) over all images.
      pred["masks"]: np.uint8[N,H,W]
      targ["masks"]: np.uint8[M,H,W]
    """
    num_aji = 0.0
    den_aji = 0.0

    for pred, gt in zip(preds, targets):
        if "masks" not in pred or "masks" not in gt:
            continue

        gm = gt["masks"].astype(bool)
        pm = pred["masks"].astype(bool)

        Ng, Np = gm.shape[0], pm.shape[0]
        if Ng == 0 and Np == 0:
            continue

        gt_areas = gm.reshape(Ng, -1).sum(1) if Ng > 0 else np.zeros(0)
        pred_areas = pm.reshape(Np, -1).sum(1) if Np > 0 else np.zeros(0)

        iou_mat, inter_mat, union_mat = pairwise_iou_masks(gm, pm)

        matched_pred = set()
        num_img, den_img = 0.0, 0.0

        for gi in range(Ng):
            if Np > 0:
                pj = int(np.argmax(iou_mat[gi]))
                inter = inter_mat[gi, pj]
                union = union_mat[gi, pj]
                if inter > 0:
                    num_img += float(inter)
                    den_img += float(union)
                    matched_pred.add(pj)
                else:
                    den_img += float(gt_areas[gi])
            else:
                den_img += float(gt_areas[gi])

        for pj in range(Np):
            if pj not in matched_pred:
                den_img += float(pred_areas[pj])

        num_aji += num_img
        den_aji += den_img

    AJI = float(num_aji / (den_aji + 1e-7)) if den_aji > 0 else float("nan")
    return AJI


def get_metrics(preds, targets, num_classes, iou_thresh=0.5):
    mAP50 = compute_map50(preds, targets)
    PQ_all, mPQ, pq_per_class = compute_pq(preds, targets, num_classes, iou_thresh=iou_thresh)
    AJI = compute_aji(preds, targets)

    return {
        "mAP50": mAP50,
        "PQ_all": PQ_all,
        "mPQ": mPQ,
        "PQ_per_class": pq_per_class,
        "AJI": AJI,
    }
