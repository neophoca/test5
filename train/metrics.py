import numpy as np
from torchmetrics.detection import MeanAveragePrecision

__all__ = [
    "pairwise_iou_masks",
    "compute_map50",
    "compute_pq",
    "compute_aji",
    "eval_instance_seg",
    "get_metrics",
    "get_metrics_split_size",
]


def pairwise_iou_masks(g_masks, p_masks):
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



def aggregated_jaccard(gt_masks: np.ndarray, pred_masks: np.ndarray) -> float:
    """
    Aggregated Jaccard Index (AJI) on *lists of instance masks*.

    Inputs:
      gt_masks:  (Ng, H, W) binary/boolean masks for GT instances
      pred_masks:(Np, H, W) binary/boolean masks for predicted instances

    Output (as in LongChen sortedAP aggregatedJaccard):
      - compute intersection matrix between every pred and GT
      - for each GT: pick the pred with maximum intersection (can repeat preds)
      - C = sum of intersections of picked pairs where intersection > 0
      - U = sum(|GT|) + sum(|Pred[picked where inter>0]|) - C + sum(|Pred[unpicked]|)
      - AJI = C / U  (with the same empty-set handling)
    """
    # ---- wrapper: get_metrics() calls aggregated_jaccard(preds, targets) ----
    if isinstance(gt_masks, list) and (len(gt_masks) == 0 or isinstance(gt_masks[0], dict)):
        preds = gt_masks
        targets = pred_masks

        C_total = 0.0
        U_total = 0.0

        for pd, gt in zip(preds, targets):
            if "masks" not in pd or "masks" not in gt:
                continue

            g = gt["masks"]
            p = pd["masks"]

            if hasattr(g, "detach"):
                g = g.detach().cpu().numpy()
            if hasattr(p, "detach"):
                p = p.detach().cpu().numpy()

            g = np.asarray(g)
            p = np.asarray(p)

            if g.ndim == 4 and g.shape[1] == 1:
                g = g[:, 0]
            if p.ndim == 4 and p.shape[1] == 1:
                p = p[:, 0]

            # Ensure boolean for correct logical ops.
            g = g.astype(bool)
            p = p.astype(bool)

            Ng = int(g.shape[0])
            Np = int(p.shape[0])

            if Ng == 0 and Np == 0:
                continue
            if Ng == 0 or Np == 0:
                U_total += float(max(g.reshape(Ng, -1).sum() if Ng > 0 else 0.0,
                                    p.reshape(Np, -1).sum() if Np > 0 else 0.0))
                continue

            gt_areas = g.reshape(Ng, -1).sum(1).astype(np.float64)
            pred_areas = p.reshape(Np, -1).sum(1).astype(np.float64)

            # reuse existing helper
            _, inter_mat, _ = pairwise_iou_masks(g, p)  # (Ng, Np)
            intersection = inter_mat.T  # (Np, Ng)

            idx = np.argmax(intersection, axis=0)  # per GT: best pred by intersection (includes 0-intersection picks)
            inter_best = intersection[idx, np.arange(Ng)]
            idx_e = inter_best > 0

            idx_pd = idx[idx_e]
            idx_gt = np.arange(Ng)[idx_e]

            C = float(intersection[idx_pd, idx_gt].sum()) if idx_pd.size > 0 else 0.0
            unmatched = list(set(range(Np)) - set(idx.tolist()))
            U = float(gt_areas.sum() + pred_areas[idx_pd].sum() - C + pred_areas[unmatched].sum())

            C_total += C
            U_total += U

        if U_total == 0.0:
            return 1.0 if C_total == 0.0 else 0.0
        return float(C_total / U_total)

    # ---- direct call on (Ng,H,W) / (Np,H,W) stacks ----
    if hasattr(gt_masks, "detach"):
        gt_masks = gt_masks.detach().cpu().numpy()
    if hasattr(pred_masks, "detach"):
        pred_masks = pred_masks.detach().cpu().numpy()

    gt_masks = np.asarray(gt_masks)
    pred_masks = np.asarray(pred_masks)

    if gt_masks.ndim == 4 and gt_masks.shape[1] == 1:
        gt_masks = gt_masks[:, 0]
    if pred_masks.ndim == 4 and pred_masks.shape[1] == 1:
        pred_masks = pred_masks[:, 0]

    # Ensure boolean for correct logical ops.
    gt_masks = gt_masks.astype(bool)
    pred_masks = pred_masks.astype(bool)

    Ng = int(gt_masks.shape[0])
    Np = int(pred_masks.shape[0])

    if Ng == 0 and Np == 0:
        return 1.0
    if Ng == 0 or Np == 0:
        U = float(max(gt_masks.reshape(Ng, -1).sum() if Ng > 0 else 0.0,
                      pred_masks.reshape(Np, -1).sum() if Np > 0 else 0.0))
        return 1.0 if U == 0.0 else 0.0

    gt_areas = gt_masks.reshape(Ng, -1).sum(1).astype(np.float64)
    pred_areas = pred_masks.reshape(Np, -1).sum(1).astype(np.float64)

    # reuse existing helper
    _, inter_mat, _ = pairwise_iou_masks(gt_masks, pred_masks)  # (Ng, Np)
    intersection = inter_mat.T  # (Np, Ng)

    idx = np.argmax(intersection, axis=0)
    inter_best = intersection[idx, np.arange(Ng)]
    idx_e = inter_best > 0

    idx_pd = idx[idx_e]
    idx_gt = np.arange(Ng)[idx_e]

    C = float(intersection[idx_pd, idx_gt].sum()) if idx_pd.size > 0 else 0.0
    unmatched = list(set(range(Np)) - set(idx.tolist()))
    U = float(gt_areas.sum() + pred_areas[idx_pd].sum() - C + pred_areas[unmatched].sum())

    if U == 0.0:
        return 1.0 if C == 0.0 else 0.0
    return float(C / U)



def compute_aji(preds, targets):
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
    #PQ_all, mPQ, pq_per_class = compute_pq(preds, targets, num_classes, iou_thresh=iou_thresh)
    AJI = aggregated_jaccard(preds, targets)

    return {
        "mAP50": mAP50,
        "PQ_all": PQ_all,
        "mPQ": mPQ,
        "PQ_per_class": pq_per_class,
        "AJI": AJI,
    }


def _areas_from_masks_or_boxes(entry):
    if "masks" in entry and entry["masks"] is not None:
        m = entry["masks"]
        if isinstance(m, np.ndarray):
            if m.size == 0:
                return np.zeros((0,), dtype=np.float32)
            mm = m.astype(bool).reshape(m.shape[0], -1)
            return mm.sum(1).astype(np.float32)
        else:
            if m.numel() == 0:
                return np.zeros((0,), dtype=np.float32)
            mm = (m > 0).flatten(1)
            return mm.sum(1).detach().cpu().numpy().astype(np.float32)

    if "boxes" in entry and entry["boxes"] is not None:
        b = entry["boxes"]
        if isinstance(b, np.ndarray):
            if b.size == 0:
                return np.zeros((0,), dtype=np.float32)
            w = (b[:, 2] - b[:, 0]).astype(np.float32)
            h = (b[:, 3] - b[:, 1]).astype(np.float32)
            return (w * h).astype(np.float32)
        else:
            if b.numel() == 0:
                return np.zeros((0,), dtype=np.float32)
            w = (b[:, 2] - b[:, 0]).detach().cpu().numpy().astype(np.float32)
            h = (b[:, 3] - b[:, 1]).detach().cpu().numpy().astype(np.float32)
            return (w * h).astype(np.float32)

    return np.zeros((0,), dtype=np.float32)


def _filter_entry(entry, keep_idx):
    out = dict(entry)
    for k in ["masks", "boxes", "labels", "scores"]:
        if k not in out or out[k] is None:
            continue
        v = out[k]
        if isinstance(v, np.ndarray):
            out[k] = v[keep_idx]
        else:
            out[k] = v[keep_idx]
    return out


def _greedy_match_iou(iou_mat, iou_thresh):
    G, P = iou_mat.shape
    gt_to_p = -np.ones(G, dtype=int)
    p_to_gt = -np.ones(P, dtype=int)

    pairs = [
        (float(iou_mat[g, p]), g, p)
        for g in range(G)
        for p in range(P)
        if iou_mat[g, p] >= iou_thresh
    ]
    pairs.sort(reverse=True, key=lambda x: x[0])

    used_g, used_p = set(), set()
    for v, g, p in pairs:
        if g in used_g or p in used_p:
            continue
        used_g.add(g)
        used_p.add(p)
        gt_to_p[g] = p
        p_to_gt[p] = g

    return gt_to_p, p_to_gt


def get_metrics_split_size(preds, targets, num_classes, iou_thresh=0.5, area_thresh=None, area_quantile=0.5):
    overall = get_metrics(preds, targets, num_classes, iou_thresh=iou_thresh)

    if area_thresh is None:
        all_gt_areas = []
        for t in targets:
            a = _areas_from_masks_or_boxes(t)
            if a.size:
                all_gt_areas.append(a)
        area_thresh = 0.0 if len(all_gt_areas) == 0 else float(np.quantile(np.concatenate(all_gt_areas), area_quantile))

    preds_short, preds_long = [], []
    targs_short, targs_long = [], []

    for p, t in zip(preds, targets):
        pa = _areas_from_masks_or_boxes(p)
        ta = _areas_from_masks_or_boxes(t)

        Ng = len(ta)
        Np = len(pa)

        gt_bucket = np.zeros(Ng, dtype=np.int64)
        gt_bucket[ta >= area_thresh] = 1  # 0=short, 1=long

        pred_bucket = -np.ones(Np, dtype=np.int64)

        if Ng > 0 and Np > 0 and ("masks" in p) and ("masks" in t):
            gm = t["masks"].astype(bool)
            pm = p["masks"].astype(bool)
            iou_mat, _, _ = pairwise_iou_masks(gm, pm)
            gt_to_p, p_to_gt = _greedy_match_iou(iou_mat, iou_thresh)

            for gi in range(Ng):
                pj = gt_to_p[gi]
                if pj >= 0:
                    pred_bucket[pj] = gt_bucket[gi]

        # unmatched preds -> bucket by their own area
        if Np > 0:
            unassigned = pred_bucket < 0
            pred_bucket[unassigned & (pa < area_thresh)] = 0
            pred_bucket[unassigned & (pa >= area_thresh)] = 1

        t_idx_s = np.where(gt_bucket == 0)[0]
        t_idx_l = np.where(gt_bucket == 1)[0]
        p_idx_s = np.where(pred_bucket == 0)[0]
        p_idx_l = np.where(pred_bucket == 1)[0]

        preds_short.append(_filter_entry(p, p_idx_s))
        preds_long.append(_filter_entry(p, p_idx_l))
        targs_short.append(_filter_entry(t, t_idx_s))
        targs_long.append(_filter_entry(t, t_idx_l))

    short = get_metrics(preds_short, targs_short, num_classes, iou_thresh=iou_thresh)
    long = get_metrics(preds_long, targs_long, num_classes, iou_thresh=iou_thresh)

    return {
        **overall,
        "area_thresh": float(area_thresh),
        "short": short,
        "long": long,
    }