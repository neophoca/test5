from typing import Any, Dict, List
import torch
import numpy as np
from .metrics import get_metrics, compute_map50, get_metrics_split_size
from .plotting import plot_predictions
from .unsupervised_metrics import UnlabeledSanity


class Evaluator:
    def __init__(self, device: torch.device):
        self.device = device

    def collect_masks(self, model, loader):
        preds, targs = [], []
        model.eval()
        with torch.no_grad():
            for images, targets in loader:
                images = [img.to(self.device) for img in images]
                outputs = model(images)
                for out, gt in zip(outputs, targets):
                    out = {k: v.cpu() for k, v in out.items()}
                    gt = {k: v.cpu() for k, v in gt.items()}
                    pred = {"boxes": out["boxes"], "scores": out["scores"], "labels": out["labels"]}
                    targ = {"boxes": gt["boxes"], "labels": gt["labels"]}
                    if "masks" in out:
                        m = out["masks"]
                        if m.ndim == 4: m = m[:, 0]
                        pred["masks"] = (m > 0.5).byte().numpy()
                    if "masks" in gt:
                        targ["masks"] = gt["masks"].numpy().astype(np.uint8)
                    preds.append(pred)
                    targs.append(targ)
        return preds, targs

    def collect_boxes(self, model, loader):
        preds, targs = [], []
        model.eval()
        with torch.no_grad():
            for images, targets in loader:
                images = [img.to(self.device) for img in images]
                outputs = model(images)
                for out, gt in zip(outputs, targets):
                    preds.append(
                        {
                            "boxes": out["boxes"].cpu(),
                            "scores": out["scores"].cpu(),
                            "labels": out["labels"].cpu(),
                        }
                    )
                    targs.append(
                        {
                            "boxes": gt["boxes"].cpu(),
                            "labels": gt["labels"].cpu(),
                        }
                    )
        return preds, targs

    def collect_for_sanity(self, model, loader):
        preds = []
        model.eval()
        with torch.no_grad():
            for images, _ in loader:
                images = [img.to(self.device) for img in images]
                outputs = model(images)
                for out in outputs:
                    preds.append(
                        {
                            "boxes": out["boxes"].detach().cpu().numpy(),
                            "labels": out["labels"].detach().cpu().numpy(),
                        }
                    )
        return preds

    def metrics_masks(self, model, loader, num_classes) -> Dict[str, Any]:
        preds, targs = self.collect_masks(model, loader)
        return get_metrics(preds, targs, num_classes=num_classes)

    def metrics_masks_split(self, model, loader, num_classes) -> Dict[str, Any]:
        preds, targs = self.collect_masks(model, loader)
        return get_metrics_split_size(preds, targs, num_classes=num_classes)

    def map50(self, model, loader) -> float:
        preds, targs = self.collect_boxes(model, loader)
        return compute_map50(preds, targs)

    def sanity(self, model, loader, sanity: UnlabeledSanity | None = None) -> Dict[str, Any]:
        sanity = sanity or UnlabeledSanity()
        preds = self.collect_for_sanity(model, loader)
        return sanity.score_batch(preds)


    @staticmethod
    def print_table(rows: List[List[str]]):
        w = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
        for r in rows:
            print("  ".join(s.ljust(wc) for s, wc in zip(r, w)))

    def show_examples(self, dataset, model, n=1, score_thresh=0.5, iou_thresh=0.5, title="", show_random =False):
        print(f"\nExamples: {title}")
        plot_predictions(dataset, model, self.device, n=n, score_thresh=score_thresh, iou_thresh = iou_thresh, use_random = show_random)
