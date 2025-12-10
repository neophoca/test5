import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class SanityResult:
    total_score: float
    count_score: float
    class_balance_score: float
    sex_score: float
    size_score: float
    n_instances: int
    class_counts: np.ndarray
    sex_pattern: Optional[str]


class UnlabeledSanity:
    def __init__(
        self,
        num_classes = 24,
        autosome_ids: Optional[List[int]] = None,
        x_id = 23,
        y_id = 24,
        expected_autosome_copies = 2,
        expected_total = 46,
        size_tolerance = 3.0,
    ):
        self.num_classes = num_classes
        self.autosome_ids = autosome_ids or list(range(1, min(num_classes, 23)))
        self.x_id, self.y_id = x_id, y_id
        self.expected_autosome_copies = expected_autosome_copies
        self.expected_total = expected_total
        self.size_tolerance = size_tolerance

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, x))

    def score_image(self, boxes: np.ndarray, labels: np.ndarray) -> SanityResult:
        labels = labels.astype(np.int64)
        if labels.size == 0:
            counts = np.zeros(self.num_classes, dtype=np.int64)
            return SanityResult(0.0, 0.0, 0.0, 0.0, 0.0, 0, counts, None)

        counts = np.bincount(labels, minlength=self.num_classes + 1)[1 : self.num_classes + 1]
        n_instances = int(counts.sum())

        # count ~ 46
        err = abs(n_instances - self.expected_total)
        count_score = self._clip01(1.0 - err / self.expected_total)

        # autosomes ~2 copies each
        autosome_scores = []
        for cid in self.autosome_ids:
            c = counts[cid - 1]
            e = abs(c - self.expected_autosome_copies)
            autosome_scores.append(self._clip01(1.0 - e / self.expected_autosome_copies))
        class_balance_score = float(np.mean(autosome_scores))

        cx, cy = counts[self.x_id - 1], counts[self.y_id - 1]
        patterns = {"XX": (2, 0), "XY": (1, 1)}
        best_pattern, best_sex_score = None, 0.0
        for name, (ex_x, ex_y) in patterns.items():
            e = abs(cx - ex_x) + abs(cy - ex_y)
            s = self._clip01(1.0 - e / 4.0)
            if s > best_sex_score:
                best_sex_score, best_pattern = s, name
        sex_score, sex_pattern = best_sex_score, best_pattern

        # box sizes
        if boxes.size == 0:
            size_score = 0.0
        else:
            boxes = boxes.reshape(-1, 4).astype(np.float32)
            w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
            areas = (w * h)[(w > 0) & (h > 0)]
            if areas.size == 0:
                size_score = 0.0
            else:
                med = float(np.median(areas))
                ratio = areas / med
                good = (ratio >= 1.0 / self.size_tolerance) & (ratio <= self.size_tolerance)
                size_score = float(np.mean(good))

        total_score = float((count_score + class_balance_score + sex_score + size_score) / 4.0)

        return SanityResult(
            total_score,
            count_score,
            class_balance_score,
            sex_score,
            size_score,
            n_instances,
            counts,
            sex_pattern,
        )

    def score_batch(self, preds: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = [self.score_image(p["boxes"], p["labels"]) for p in preds]
        if not results:
            return {
                "dataset_scores": {
                    "total": 0.0,
                    "count": 0.0,
                    "class_balance": 0.0,
                    "sex": 0.0,
                    "size": 0.0,
                },
                "per_image": [],
            }

        def arr(name): return np.array([getattr(r, name) for r in results], dtype=np.float32)

        totals = arr("total_score")
        counts = arr("count_score")
        balances = arr("class_balance_score")
        sexes = arr("sex_score")
        sizes = arr("size_score")

        dataset_scores = {
            "total": float(totals.mean()),
            "count": float(counts.mean()),
            "class_balance": float(balances.mean()),
            "sex": float(sexes.mean()),
            "size": float(sizes.mean()),
        }
        return {"dataset_scores": dataset_scores, "per_image": results}
