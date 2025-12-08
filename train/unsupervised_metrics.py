import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class SanityResult:
    """Per-image sanity metrics for a predicted karyotype."""
    total_score: float            # Overall sanity score (0..1)
    count_score: float            # Chromosome count sanity (near expected_total)
    class_balance_score: float    # 2 copies of each autosome
    sex_score: float              # Consistent XX / XY pattern
    size_score: float             # Similar chromosome sizes
    n_instances: int              # Total number of chromosomes
    class_counts: np.ndarray      # Counts per class (1..num_classes)
    sex_pattern: Optional[str]    # "XX", "XY" or None


class UnlabeledSanity:
    def __init__(
        self,
        num_classes: int = 24,
        autosome_ids: Optional[List[int]] = None,
        x_id: int = 23,
        y_id: int = 24,
        expected_autosome_copies: int = 2,
        expected_total: int = 46,
        size_tolerance: float = 3.0,
    ):
        self.num_classes = num_classes
        self.autosome_ids = autosome_ids or list(range(1, min(num_classes, 23)))
        self.x_id = x_id
        self.y_id = y_id
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
            return SanityResult(
                total_score=0.0,
                count_score=0.0,
                class_balance_score=0.0,
                sex_score=0.0,
                size_score=0.0,
                n_instances=0,
                class_counts=counts,
                sex_pattern=None,
            )

        counts = np.bincount(labels, minlength=self.num_classes + 1)
        counts = counts[1 : self.num_classes + 1]
        n_instances = int(counts.sum())

        count_error = abs(n_instances - self.expected_total)
        if self.expected_total > 0:
            count_score = 1.0 - count_error / self.expected_total
        else:
            count_score = 0.0
        count_score = self._clip01(count_score)

        autosome_scores = []
        for cid in self.autosome_ids:
            if 1 <= cid <= self.num_classes:
                c = counts[cid - 1]
                err = abs(c - self.expected_autosome_copies)
                if self.expected_autosome_copies > 0:
                    s = 1.0 - err / self.expected_autosome_copies
                else:
                    s = 0.0
                autosome_scores.append(self._clip01(s))

        class_balance_score = float(np.mean(autosome_scores)) if autosome_scores else 0.0

        # 4) Sex chromosome pattern sanity (XX / XY)
        cx = counts[self.x_id - 1] if 1 <= self.x_id <= self.num_classes else 0
        cy = counts[self.y_id - 1] if 1 <= self.y_id <= self.num_classes else 0

        patterns = {
            "XX": (2, 0), 
            "XY": (1, 1),  
        }

        best_pattern = None
        best_sex_score = 0.0

        for name, (ex_x, ex_y) in patterns.items():
            err = abs(cx - ex_x) + abs(cy - ex_y)
            s = 1.0 - err / 4.0
            s = self._clip01(s)
            if s > best_sex_score:
                best_sex_score = s
                best_pattern = name

        sex_score = best_sex_score
        sex_pattern = best_pattern

        if boxes is None or boxes.size == 0:
            size_score = 0.0
        else:
            boxes = boxes.reshape(-1, 4).astype(np.float32)
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            areas = w * h

            valid = areas > 0
            if not np.any(valid):
                size_score = 0.0
            else:
                areas = areas[valid]
                med = float(np.median(areas))
                if med <= 0:
                    size_score = 0.0
                else:
                    ratio = areas / med
                    good = (ratio >= 1.0 / self.size_tolerance) & (
                        ratio <= self.size_tolerance
                    )
                    size_score = float(np.mean(good))

        total_score = float(
            (count_score + class_balance_score + sex_score + size_score) / 4.0
        )

        return SanityResult(
            total_score=total_score,
            count_score=count_score,
            class_balance_score=class_balance_score,
            sex_score=sex_score,
            size_score=size_score,
            n_instances=n_instances,
            class_counts=counts,
            sex_pattern=sex_pattern,
        )

    def score_batch(self, preds: List[Dict[str, Any]]) -> Dict[str, Any]: results: List[SanityResult] = []

        for p in preds:
            r = self.score_image(p["boxes"], p["labels"])
            results.append(r)

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

        totals = np.array([r.total_score for r in results], dtype=np.float32)
        counts = np.array([r.count_score for r in results], dtype=np.float32)
        balances = np.array([r.class_balance_score for r in results], dtype=np.float32)
        sexes = np.array([r.sex_score for r in results], dtype=np.float32)
        sizes = np.array([r.size_score for r in results], dtype=np.float32)

        dataset_scores = {
            "total": float(totals.mean()),
            "count": float(counts.mean()),
            "class_balance": float(balances.mean()),
            "sex": float(sexes.mean()),
            "size": float(sizes.mean()),
        }

        return {
            "dataset_scores": dataset_scores,
            "per_image": results,
        }
