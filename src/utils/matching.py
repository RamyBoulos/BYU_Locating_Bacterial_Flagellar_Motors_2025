
import sys
from typing import Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import logging
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
console_handler.setFormatter(formatter)
log.addHandler(console_handler)


@dataclass
class EvalResult:

    tp: int = 0
    fp: int = 0
    fn: int = 0

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0

    dist: np.ndarray = field(default_factory=lambda: np.array([]))
    mean_dist: float = 0.0

    panoptic_quality: float = 0.0

    false_negatives: Tuple[int, ...] = field(default_factory=tuple)
    false_positives: Tuple[int, ...] = field(default_factory=tuple)
    matched_pairs: Tuple[Tuple[int, int], ...] = field(default_factory=tuple)


def points_matching(
    p1,
    p2,
    cutoff_distance=3,
    eps=1e-8,
    class_label_p1: Optional[int] = None,
    class_label_p2: Optional[int] = None,
):
    """finds matching that minimizes sum of mean squared distances"""
    if class_label_p1 is not None:
        p1 = p1[p1[:, -1] == class_label_p1][:, :-1]
    elif class_label_p2 is not None:
        p2 = p2[p2[:, -1] == class_label_p2][:, :-1]

    assert p1.shape[1] == p2.shape[1], "Last dimensions of p1, p2 must match"

    if len(p1) == 0 or len(p2) == 0:
        D = np.zeros((0, 0))
    else:
        D = cdist(p1, p2, metric="sqeuclidean")

    if D.size > 0:
        D[D > cutoff_distance**2] = 1e10 * (1 + D.max())

    i, j = linear_sum_assignment(D)
    valid = D[i, j] <= cutoff_distance**2

    i, j = i[valid], j[valid]

    res = EvalResult()

    tp = len(i)
    fp = len(p2) - tp
    fn = len(p1) - tp
    res.tp = tp
    res.fp = fp
    res.fn = fn

    # when there is no tp and we dont predict anything the accuracy should be 1 not 0
    tp_eps = tp + eps

    res.accuracy = tp_eps / (tp_eps + fp + fn) if tp_eps > 0 else 0
    res.precision = tp_eps / (tp_eps + fp) if tp_eps > 0 else 0
    res.recall = tp_eps / (tp_eps + fn) if tp_eps > 0 else 0
    res.f1 = (2 * tp_eps) / (2 * tp_eps + fp + fn) if tp_eps > 0 else 0
    res.dist = np.sqrt(D[i, j])
    res.mean_dist = float(np.mean(res.dist)) if len(res.dist) > 0 else 0.0

    pq_num = np.sum(cutoff_distance - res.dist) / cutoff_distance
    pq_den = tp_eps + fp / 2 + fn / 2
    res.panoptic_quality = float(pq_num / pq_den) if tp_eps > 0 else 0.0

    res.false_negatives = tuple(set(range(len(p1))).difference(set(i)))
    res.false_positives = tuple(set(range(len(p2))).difference(set(j)))
    res.matched_pairs = tuple(zip(i, j))
    return res


def points_matching_dataset(
    p1s,
    p2s,
    cutoff_distance=3,
    by_image=True,
    eps=1e-8,
    class_label_p1: Optional[int] = None,
    class_label_p2: Optional[int] = None,
):
    """
    by_image is True -> metrics are computed by image and then averaged
    by_image is False -> TP/FP/FN are aggregated and only then are metrics computed
    """
    stats = tuple(
        points_matching(
            p1,
            p2,
            cutoff_distance=cutoff_distance,
            eps=eps,
            class_label_p1=class_label_p1,
            class_label_p2=class_label_p2,
        )
        for p1, p2 in zip(p1s, p2s)
    )

    if by_image:
        res: EvalResult = EvalResult()
        for k, v in vars(stats[0]).items():
            if np.isscalar(v):
                setattr(res, k, float(np.mean([vars(s)[k] for s in stats])))
        return res
    else:
        res = EvalResult()
        res.tp = 0
        res.fp = 0
        res.fn = 0

        for s in stats:
            for k in ("tp", "fp", "fn"):
                setattr(res, k, getattr(res, k) + getattr(s, k))

        dists = np.concatenate([s.dist for s in stats])

        tp_eps = res.tp + eps
        res.accuracy = tp_eps / (tp_eps + res.fp + res.fn) if tp_eps > 0 else 0
        res.precision = tp_eps / (tp_eps + res.fp) if tp_eps > 0 else 0
        res.recall = tp_eps / (tp_eps + res.fn) if tp_eps > 0 else 0
        res.f1 = (2 * tp_eps) / (2 * tp_eps + res.fp + res.fn) if tp_eps > 0 else 0

        pq_num = np.sum(cutoff_distance - dists) / cutoff_distance
        pq_den = tp_eps + res.fp / 2 + res.fn / 2

        res.panoptic_quality = float(pq_num / pq_den) if tp_eps > 0 else 0.0
        res.mean_dist = float(np.mean(dists)) if len(dists) > 0 else 0.0
        return res