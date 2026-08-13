"""SusScore helpers: off-task saliency scoring and blob-wise intervention."""

from __future__ import annotations

import cv2
import numpy as np

INTERVENTION_MIN_AREA = 80


def compute_sus_map(heatmap: np.ndarray, task_mask: np.ndarray) -> tuple[np.ndarray, float]:
    if task_mask.shape != heatmap.shape:
        task_mask = cv2.resize(
            task_mask.astype(np.float32),
            (heatmap.shape[1], heatmap.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    sus_map = (1.0 - task_mask) * heatmap
    return sus_map.astype(np.float32), float(sus_map.max())


def build_intervention_mask(
    sus_map: np.ndarray,
    tau: float,
    min_area: int = INTERVENTION_MIN_AREA,
) -> np.ndarray:
    """Mask high SusScore pixels inside anomalous connected components.

    1) Grow components on weakly positive off-task scores (S > 0).
    2) Keep components whose peak reaches tau.
    3) Within those components, intervene only where S >= tau
       (do not white-out the weak hinterland of the blob).
    """
    intervention = np.zeros(sus_map.shape, dtype=bool)
    peak_score = float(sus_map.max())
    if peak_score <= tau:
        return intervention

    binary = (sus_map > 0.0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    peak_y, peak_x = np.unravel_index(int(sus_map.argmax()), sus_map.shape)
    peak_label = int(labels[peak_y, peak_x])

    selected = np.zeros(sus_map.shape, dtype=bool)
    for label in range(1, num_labels):
        component = labels == label
        if float(sus_map[component].max()) < tau:
            continue
        if label == peak_label or stats[label, cv2.CC_STAT_AREA] >= min_area:
            selected[component] = True

    # Option B: only the high-score core, not the whole weak blob.
    intervention = selected & (sus_map >= tau)
    return intervention


def mask_rgb_by_intervention(rgb: np.ndarray, intervention_mask: np.ndarray) -> np.ndarray:
    masked = rgb.copy()
    masked[intervention_mask] = 255
    return masked
