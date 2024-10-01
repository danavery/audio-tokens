from typing import Dict, List

import numpy as np
from sklearn.metrics import average_precision_score


class MetricsCalculator:
    def compute_metrics(
        self, predictions: List[np.ndarray], labels: List[np.ndarray]
    ) -> Dict[str, float]:
        all_predictions = np.concatenate(predictions, axis=0)
        all_labels = np.concatenate(labels, axis=0)
        # For F1 and Hamming loss, we need binary predictions
        # binary_predictions = (predictions > self.config.prediction_threshold).astype(
        #     int
        # )
        return {
            # "f1_score_micro": f1_score(labels, binary_predictions, average="micro"),
            # "f1_score_macro": f1_score(labels, binary_predictions, average="macro"),
            # "hamming_loss": hamming_loss(labels, binary_predictions),
            # "mAP": average_precision_score(labels, predictions, average="macro"),
            "mAP": self.calculate_mAP(all_labels, all_predictions),
        }

    def calculate_mAP(self, labels, predictions):
        aps = []
        for i in range(labels.shape[1]):
            if (
                labels[:, i].sum() > 0
            ):  # Only calculate AP for classes with positive samples
                ap = average_precision_score(labels[:, i], predictions[:, i])
                aps.append(ap)
        return np.mean(aps) if aps else 0.0
