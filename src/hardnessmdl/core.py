from .pygmdl import GMDL

import numpy as np
from typing import List, Dict, Any, Optional


Sample = Dict[str, Any]
Prediction = Dict[str, Any]


class HardnessMDL(GMDL):
    """
    Instance hardness measures based on the Minimum Description Length principle.

    This class extends the GMDL learning algorithm by providing methods to
    compute instance hardness measures derived from description lengths.
    These measures quantify the difficulty of classifying individual
    samples according to the MDL principle.
    """

    def __init__(self, n_classes: int, n_dims: int, seed: int = 42):
        super().__init__(n_classes, n_dims, seed)

    def predict(self, features: np.ndarray) -> Prediction:
        """
        Predicts the class label for a single sample.

        Args:
            features: A 1D NumPy array of feature values.

        Returns:
            A Prediction dictionary containing the predicted 'label' and the
            'description_lengths' for each class.
        """
        distances = self._get_distances(features)

        description_lengths_unnormalized = np.array(
            [self._L_hat(features, c, distances) for c in range(self.n_classes)]
        )

        l_total = np.sum(description_lengths_unnormalized)
        if np.isclose(l_total, 0):
            description_lengths = np.full(self.n_classes, 1.0 / self.n_classes)
        else:
            description_lengths = description_lengths_unnormalized / l_total

        predicted_label = np.argmin(description_lengths)

        return {
            "label": predicted_label,
            "description_lengths": description_lengths,
            "description_lengths_unnormalized": description_lengths_unnormalized,
        }

    def hardness(self, features: np.ndarray, label: int) -> dict[str, float]:
        """Compute multiple hardness measures for a given instance.

        Aggregates different hardness metrics based on description
        lengths, including R_min, R_med, relative position,
        pseudo-probability, and normalized entropy.

        Args:
            features: A 1D NumPy array of feature values.
            label: The integer class label for the sample.

        Returns:
            dict[str, float]: Dictionary with all hardness measures.
        """
        if features.shape[0] != self.n_dims:
            raise ValueError(
                f"Feature vector has wrong dimension {features.shape[0]}, expected {self.n_dims}"
            )

        predictions = self.predict(features)
        description_lengths = predictions["description_lengths"]
        predicted_label = predictions["label"]

        return {
            "label": predicted_label,
            "true_label": label,
            "r_min": self._r_min(description_lengths, label),
            "r_med": self._r_med(description_lengths, label),
            "relative_position": self._relative_position(description_lengths, label),
            "pseudo_probability": self._pseudo_probability(description_lengths, label),
            "normalized_entropy": self._normalized_entropy(description_lengths),
        }

    def _r_min(self, description_lenghts: np.ndarray, label: int) -> float:
        """Compute the normalized minimum ratio (R_min).

        Measures how close the description length of the true class is
        to the minimum description length among the other classes.

        Args:
            description_lengths: A 1D NumPy array of description lengths.
            label: The integer class label for the sample.

        Returns:
            float: Normalized minimum ratio (R_min).
        """
        return min(
            1,
            description_lenghts[label] / np.min(np.delete(description_lenghts, label)),
        )

    def _r_med(self, description_lenghts: np.ndarray, label: int) -> float:
        """Compute the normalized mean ratio (R_med).

        Measures how close the description length of the true class is
        to the mean description length among the other classes.

        Args:
            description_lengths: A 1D NumPy array of description lengths.
            label: The integer class label for the sample.

        Returns:
            float: Normalized mean ratio (R_med).
        """
        return min(
            1,
            description_lenghts[label] / np.mean(np.delete(description_lenghts, label)),
        )

    def _relative_position(self, description_lenghts: np.ndarray, label: int) -> float:
        """Compute the relative position (PR).

        Places the description length of the true class on a scale
        from 0 to 1, where 0 means it is the smallest and
        1 means it is the largest.

        Args:
            description_lengths: A 1D NumPy array of description lengths.
            label: The integer class label for the sample.

        Returns:
            float: Relative position (PR).
        """
        return (description_lenghts[label] - np.min(description_lenghts)) / (
            np.max(description_lenghts) - np.min(description_lenghts)
        )

    def _pseudo_probability(self, description_lenghts: np.ndarray, label: int) -> float:
        """Compute the pseudo-probability measure (PP).

        Based on Shannon-Fano coding, it derives a pseudo-probability
        for the true class and defines difficulty as
        one minus this probability.

        Args:
            description_lengths: A 1D NumPy array of description lengths.
            label: The integer class label for the sample.

        Returns:
            float: Pseudo-probability difficulty (PP).
        """
        pseudo_probabilities = np.exp2(-description_lenghts)
        return 1 - pseudo_probabilities[label] / np.sum(pseudo_probabilities)

    def _normalized_entropy(self, description_lenghts: np.ndarray) -> float:
        """Compute the normalized entropy (EN).

        Uses pseudo-probabilities to measure the uncertainty
        in classification, normalized by the maximum entropy.

        Args:
            description_lengths: A 1D NumPy array of description lengths.

        Returns:
            float: Normalized entropy (EN).
        """
        pseudo_probabilities = np.exp2(-description_lenghts)
        H = np.dot(pseudo_probabilities, description_lenghts)

        return H / self.n_dims
