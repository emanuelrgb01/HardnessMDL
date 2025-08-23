from .pygmdl import GMDL

import numpy as np
from typing import List, Dict, Any, Optional


Sample = Dict[str, Any]
Prediction = Dict[str, Any]


class HardnessMDL(GMDL):
    """
    Instance hardness measures based on the Minimum Description Length principle.

    This class implements an online learning algorithm that uses Online Kernel
    Density Estimators (oKDE) to model the probability distribution of each
    class. It classifies new samples based on the Minimum Description Length
    (MDL) principle.
    """

    def __init__(self, n_classes: int, n_dims: int, seed: int = 42):
        super.__init__(n_classes, n_dims, seed)

    def hardness(self, features: np.ndarray, label: int) -> dict:
        """"""
        # TODO: implement
        return {}

    def _r_min(description_lenght: np.ndarray, label: int) -> float:
        """"""
        # TODO: implement
        return 0

    def _r_med(description_lenght: np.ndarray, label: int) -> float:
        """"""
        # TODO: implement
        return 0

    def _pr(description_lenght: np.ndarray, label: int) -> float:
        """"""
        # TODO: implement
        return 0

    def _pp(description_lenght: np.ndarray, label: int) -> float:
        """"""
        # TODO: implement
        return 0

    def _en(description_lenght: np.ndarray) -> float:
        """"""
        # TODO: implement
        return 0
