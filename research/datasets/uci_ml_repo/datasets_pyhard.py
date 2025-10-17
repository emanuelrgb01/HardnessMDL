import pandas as pd
import numpy as np
import utils_PyHard as utils_PyHard
from datetime import datetime


uci_ids = {
    "fertility": 244,
    "iris": 53,
    "wine": 109,
    "libras": 181,
    "wdbc": 17,  # Wisconsin Diagnostic Breast Cancer
    "contrac": 23,  # Contraceptive Method Choice
    "wine-red": 186,
    "wine-white": 187,
    "letter": 59,
    "adult": 2,
    # "miniboone": 199,
    # "skin": 229,
    # "covertype": 31,
    # "hill-valey": 606,
    # "susy": 279,
    # "ht-sensor": 362,  # Gas sensors for home activity monitoring
}

datasets = list(uci_ids.keys())

dataset_config = utils_PyHard.load_dataset_config("dataset.yaml")

for dataset in datasets:
    print(datetime.now())
    df = utils_PyHard.compute_instance_hardness_with_pyhard(
        dataset_config, dataset, False
    )
