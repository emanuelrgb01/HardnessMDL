import pandas as pd
import numpy as np
import utils_PyHard as utils_PyHard
from datetime import datetime


uci_ids = {
    # "fertility": 244,  # DONE
    # "iris": 53,  # DONE
    # "wine": 109,  # DONE
    # "libras": 181,  # DONE
    # "wdbc": 17,  # Wisconsin Diagnostic Breast Cancer # DONE
    # "contrac": 23,  # Contraceptive Method Choice # DONE
    # "wine-red": 186,  # DONE
    # "wine-white": 187, WIP
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
    print(f"START {dataset}", datetime.now())
    df = utils_PyHard.compute_instance_hardness_with_pyhard(
        dataset_config, dataset, False
    )
