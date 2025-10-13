import pandas as pd
import numpy as np
import utils_HardnessMDL as utils_MDL
from datetime import datetime


uci_ids = {
    # "fertility": 244, # DONE
    # "iris": 53, # DONE
    # "wine": 109, # DONE
    # "libras": 181, # DONE
    # "wdbc": 17,  # DONE # Wisconsin Diagnostic Breast Cancer
    # "contrac": 23,  # Contraceptive Method Choice
    # "wine-red": 186,
    # "wine-white": 187,
    "letter": 59,
    "adult": 2,
    "miniboone": 199,
    "skin": 229,
    "covertype": 31,
    # "hill-valey": 606,
    # "susy": 279,
    # "ht-sensor": 362,  # Gas sensors for home activity monitoring
}

datasets = list(uci_ids.keys())

dataset_config = utils_MDL.load_dataset_config("dataset.yaml")

for dataset in datasets:
    print(datetime.now())
    df = utils_MDL.compute_instance_hardness_with_mdl(dataset_config, dataset, False)
