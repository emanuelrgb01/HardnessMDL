import pandas as pd
from pyhard.measures import ClassificationMeasures
from pyhard.classification import ClassifiersPool

files_path = [
    "two_classes/test1.csv",
    "two_classes/test2.csv",
    "two_classes/test3.csv",
    "two_classes/test4.csv",
    "three_classes/test1.csv",
    "three_classes/test2.csv",
    "three_classes/test3.csv",
    "three_classes/test4.csv",
    "five_classes/test1.csv",
    "five_classes/test2.csv",
    "five_classes/test3.csv",
    "five_classes/test4.csv",
]

df_meta_feats_dict = {}

for path in files_path:
    df = pd.read_csv(path)
    n_inst = len(df)

    m = ClassificationMeasures(df)
    learners = ClassifiersPool(df)

    df_meta_feat = m.calculate_all()

    instances_index = "instances"
    df_algo = learners.run_all(
        metric="logloss",
        n_folds=5,
        n_iter=1,
        algo_list=[
            "svc_linear",
            "svc_rbf",
            "random_forest",
            "gradient_boosting",
            "bagging",
            "logistic_regression",
            "mlp",
        ],
        parameters={"random_forest": {"n_jobs": -1}, "bagging": {"n_jobs": -1}},
        hyper_param_optm=True,
        hpo_evals=10,
        hpo_timeout=90,
        verbose=False,
    )

    ih_values = learners.estimate_ih()

    df_ih = pd.DataFrame(
        {"instance_hardness": ih_values},
        index=pd.Index(range(1, n_inst + 1), name=instances_index),
    )

    df.reset_index(drop=True, inplace=True)
    df_meta_feat.reset_index(drop=True, inplace=True)
    df_ih.reset_index(drop=True, inplace=True)

    df_final = pd.concat([df, df_meta_feat, df_ih], axis=1)
    df_final.to_csv(path.split(".")[0] + "_PyHard.csv", index=False)

    df_meta_feats_dict[path] = df_final

    print("Done " + path)
