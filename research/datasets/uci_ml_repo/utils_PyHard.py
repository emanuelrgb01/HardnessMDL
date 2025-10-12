import pandas as pd
from pyhard.measures import ClassificationMeasures
from pyhard.classification import ClassifiersPool


def compute_meta_features(df: pd.DataFrame):
    m = ClassificationMeasures(df)

    return m.calculate_all()


def compute_instance_hardness_by_models(df: pd.DataFrame, n_folds: int):
    n_inst = len(df)
    learners = ClassifiersPool(df)
    instances_index = "instances"
    df_algo = learners.run_all(
        metric="logloss",
        n_folds=n_folds,
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


def calculate_pyhard_measures(df: pd.DataFrame, n_folds: int = 10):

    df_meta_feat = compute_meta_features(df)
    df_ih = compute_instance_hardness_by_models(df, n_folds)

    df.reset_index(drop=True, inplace=True)
    df_meta_feat.reset_index(drop=True, inplace=True)
    df_ih.reset_index(drop=True, inplace=True)

    return pd.concat([df, df_meta_feat, df_ih], axis=1)
