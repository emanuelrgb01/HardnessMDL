import pandas as pd
from pyhard.measures import ClassificationMeasures
from pyhard.classification import ClassifiersPool
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import yaml


def load_dataset_config(path: str) -> dict:
    with open(path, "r") as f:
        dataset = yaml.safe_load(f)["datasets"]
    return dataset


def one_hot_encode(
    df: pd.DataFrame, categorical_attributes: list
) -> tuple[pd.DataFrame, list[str], OneHotEncoder]:

    encoder = OneHotEncoder(sparse_output=False)

    encoded_array = encoder.fit_transform(df[categorical_attributes])
    encoded_df = pd.DataFrame(
        encoded_array, columns=encoder.get_feature_names_out(categorical_attributes)
    )

    df_encoded = pd.concat(
        [
            df.drop(columns=categorical_attributes).reset_index(drop=True),
            encoded_df.reset_index(drop=True),
        ],
        axis=1,
    )

    new_columns = encoder.get_feature_names_out(categorical_attributes)

    return df_encoded, new_columns, encoder


def read_datasets_from_config(
    config_path_list: list, dry_run: bool = False
) -> pd.DataFrame:
    path_list = ["datasets/" + path for path in config_path_list]
    df_list = [pd.read_csv(path) for path in path_list]

    if dry_run:
        return pd.concat(df_list, ignore_index=True).sample(150, random_state=42)
    return pd.concat(df_list, ignore_index=True)


def compute_meta_features(df: pd.DataFrame, target_col: str = None):
    m = ClassificationMeasures(df, target_col)

    return m.calculate_all()


def compute_instance_hardness_by_models(
    df: pd.DataFrame, target_col: str, n_folds: int
):
    n_inst = len(df)
    learners = ClassifiersPool(df, target_col)
    instances_index = "instances"
    df_algo = learners.run_all(
        metric="logloss",
        n_folds=n_folds,
        n_iter=1,
        algo_list=[
            # "svc_linear",
            # "svc_rbf",
            "random_forest",
            "gradient_boosting",
            "bagging",
            "logistic_regression",
            "mlp",
        ],
        parameters={
            # "svc_linear": {"n_jobs": 1},
            # "svc_rbf": {"n_jobs": 1},
            "random_forest": {"n_jobs": 1},
            "bagging": {"n_jobs": 1},
        },
        hyper_param_optm=True,
        hpo_evals=3,
        hpo_timeout=90,
        verbose=False,
    )

    ih_values = learners.estimate_ih()

    df_ih = pd.DataFrame(
        {"instance_hardness": ih_values},
        index=pd.Index(range(1, n_inst + 1), name=instances_index),
    )

    comittee_classes, _ = learners.majority_prediction()
    df_algo["comittee_vote"] = comittee_classes
    df_ih["comittee_vote"] = comittee_classes
    return df_ih, df_algo


def calculate_pyhard_measures(df: pd.DataFrame, target_col: str, n_folds: int = 10):

    df_meta_feat = compute_meta_features(df, target_col)
    df_ih, df_algo = compute_instance_hardness_by_models(df, target_col, n_folds)

    df.reset_index(drop=True, inplace=True)
    df_meta_feat.reset_index(drop=True, inplace=True)
    df_ih.reset_index(drop=True, inplace=True)

    return pd.concat([df, df_meta_feat, df_ih], axis=1), df_algo


def compute_instance_hardness_with_pyhard(
    config_all: dict, dataset_name: str, dry_run: bool = False
):
    if dataset_name not in config_all:
        print(f"[ERROR] Dataset '{dataset_name}' not in config.")
        return None

    config = config_all[dataset_name]
    numerical_columns = config["numeric_attributes"]
    categorical_columns = config["categorical_attributes"]

    label_column = config["class_attribute"]

    df = read_datasets_from_config(config["file_name"], dry_run)

    categorical_transformed_columns = []
    if categorical_columns:
        df, categorical_transformed_columns, _ = one_hot_encode(df, categorical_columns)

    pyhard_measures, df_algo = calculate_pyhard_measures(
        df=df[
            numerical_columns + list(categorical_transformed_columns) + [label_column]
        ],
        # feature_cols=numerical_columns + list(categorical_transformed_columns),
        target_col=label_column,
        n_folds=5,
    )

    df_algo.to_csv(f"results/{dataset_name}_algo.csv", index=False)
    pyhard_measures.to_csv(f"results/{dataset_name}_pyhard.csv", index=False)
    pyhard_measures = pyhard_measures.reset_index(drop=True)

    pyhard_measures.to_csv(f"results/{dataset_name}_pyhard.csv", index=False)

    print(f"DONE: {dataset_name}")

    return pyhard_measures
