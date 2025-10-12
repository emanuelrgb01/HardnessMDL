import yaml
import os
import sys
import pandas as pd
import numpy as np
from typing import Any, Dict, List
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from scipy.stats import entropy

# path_to_research = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))

# if path_to_research not in sys.path:
#    sys.path.insert(0, path_to_research)

# print(f"sys.path: {path_to_research}")

import hardnessmdl


def load_dataset_config(path: str) -> dict:
    with open(path, "r") as f:
        dataset = yaml.safe_load(f)["datasets"]
    return dataset


def _compute_description_lenght_fold(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    class_map: Dict[str, int],
    n_classes: int,
    n_dims: int,
    kwargs: Dict[str, Any],
):
    """K-Folf run"""
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    model = hardnessmdl.HardnessMDL(n_classes=n_classes, n_dims=n_dims)

    model.set_learning_rate(kwargs.get("learning_rate", 0.01))
    model.set_momentum(kwargs.get("momentum", 0.9))
    model.set_tau(kwargs.get("tau", 0))
    model.set_omega(kwargs.get("omega", 32.0))
    model.set_forgetting_factor(kwargs.get("forgetting_factor", 1.0))
    model.set_sigma(kwargs.get("sigma", 1.0))

    X_train = train_df[feature_cols].to_numpy()
    y_train_names = train_df[label_col].to_numpy()

    X_test = test_df[feature_cols].to_numpy()
    y_test_names = test_df[label_col].to_numpy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for j in range(len(X_train_scaled)):
        features = X_train_scaled[j]
        label = class_map[y_train_names[j]]
        model.train(features, label)

    results = []
    for i, (features, y_name, idx) in enumerate(
        zip(X_test_scaled, y_test_names, test_df.index)
    ):
        prediction_dict = model.predict(features)
        feature_dict = {col: val for col, val in zip(feature_cols, X_test[i])}
        results.append(
            {
                "index": idx,
                "true_label": class_map[y_name],
                **feature_dict,
                **prediction_dict,
            }
        )

    return results


def compute_kfold_description_lenght(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    k: int = 10,
    stratified: bool = True,
    n_jobs: int = os.cpu_count() - 1,
    **kwargs: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Parallel K-Fold Description Lenght computation.
    """

    class_names = sorted(df[label_col].unique().tolist())
    class_map = {name: i for i, name in enumerate(class_names)}
    n_classes = len(class_names)
    n_dims = len(feature_cols)

    if stratified:
        splitter = StratifiedKFold(
            n_splits=k, shuffle=True, random_state=kwargs.get("random_state", 42)
        )
        splits = splitter.split(df, df[label_col])
    else:
        splitter = KFold(
            n_splits=k, shuffle=True, random_state=kwargs.get("random_state", 42)
        )
        splits = splitter.split(df)

    results_nested = Parallel(n_jobs=n_jobs, batch_size="auto")(
        delayed(_compute_description_lenght_fold)(
            train_idx,
            test_idx,
            df,
            feature_cols,
            label_col,
            class_map,
            n_classes,
            n_dims,
            kwargs,
        )
        for train_idx, test_idx in splits
    )

    results = [item for sublist in results_nested for item in sublist]

    return results


def transform_dl_string_to_columns(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()  # without side-effects

    df_copy["is_error"] = (df_copy["true_label"] != df_copy["label"]).astype(int)

    df_copy["description_lengths_list"] = df_copy[
        "description_lengths_unnormalized"
    ].apply(lambda x: [float(i) for i in x.strip("[]").split(" ") if i])

    description_lengths_columns = pd.DataFrame(
        df_copy["description_lengths_list"].tolist(),
        columns=[
            f"L_{i}" for i in range(len(df_copy["description_lengths_list"].iloc[0]))
        ],
        index=df_copy.index,
    )

    return pd.concat(
        [
            df_copy.drop(columns=["description_lengths_list"]),
            description_lengths_columns,
        ],
        axis=1,
    )


def compute_hardness_measures(df):
    """
    Calculates a suite of instance hardness measures based on the model's
    description length (L_*) outputs.

    All measures are standardized to [0, 1], where 0.0 is easy and 1.0 is hard.
    """
    l_cols = [col for col in df.columns if col.startswith("L_")]
    n_classes = len(l_cols)

    def row_hardness(row):
        description_lengths = np.array([row[col] for col in l_cols])
        true_label_idx = int(row["true_label"])

        epsilon = 1e-9
        description_lengths[description_lengths <= 0] = epsilon

        dl_true = description_lengths[true_label_idx]
        other_dls = np.delete(description_lengths, true_label_idx)
        dl_range = np.max(description_lengths) - np.min(description_lengths)

        r_min = min(1.0, dl_true / (np.min(other_dls) + epsilon))
        r_med = min(1.0, dl_true / (np.mean(other_dls) + epsilon))
        rel_pos = (dl_true - np.min(description_lengths)) / (dl_range + epsilon)

        probs = np.exp(-(description_lengths - np.min(description_lengths)))
        probs /= np.sum(probs)

        pseudo_prob = 1.0 - probs[true_label_idx]

        max_entropy = np.log2(n_classes) if n_classes > 1 else 1.0
        norm_entropy = entropy(probs, base=2) / (max_entropy + epsilon)

        sorted_dls = np.sort(description_lengths)
        if len(sorted_dls) > 1:
            margin = sorted_dls[1] - sorted_dls[0]
            description_lenght_margin = 1.0 - (margin / (np.sum(sorted_dls) + epsilon))
        else:
            description_lenght_margin = 0.0

        description_lenght_true_cost = dl_true / (np.sum(description_lengths) + epsilon)

        ideal_dist = np.full(n_classes, epsilon)
        ideal_dist[true_label_idx] = 1.0

        ideal_dist /= np.sum(ideal_dist)

        kl_div = entropy(pk=ideal_dist, qk=probs, base=2)

        kullback_leibler_divergence = 1.0 / (1.0 + np.exp(-kl_div))

        return pd.Series(
            {
                "r_min": r_min,
                "r_med": r_med,
                "relative_position": rel_pos,
                "pseudo_probability": pseudo_prob,
                "normalized_entropy": norm_entropy,
                "description_lenght_margin": description_lenght_margin,
                "description_lenght_true_cost": description_lenght_true_cost,
                "kullback_leibler_divergence": kullback_leibler_divergence,
            }
        )

    return df.join(df.apply(row_hardness, axis=1))


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    return compute_hardness_measures(transform_dl_string_to_columns(df))
