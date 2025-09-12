from typing import Any, Dict, List
import hardnessmdl
import pandas as pd


def compute_loo_hardness(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    **kwargs: Dict[str, Any],
) -> List[Any]:
    """
    Computes hardness measures for each sample in a dataframe using
    Leave-One-Out cross-validation.

    Args:
        df: The full dataframe containing all samples.
        feature_cols: A list of column names to be used as features.
        label_col: The name of the column containing the class label.
        **kwargs: Hyperparameters for the GMDL model.

    Returns:
        A list of hardness measures, one for each sample in the original dataframe.
    """
    class_names = sorted(df[label_col].unique().tolist())
    class_map = {name: i for i, name in enumerate(class_names)}
    n_classes = len(class_names)
    n_dims = len(feature_cols)

    hardness_measures = []
    total_samples = len(df)

    for i, test_index in enumerate(df.index):
        print(f"Processing sample {i + 1}/{total_samples} (index: {test_index})...")

        test_df = df.loc[[test_index]]
        train_df = df.drop(index=test_index)

        model = hardnessmdl.HardnessMDL(n_classes=n_classes, n_dims=n_dims)

        model.set_learning_rate(kwargs.get("learning_rate", 0.01))
        model.set_momentum(kwargs.get("momentum", 0.9))
        model.set_tau(kwargs.get("tau", 1.0))
        model.set_omega(kwargs.get("omega", 32.0))
        model.set_forgetting_factor(kwargs.get("forgetting_factor", 1.0))
        model.set_sigma(kwargs.get("sigma", 1.0))

        X_train = train_df[feature_cols].to_numpy()
        y_train_names = train_df[label_col].to_numpy()

        for j in range(len(X_train)):
            features = X_train[j]
            label = class_map[y_train_names[j]]
            model.train(features, label)

        X_test = test_df[feature_cols].to_numpy()[0]
        y_test_name = test_df[label_col].to_numpy()[0]
        true_label = class_map[y_test_name]

        prediction = model.hardness(X_test, true_label)

        hardness_measures.append(prediction)

    print("\nLeave-One-Out processing complete.")
    return hardness_measures


if __name__ == "__main__":
    pass
