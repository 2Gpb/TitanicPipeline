import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from data.dataset import load_data, prepare_data
from models.factory import create_model
from train.ensemble import make_member_config
from train.validation import run_cv


def create_meta_model(config):
    """Create the configured stacking model."""
    constructors = {
        "logistic_regression": LogisticRegression,
        "linear_regression": LinearRegression,
        "ridge": Ridge,
    }
    name = config.ensemble.stacking.meta_model
    if name not in constructors:
        raise ValueError(f"Unknown meta model: {name}")
    return constructors[name](**dict(config.ensemble.stacking.meta_params[name]))


def meta_scores(model, features):
    """Return positive-class probabilities or regression scores."""
    if hasattr(model, "predict_proba"):
        index = list(model.classes_).index(1)
        return model.predict_proba(features)[:, index]
    return model.predict(features)


def run_stacking(config, members):
    """Evaluate the entire stack with nested cross-validation."""
    if not members:
        raise ValueError("Provide at least one model")
    # Validate meta settings before training the base models.
    create_meta_model(config)
    df_train, df_test = load_data(config)
    X, y, X_test, test_ids = prepare_data(df_train, df_test, config)
    outer_cv = StratifiedKFold(
        n_splits=config.validation.n_splits,
        shuffle=True, random_state=config.general.seed,
    )
    inner_splits = config.ensemble.stacking.inner_splits
    if inner_splits < 2:
        raise ValueError("inner_splits must be at least 2")
    oof_scores = np.full(len(X), np.nan)
    fold_scores, test_scores = [], []

    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), start=1):
        outer_train, outer_y = X.iloc[train_idx], y.iloc[train_idx]
        if outer_y.value_counts().min() < inner_splits:
            raise ValueError("Not enough samples per class for inner CV")
        # Neither outer validation nor Kaggle test participates in fitting.
        prediction_X = pd.concat([X.iloc[val_idx], X_test], ignore_index=True)
        train_features, prediction_features = [], []

        for name, settings in members.items():
            member_config = make_member_config(config, name, settings)
            member_config.validation.n_splits = inner_splits
            print(f"Outer fold {fold}, base model: {name}")
            result = run_cv(
                create_model(member_config), outer_train, outer_y,
                prediction_X, member_config,
            )
            train_features.append(result["oof_probability"])
            prediction_features.append(result["test_probability"])

        train_features = np.column_stack(train_features)
        prediction_features = np.column_stack(prediction_features)
        meta_model = create_meta_model(config)
        meta_model.fit(train_features, outer_y.to_numpy())
        scores = meta_scores(meta_model, prediction_features)
        val_scores = scores[:len(val_idx)]
        oof_scores[val_idx] = val_scores
        test_scores.append(scores[len(val_idx):])
        accuracy = accuracy_score(y.iloc[val_idx], val_scores >= 0.5)
        fold_scores.append(float(accuracy))
        print(f"Outer fold {fold}: stacking accuracy={accuracy:.5f}")

    mean_test_score = np.mean(test_scores, axis=0)
    summary = pd.DataFrame([{
        "model": f"stacking_{config.ensemble.stacking.meta_model}",
        "cv_mean": np.mean(fold_scores),
        "cv_sd": np.std(fold_scores),
    }])
    print(summary.to_string(index=False))
    return {
        "scores": summary,
        "fold_scores": fold_scores,
        "oof_scores": oof_scores,
        "test_scores": mean_test_score,
        "test_predictions": {"stacking": (mean_test_score >= 0.5).astype(int)},
        "test_ids": test_ids,
    }
