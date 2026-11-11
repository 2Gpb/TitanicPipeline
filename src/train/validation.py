import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from data.dataset import preprocess_data
from models.factory import fit_model


def _evaluate_split(model, X_train, X_val, y_train, y_val, X_test, config):
    """Train and evaluate a model on one split."""
    X_train, X_val, fold_test = preprocess_data(
        X_train, X_val, X_test, config,
    )
    fold_model = clone(model)
    fit_model(fold_model, X_train, y_train)
    positive_class = list(fold_model.classes_).index(1)
    val_pred = fold_model.predict(X_val)
    return {
        "train_score": float(accuracy_score(y_train, fold_model.predict(X_train))),
        "val_score": float(accuracy_score(y_val, val_pred)),
        "val_probability": fold_model.predict_proba(X_val)[:, positive_class],
        "test_probability": fold_model.predict_proba(fold_test)[:, positive_class],
    }


def run_holdout(model, X, y, X_test, config):
    """Evaluate a model using a holdout split."""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config.validation.test_size,
        random_state=config.general.seed, stratify=y,
    )
    result = _evaluate_split(model, X_train, X_val, y_train, y_val, X_test, config)
    return {
        "train_scores": [result["train_score"]],
        "val_scores": [result["val_score"]],
        "test_probability": result["test_probability"],
    }


def run_cv(model, X, y, X_test, config):
    """Evaluate a model using cross-validation."""
    cv = StratifiedKFold(
        n_splits=config.validation.n_splits, shuffle=True,
        random_state=config.general.seed,
    )
    train_scores, val_scores, test_probabilities = [], [], []
    oof_probability = np.full(len(X), np.nan)
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        result = _evaluate_split(model, X_train, X_val, y_train, y_val, X_test, config)
        train_scores.append(result["train_score"])
        val_scores.append(result["val_score"])
        test_probabilities.append(result["test_probability"])
        oof_probability[val_idx] = result["val_probability"]
        print(f"Fold {fold}: train={result['train_score']:.5f}, val={result['val_score']:.5f}")
    return {
        "train_scores": train_scores,
        "val_scores": val_scores,
        "test_probability": np.mean(test_probabilities, axis=0),
        "oof_probability": oof_probability,
    }
