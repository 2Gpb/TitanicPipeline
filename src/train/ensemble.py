import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from data.dataset import load_data, prepare_data
from models.factory import create_model
from train.validation import run_cv


def make_member_config(config, name, settings):
    """Create member settings with shared data and validation splits."""
    member_config = OmegaConf.merge(config, settings)
    member_config.model.name = name
    member_config.validation = config.validation
    member_config.general.seed = config.general.seed
    member_config.data = config.data
    member_config.tuning.enabled = False
    return member_config


def run_ensemble(config, members):
    """Evaluate probability averaging and hard voting on shared CV folds."""
    if not members:
        raise ValueError("Provide at least one model")
    df_train, df_test = load_data(config)
    X, y, X_test, test_ids = prepare_data(df_train, df_test, config)
    oof_probabilities = pd.DataFrame(index=X.index)
    test_probabilities = pd.DataFrame(index=X_test.index)
    rows = []

    cv = StratifiedKFold(
        n_splits=config.validation.n_splits, shuffle=True,
        random_state=config.general.seed,
    )
    splits = list(cv.split(X, y))

    for name, settings in members.items():
        member_config = make_member_config(config, name, settings)
        print(f"Model: {name}")
        result = run_cv(create_model(member_config), X, y, X_test, member_config)
        oof_probabilities[name] = result["oof_probability"]
        test_probabilities[name] = result["test_probability"]
        rows.append({
            "model": name,
            "cv_mean": np.mean(result["val_scores"]),
            "cv_sd": np.std(result["val_scores"]),
        })

    oof_predictions = {
        "soft_voting": (oof_probabilities.mean(axis=1) >= 0.5).astype(int),
        "hard_voting": ((oof_probabilities >= 0.5).mean(axis=1) >= 0.5).astype(int),
    }
    test_predictions = {
        "soft_voting": (test_probabilities.mean(axis=1) >= 0.5).astype(int).to_numpy(),
        "hard_voting": ((test_probabilities >= 0.5).mean(axis=1) >= 0.5).astype(int).to_numpy(),
    }
    for name, prediction in oof_predictions.items():
        scores = [accuracy_score(y.iloc[val_idx], prediction.iloc[val_idx]) for _, val_idx in splits]
        rows.append({"model": name, "cv_mean": np.mean(scores), "cv_sd": np.std(scores)})

    scores = pd.DataFrame(rows)
    print(scores.to_string(index=False))
    return {
        "scores": scores,
        "oof_probabilities": oof_probabilities,
        "test_probabilities": test_probabilities,
        "test_predictions": test_predictions,
        "test_ids": test_ids,
    }
