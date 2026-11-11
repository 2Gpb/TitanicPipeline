import numpy as np

from data.dataset import load_data, prepare_data
from models.factory import create_model
from train.validation import run_cv, run_holdout
from train.tuning import tune_model
from train.ensemble import run_ensemble
from train.stacking import run_stacking
from utils.submission import save_test_predictions


def run(config):
    """Run model evaluation and save the submission."""
    if config.ensemble.enabled:
        if not config.validation.use_cv:
            raise ValueError("Ensemble evaluation requires use_cv=true")
        if config.tuning.enabled:
            raise ValueError("Disable tuning before running the ensemble")
        method = config.ensemble.method
        if method not in ("soft_voting", "hard_voting", "stacking"):
            raise ValueError("Ensemble method must be soft_voting, hard_voting or stacking")
        if method == "stacking":
            result = run_stacking(config, config.ensemble.members)
        else:
            result = run_ensemble(config, config.ensemble.members)
        result["submission_path"] = save_test_predictions(
            result["test_predictions"][method], result["test_ids"], config,
        )
        return result

    if config.tuning.enabled:
        if not config.validation.use_cv:
            raise ValueError("Tuning uses all training data through CV; use_cv must be true.")

    df_train, df_test = load_data(config)
    X, y, X_test, test_ids = prepare_data(df_train, df_test, config)

    if config.tuning.enabled:
        study = tune_model(
            X, y, X_test, config,
            n_trials=config.tuning.n_trials,
        )
        model = create_model(config, overrides=study.best_params)
    else:
        model = create_model(config)

    evaluate = run_cv if config.validation.use_cv else run_holdout
    result = evaluate(model, X, y, X_test, config)
    metrics = {
        "model": config.model.name,
        "model_params": model.get_params(),
        "validation": "cv" if config.validation.use_cv else "holdout",
        "train_scores": result["train_scores"],
        "val_scores": result["val_scores"],
        "train_mean": float(np.mean(result["train_scores"])),
        "val_mean": float(np.mean(result["val_scores"])),
        "val_sd": float(np.std(result["val_scores"])),
        "tuned_on_same_cv": bool(config.tuning.enabled),
    }
    print(f"Mean train accuracy: {metrics['train_mean']:.5f}")
    if config.validation.use_cv:
        print(f"CV accuracy: {metrics['val_mean']:.5f} ± {metrics['val_sd']:.5f}")
    else:
        print(f"Holdout accuracy: {metrics['val_mean']:.5f}")

    if config.tuning.enabled:
        print("Best parameters:", study.best_params)

    test_pred = (result["test_probability"] >= 0.5).astype(int)
    submission_path = save_test_predictions(test_pred, test_ids, config)
    metrics["submission_path"] = str(submission_path)
    return {"metrics": metrics, "test_probability": result["test_probability"]}
