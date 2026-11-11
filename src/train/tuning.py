import numpy as np
import optuna
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from data.dataset import preprocess_data
from models.factory import create_model, fit_model


def tune_model(X, y, X_test, config, n_trials=30):
    """Tune the selected model using the supplied search space."""
    if n_trials < 1:
        raise ValueError("n_trials must be positive")
    cv = StratifiedKFold(
        n_splits=config.validation.n_splits,
        shuffle=True,
        random_state=config.general.seed,
    )

    prepared_folds = []
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val, _ = preprocess_data(
            X.iloc[train_idx], X.iloc[val_idx], X_test, config,
        )
        prepared_folds.append((X_train, X_val, y.iloc[train_idx], y.iloc[val_idx]))

    def objective(trial):
        params = suggest_params(trial, config.model.name)
        fold_scores = []
        for X_train, X_val, y_train, y_val in prepared_folds:
            model = create_model(config, overrides=params)
            fit_model(model, X_train, y_train)
            fold_scores.append(float(accuracy_score(y_val, model.predict(X_val))))

        trial.set_user_attr("fold_scores", fold_scores)
        trial.set_user_attr("cv_sd", float(np.std(fold_scores)))
        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.general.seed),
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    print("Best parameters:", study.best_params)
    print(f"Best CV: {study.best_value:.5f} ± {study.best_trial.user_attrs['cv_sd']:.5f}")
    return study


def suggest_params(trial, model_name):
    """Suggest parameters from the search space of the selected model."""
    if model_name not in SEARCH_SPACES:
        raise ValueError(f"No search space for model: {model_name}")
    return SEARCH_SPACES[model_name](trial)


SEARCH_SPACES = {
    "logistic_regression": lambda trial: {
        "C": trial.suggest_float("C", 0.001, 10.0, log=True),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    },
    "decision_tree": lambda trial: {
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "max_features": trial.suggest_categorical("max_features", [None, "sqrt"]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "ccp_alpha": trial.suggest_float("ccp_alpha", 0.0, 0.02),
    },
    "random_forest": lambda trial: {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
        "max_depth": trial.suggest_categorical("max_depth", [None, 4, 6, 8, 10]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.5, None]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "ccp_alpha": trial.suggest_float("ccp_alpha", 0.0, 0.01),
        "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
    },
    "knn": lambda trial: {
        "n_neighbors": trial.suggest_int("n_neighbors", 3, 31, step=2),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "p": trial.suggest_int("p", 1, 2),
    },
    "cat_boost": lambda trial: {
        "iterations": trial.suggest_int("iterations", 200, 1000, step=100),
        "depth": trial.suggest_int("depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        "rsm": trial.suggest_float("rsm", 0.5, 1.0),
    },
    "lightgbm": lambda trial: {
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 4, 64),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10.0, log=True),
    },
    "xgboost": lambda trial: {
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 30.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    },
}
