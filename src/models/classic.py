from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


def create_model(config, overrides=None):
    """Create the configured classifier."""
    constructors = {
        "logistic_regression": LogisticRegression,
        "decision_tree": DecisionTreeClassifier,
        "random_forest": RandomForestClassifier,
        "knn": KNeighborsClassifier,
        "cat_boost": CatBoostClassifier,
        "lightgbm": LGBMClassifier,
        "xgboost": XGBClassifier,
    }

    name = config.model.name
    if name not in constructors:
        raise ValueError(f"Unknown model: {name}. Choose from {list(constructors)}")
    params = dict(config.model_params[name])
    if overrides:
        params.update(overrides)
    if name != "knn":
        params.setdefault("random_state", config.general.seed)
    if name == "cat_boost":
        params.setdefault("verbose", False)
    if name == "xgboost":
        params.setdefault("enable_categorical", True)
        params.setdefault("tree_method", "hist")
    return constructors[name](**params)


def fit_model(model, X_train, y_train):
    """Fit a classifier with its categorical feature settings."""
    if isinstance(model, CatBoostClassifier):
        categorical_columns = X_train.select_dtypes(
            include=["object", "string", "category"],
        ).columns.tolist()
        model.fit(X_train, y_train, cat_features=categorical_columns)
    else:
        model.fit(X_train, y_train)
    return model
