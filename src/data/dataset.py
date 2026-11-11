from pathlib import Path
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMBEDDING_COLUMNS = ["Sex", "Pclass", "Embarked", "FamilyGroup"]


def load_data(config) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(PROJECT_ROOT / config.data.train_path)
    test = pd.read_csv(PROJECT_ROOT / config.data.test_path)
    return train, test


def prepare_data(train, test, config) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    columns = [
        col for col in train.columns
        if col not in (config.data.target, config.data.id_column)
    ]

    X = train[columns]
    y = train[config.data.target]
    X_test = test[columns]
    test_ids = test[config.data.id_column]

    return X, y, X_test, test_ids


def encode_categories(X_train, X_val, X_test, columns, start_index=0):
    """Encode categories using training vocabulary."""
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype="int64",
    )
    encoder.fit(X_train[columns].astype(str))
    for frame in (X_train, X_val, X_test):
        encoded = encoder.transform(frame[columns].astype(str)) + start_index
        for i, column in enumerate(columns):
            frame[column] = encoded[:, i]

    return X_train, X_val, X_test


def preprocess_data(X_train, X_val, X_test, config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit preprocessing on training data and transform all splits."""
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    categorical_columns = []
    if config.preprocessing.embarked_ohe:
        categorical_columns.append("Embarked")
    if config.preprocessing.pclass_ohe:
        categorical_columns.append("Pclass")

    for frame in (X_train, X_val, X_test):
        frame["Sex"] = frame["Sex"].map({"male": 1, "female": 0})
        if config.features.family_group:
            family_size = frame["SibSp"] + frame["Parch"] + 1
            frame["FamilyGroup"] = pd.cut(
                family_size,
                bins=[0, 1, 4, float("inf")],
                labels=["alone", "small", "large"],
            )

            frame.drop(columns=["SibSp", "Parch"], inplace=True)

        if config.features.cabin_known:
            frame["cabin_known"] = frame["Cabin"].notna().astype(int)

        if config.features.age_missing:
            frame["age_missing"] = frame["Age"].isna().astype(int)

        frame.drop(columns=list(config.preprocessing.drop_columns), inplace=True)

    if config.features.family_group:
        categorical_columns.append("FamilyGroup")

    if config.preprocessing.embarked_fill == "mode":
        embarked_fill = X_train["Embarked"].mode().iloc[0]
    elif config.preprocessing.embarked_fill == "unknown":
        embarked_fill = "Unknown"
    else:
        raise ValueError("embarked_fill must be 'mode' or 'unknown'")

    fill_values = {
        "Embarked": embarked_fill,
        "Age": X_train["Age"].median(),
        "Fare": X_train["Fare"].median(),
    }
    for frame in (X_train, X_val, X_test):
        frame.fillna(fill_values, inplace=True)

    if config.preprocessing.use_embeddings:
        if config.model.name != "mlp":
            raise ValueError("Embeddings are supported only for MLP")

        embedding_columns = [
            column for column in EMBEDDING_COLUMNS if column in X_train.columns
        ]

        numeric_columns = [
            column for column in X_train.columns
            if column not in embedding_columns
        ]

        X_train, X_val, X_test = encode_categories(
            X_train, X_val, X_test, embedding_columns, start_index=1,
        )

        if config.preprocessing.scale and numeric_columns:
            scaler = StandardScaler().set_output(transform="pandas")

            X_train[numeric_columns] = scaler.fit_transform(
                X_train[numeric_columns]
            )
            X_val[numeric_columns] = scaler.transform(
                X_val[numeric_columns]
            )
            X_test[numeric_columns] = scaler.transform(
                X_test[numeric_columns]
            )

        return X_train, X_val, X_test

    # Boosting models handle categories natively and need neither OHE nor scaling.
    if config.model.name in ("cat_boost", "lightgbm", "xgboost"):
        native_columns = ["Embarked", "Pclass"]
        if config.features.family_group:
            native_columns.append("FamilyGroup")

        for column in native_columns:
            if config.model.name == "cat_boost":
                for frame in (X_train, X_val, X_test):
                    frame[column] = frame[column].astype(str).astype(object)
            else:
                dtype = pd.CategoricalDtype(categories=X_train[column].dropna().unique())
                for frame in (X_train, X_val, X_test):
                    frame[column] = frame[column].astype(dtype)
        return X_train, X_val, X_test

    if not config.preprocessing.embarked_ohe:
        X_train, X_val, X_test = encode_categories(
            X_train, X_val, X_test, ["Embarked"],
        )

    if categorical_columns:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first").set_output(transform="pandas")
        encoded_train = ohe.fit_transform(X_train[categorical_columns])
        encoded_val = ohe.transform(X_val[categorical_columns])
        encoded_test = ohe.transform(X_test[categorical_columns])

        X_train = pd.concat([X_train.drop(columns=categorical_columns), encoded_train], axis=1)
        X_val = pd.concat([X_val.drop(columns=categorical_columns), encoded_val], axis=1)
        X_test = pd.concat([X_test.drop(columns=categorical_columns), encoded_test], axis=1)

    if config.preprocessing.scale:
        scaler = StandardScaler().set_output(transform="pandas")
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    return X_train, X_val, X_test
