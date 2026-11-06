from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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

def preprocess_data(X, y, X_test, config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = X.copy()
    X_test = X_test.copy()
    
    X["Sex"] = X["Sex"].map({"male": 1, "female": 0})
    X_test["Sex"] = X_test["Sex"].map({"male": 1, "female": 0})

    for col in config.preprocessing.drop_columns:
        X = X.drop(col, axis=1)
        X_test = X_test.drop(col, axis=1)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config.data.test_size, random_state=config.general.seed, stratify=y
    )

    embarked_mode = X_train["Embarked"].mode()[0]
    age_median = X_train["Age"].median()

    X_train.fillna({"Embarked": embarked_mode, "Age": age_median}, inplace=True)
    X_val.fillna({"Embarked": embarked_mode, "Age": age_median}, inplace=True)
    X_test.fillna({"Embarked": embarked_mode, "Age": age_median}, inplace=True)

    X_test.fillna({"Fare": X_train["Fare"].median()}, inplace=True)

    ohe = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    ).set_output(transform="pandas")

    encoded = ohe.fit_transform(X_train[["Embarked"]])
    encoded_val = ohe.transform(X_val[["Embarked"]])
    encoded_test = ohe.transform(X_test[["Embarked"]])

    X_train = pd.concat([X_train.drop(columns="Embarked"), encoded], axis=1)
    X_val = pd.concat([X_val.drop(columns="Embarked"), encoded_val], axis=1)
    X_test = pd.concat([X_test.drop(columns="Embarked"), encoded_test], axis=1)

    if config.preprocessing.scale:
        scaler = StandardScaler()

        scaler.fit(X_train)

        X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
        X_val = pd.DataFrame(scaler.transform(X_val), columns=X_train.columns, index=X_val.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns, index=X_test.index)

    return X_train, X_val, X_test, y_train, y_val
