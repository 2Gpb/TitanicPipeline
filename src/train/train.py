from data.dataset import load_data, prepare_data, preprocess_data
from utils.submission import save_test_predictions

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train(model, X_train, y_train):
    model.fit(X_train, y_train)
    pred = model.predict(X_train)
    return accuracy_score(y_train, pred)

def validation(model, X_val, y_val):
    pred = model.predict(X_val)
    return accuracy_score(y_val, pred)

def run(config):
    df_train, df_test = load_data(config)
    X, y, X_test, test_ids = prepare_data(df_train, df_test, config)
    X_train, X_val, X_test, y_train, y_val = preprocess_data(X, y, X_test, config)

    model = LogisticRegression()
    train_score = train(model, X_train, y_train)
    val_score = validation(model, X_val, y_val)

    print(f"Current train score: {train_score}")
    print(f"Current val score: {val_score}")

    save_test_predictions(model, X_test, test_ids, config)

