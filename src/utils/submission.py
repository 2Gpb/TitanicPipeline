import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def save_test_predictions(model, X_test, test_ids, config):
    test_pred = model.predict(X_test)

    submission = pd.DataFrame({
        config.data.id_column: test_ids.to_numpy(),
        config.data.target: test_pred.ravel().astype(int),
    })

    submission.to_csv(PROJECT_ROOT / f"{config.general.experiment_name}.csv", index=False)