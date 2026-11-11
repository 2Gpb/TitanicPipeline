import pandas as pd

from data.dataset import PROJECT_ROOT


def save_test_predictions(test_pred, test_ids, config):
    """Save predictions as a competition submission."""

    submission = pd.DataFrame({
        config.data.id_column: test_ids.to_numpy(),
        config.data.target: test_pred.ravel().astype(int),
    })

    directory = PROJECT_ROOT / "submissions"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{config.general.experiment_name}.csv"
    submission.to_csv(path, index=False)
    return path
