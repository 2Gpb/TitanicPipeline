import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def to_tensor(values):
    """Convert tabular values to a CPU float tensor."""
    return torch.as_tensor(np.asarray(values, dtype=np.float32))


def make_loader(X, y, batch_size, seed):
    """Create shuffled training batches."""
    dataset = TensorDataset(to_tensor(X), to_tensor(y).reshape(-1))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

def make_embedding_loader(X, y, categorical_columns, batch_size, seed):
    """Create training batches with separate categorical inputs."""
    numeric_columns = [
        column for column in X.columns
        if column not in categorical_columns
    ]

    numeric = to_tensor(X[numeric_columns])

    categorical = torch.as_tensor(
        X[categorical_columns].to_numpy(dtype=np.int64),
        dtype=torch.long,
    )

    targets = to_tensor(y)

    dataset = TensorDataset(numeric, categorical, targets)
    generator = torch.Generator().manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    category_sizes = [
        int(X[column].max()) + 1
        for column in categorical_columns
    ]

    return loader, numeric_columns, category_sizes