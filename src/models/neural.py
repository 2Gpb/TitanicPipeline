import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from torch import nn

from data.dataset import EMBEDDING_COLUMNS
from data.neural import make_embedding_loader, make_loader, to_tensor


ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU}


class MLP(nn.Module):
    """A binary classifier with optional categorical embeddings."""

    def __init__(
        self,
        input_size,
        hidden_size,
        n_layers=1,
        activation="gelu",
        batch_norm=True,
        dropout=0.3,
        category_sizes=None,
        embedding_dim=4,
    ):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation: {activation}")

        category_sizes = [] if category_sizes is None else category_sizes

        self.embeddings = nn.ModuleList([
            nn.Embedding(
                num_embeddings=size,
                embedding_dim=embedding_dim,
                padding_idx=0,
            )
            for size in category_sizes
        ])

        layers = []
        in_size = input_size + len(category_sizes) * embedding_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_size, hidden_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(ACTIVATIONS[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            in_size = hidden_size
        layers.append(nn.Linear(in_size, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, numeric, categorical=None):
        """Return one logit per row."""
        if len(self.embeddings) > 0:
            vectors = [
                embedding(categorical[:, i])
                for i, embedding in enumerate(self.embeddings)
            ]

            numeric = torch.cat([numeric, *vectors], dim=1)

        return self.layers(numeric).squeeze(-1)


class MLPClassifier(ClassifierMixin, BaseEstimator):
    """Expose a PyTorch MLP through the classifier interface."""

    def __init__(self, hidden_size=32, n_layers=1, activation="gelu", batch_norm=True,
                 epochs=30, batch_size=64, learning_rate=0.001,
                 random_state=42, use_scheduler=False, min_lr=0.00001,
                 optimizer="adam", weight_decay=0.0, momentum=0.9, use_embeddings=False,
                 embedding_dim=4, dropout=0.3):
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.activation = activation
        self.batch_norm = batch_norm
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.use_scheduler = use_scheduler
        self.min_lr = min_lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.use_embeddings = use_embeddings
        self.embedding_dim = embedding_dim
        self.dropout = dropout

    def fit(self, X, y):
        """Train the network on numeric features."""
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.use_scheduler and not 0 <= self.min_lr <= self.learning_rate:
            raise ValueError("min_lr must be between zero and learning_rate")
        self.classes_ = np.unique(y)
        if not np.array_equal(self.classes_, [0, 1]):
            raise ValueError("MLPClassifier requires target labels 0 and 1")
        torch.manual_seed(self.random_state)
        self.n_features_in_ = X.shape[1]

        if self.use_embeddings:
            self.categorical_columns_ = [
                column for column in EMBEDDING_COLUMNS if column in X.columns
            ]

            loader, self.numeric_columns_, category_sizes = make_embedding_loader(
                X,
                y,
                self.categorical_columns_,
                self.batch_size,
                self.random_state,
            )

            input_size = len(self.numeric_columns_)
        else:
            loader = make_loader(
                X, y, self.batch_size, self.random_state
            )
            category_sizes = None
            input_size = self.n_features_in_

        self.network_ = MLP(
            input_size=input_size,
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
            activation=self.activation,
            batch_norm=self.batch_norm,
            dropout=self.dropout,
            category_sizes=category_sizes,
            embedding_dim=self.embedding_dim,
        )

        loss_fn = nn.BCEWithLogitsLoss()
        optimizers = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
        }
        if self.optimizer not in optimizers:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
        optimizer_params = {"lr": self.learning_rate, "weight_decay": self.weight_decay}
        if self.optimizer == "sgd":
            optimizer_params["momentum"] = self.momentum
        optimizer = optimizers[self.optimizer](self.network_.parameters(), **optimizer_params)
        scheduler = None
        if self.use_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs, eta_min=self.min_lr,
            )
        self.loss_history_ = []
        self.lr_history_ = []

        for _ in range(self.epochs):
            self.network_.train()
            self.lr_history_.append(optimizer.param_groups[0]["lr"])
            total_loss = 0.0
            for batch in loader:
                optimizer.zero_grad()

                if self.use_embeddings:
                    numeric, categorical, targets = batch
                    logits = self.network_(numeric, categorical)
                else:
                    features, targets = batch
                    logits = self.network_(features)

                loss = loss_fn(logits, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(targets)
            self.loss_history_.append(total_loss / len(loader.dataset))
            if scheduler is not None:
                scheduler.step()
        return self

    def predict_proba(self, X):
        """Return probabilities for classes zero and one."""
        check_is_fitted(self, "network_")
        self.network_.eval()

        with torch.no_grad():
            if self.use_embeddings:
                numeric = to_tensor(X[self.numeric_columns_])
                categorical = torch.as_tensor(
                    X[self.categorical_columns_].to_numpy(dtype=np.int64),
                    dtype=torch.long,
                )
                logits = self.network_(numeric, categorical)
            else:
                logits = self.network_(to_tensor(X))

            probability = torch.sigmoid(logits).numpy()

        return np.column_stack([1 - probability, probability])

    def predict(self, X):
        """Return binary class predictions."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
