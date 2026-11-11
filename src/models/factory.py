def create_model(config, overrides=None):
    """Create a classical model or a neural network."""
    if config.model.name == "mlp":
        from models.neural import MLPClassifier

        params = dict(config.model_params.mlp)
        params["use_embeddings"] = config.preprocessing.use_embeddings
        if overrides:
            params.update(overrides)
        params.setdefault("random_state", config.general.seed)
        return MLPClassifier(**params)

    from models.classic import create_model as create_classic_model

    return create_classic_model(config, overrides=overrides)


def fit_model(model, X_train, y_train):
    """Fit the selected model."""
    if type(model).__module__ == "models.neural":
        return model.fit(X_train, y_train)

    from models.classic import fit_model as fit_classic_model

    return fit_classic_model(model, X_train, y_train)
