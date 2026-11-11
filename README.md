# MLPipeline

A project for the Kaggle competition [Titanic — Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic).

Place `train.csv` and `test.csv` in the `raw_data/` directory at the project root.
To run the pipeline from the project root, execute `uv sync --locked`, then `uv run --locked python src/main.py`.
EDA and the first baseline are in `notebooks/draft.ipynb`.

## How the pipeline works

Everything starts from `src/main.py`. It reads `src/configs/config.yml` and calls `run()` from `src/train/train.py`. Depending on the config, it trains one model (with Optuna if tuning is on) or an ensemble.

- `data/dataset.py` loads the data and does all preprocessing: new features, missing values, category encoding for each model type, scaling. Everything is fitted only on the train fold.
- `data/neural.py` makes tensors and data loaders for the MLP.
- `models/classic.py` has the sklearn models and boostings, `models/neural.py` has the MLP with a sklearn-style wrapper, and `models/factory.py` picks one of them by name.
- `train/validation.py` runs holdout or stratified 5-fold CV. Test predictions are averaged over the fold models.
- `train/tuning.py` runs Optuna, with a separate search space for each model.
- `train/ensemble.py` does soft and hard voting on the same folds, `train/stacking.py` does stacking with nested CV.
- `utils/submission.py` saves the submission to `submissions/<experiment_name>.csv`.

To run another experiment I only change the config: model and its params, preprocessing flags, features, ensemble on/off. Right now the config is set to `mlp_v6`, which has the best public score.

## Weak spots

I know about these and will fix them in the next project:

- CV after Optuna is too optimistic, because params were tuned and scored on the same folds. Ensemble members were also picked by this CV.
- CV is 0.82–0.85, public is 0.76–0.79. Fold SD is about 0.015–0.02, and the test has only 418 rows (one passenger is about 0.24% accuracy), so most of the difference between models is noise. The best public model is mostly luck.
- Stacking uses Ridge as the meta model, which is just linear regression on a 0/1 target with a 0.5 threshold. Logistic regression fits classification better. It's supported in the code, but I didn't run it.
- `preprocess_data` branches by model name, so the data part knows about models. This is the messiest part of the code.
- Titanic columns are hardcoded (Sex mapping, FamilyGroup, Embarked/Pclass). For another dataset they should be moved to the config.
- Some early results can't be reproduced exactly. Preprocessing changed a bit during the experiments, and I rewrote the Optuna search space for each model. Next time I'll save the code state together with each result.
- Metrics are only printed, so I filled in the table by hand.

## Summary

This project isn't about getting the best possible score in this competition. It's a first version of a pipeline for quickly trying out ideas in competitions. The idea is that the core can be reused, and for a new competition you mostly change the data part and the config. The pipeline will definitely get better over time, the main gaps are listed in Weak spots above.

Below are the results after simple preprocessing and trying different models with different hyperparameters. To improve the score, you need to dig into the data a bit deeper. As I understand it, the main thing is to pull something out of the text columns I dropped in my simple version (Name, Ticket, Cabin), or really anything else you can experiment with :)

| Experiment                 | Model and setup                                                                                                                                                                                  | Local CV accuracy ± SD | Public accuracy |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------: | --------------: |
| logreg_baseline_fe_v1      | LogReg, default parameters; median/mode imputation, OHE, scaling; excluding Name/Ticket/Cabin; seed=42                                                                                           | 0.79687 ± 0.01457      | 0.76555         |
| logreg_baseline_fe_v2      | LogReg + feature engineering: Pclass OHE, family groups replacing SibSp/Parch, cabin_known, age_missing; Unknown Embarked; scaling; seed=42                                                      | 0.81595 ± 0.02219      | 0.75837         |
| logreg_baseline_v2_optuna  | LogReg + feature engineering + Optuna; C=0.022941879158967466, class_weight=None                                                                                                                 | 0.82157 ± 0.01804      | 0.76555         |
| decision_tree_fe_v1        | DecisionTree + feature engineering; max_depth=3, min_samples_leaf=5; no scaling; no tuning; seed=42                                                                                              | 0.80920 ± 0.02711      | 0.78708         |
| decision_tree_fe_v1_optuna | DecisionTree + feature engineering + Optuna; max_depth=4, min_samples_leaf=5, min_samples_split=15, criterion=entropy, max_features=None, class_weight=None, ccp_alpha=0.01; no scaling; seed=42 | 0.82043 ± 0.01414      | 0.78229         |
| decision_tree_v2_optuna    | DecisionTree, Optuna-selected parameters; family_group/cabin_known/age_missing disabled; Pclass OHE; no scaling; seed=42                                                                         | 0.82153 ± 0.01942      | 0.78229         |
| random_forest_v1           | RandomForest; n_estimators=300, max_depth=5, min_samples_leaf=3; Pclass without OHE; no extra features or scaling; seed=42                                                                       | 0.83499 ± 0.01381      | 0.78468         |
| random_forest_v1_optuna    | RandomForest + Optuna; Pclass without OHE, Embarked OHE; no extra features or scaling; seed=42                                                                                                   | 0.84285 ± 0.01355      | 0.77751         |
| knn_v1                     | KNN; n_neighbors=5, weights=uniform; Pclass/Embarked OHE, scaling; no extra features; seed=42                                                                                                    | 0.81034 ± 0.01740      | 0.75598         |
| knn_v1_optuna              | KNN + Optuna; n_neighbors=7, weights=uniform, p=1; Pclass/Embarked OHE, scaling; no extra features; seed=42                                                                                      | 0.81371 ± 0.01771      | 0.76315         |
| cat_boost_v1 | CatBoost; iterations=1000, depth=6; native Embarked/Pclass categories; no extra features, no scaling; seed=42 | 0.83724 ± 0.01309 | 0.78229 |
| cat_boost_v1_optuna | CatBoost + Optuna; iterations=400, depth=7, learning_rate=0.04179830537843056, l2_leaf_reg=19.38748179654593, random_strength=0.9256927019761361, rsm=0.8795642931354145; native Embarked/Pclass categories; no extra features, no scaling; seed=42 | 0.83388 ± 0.01623 | 0.77751 |
| lightgbm_v1 | LightGBM; n_estimators=100, learning_rate=0.1; native Embarked/Pclass categories; no extra features or scaling; seed=42 | 0.83162 ± 0.01997 | 0.77033 |
| lightgbm_v1_optuna | LightGBM + Optuna; max_depth=5, n_estimators=500, learning_rate=0.02142290514299836, num_leaves=23, min_child_samples=27, colsample_bytree=0.6897243752201568, reg_alpha=0.08819067774470501, reg_lambda=0.013943332610388649; native Embarked/Pclass categories; no extra features or scaling; seed=42 | 0.84510 ± 0.01703 | 0.75598 |
| xgboost_v1 | XGBoost; max_depth=3, n_estimators=100, min_child_weight=7; native Embarked/Pclass categories; no extra features or scaling; seed=42 | 0.84285 ± 0.02115 | 0.76076 |
| xgboost_v1_optuna | XGBoost + Optuna; max_depth=5, min_child_weight=4.105961221973598, n_estimators=800, learning_rate=0.07618089083517389, subsample=0.6231645329153891, colsample_bytree=0.653791141555321, reg_lambda=0.5294298129797259, reg_alpha=0.5323702695682107, gamma=0.7794550347818443; native Embarked/Pclass categories; no extra features or scaling; seed=42 | 0.83836 ± 0.02245 | 0.76076 |
| mlp_v1 | MLP: Linear(32) → ReLU → Linear(1); Adam, learning_rate=0.001, epochs=30, batch_size=64; OHE and scaling; no extra features; seed=42 | 0.81818 ± 0.02294 | 0.77033 |
| mlp_v2 | MLP: 3 hidden layers of 128, ReLU, one output; Adam, learning_rate=0.001, epochs=30, batch_size=64; OHE and scaling; seed=42 | 0.81930 ± 0.01727 | 0.77033 |
| mlp_v3 | MLP: Linear(128) → BatchNorm → ReLU → Linear(1); Adam, learning_rate=0.001, epochs=30, batch_size=64; OHE and scaling; seed=42 | 0.82041 ± 0.01759 | 0.78468 |
| mlp_v4 | MLP: Linear(128) → BatchNorm → ReLU → Dropout(0.3) → Linear(1); Adam, learning_rate=0.001, epochs=30, batch_size=64; OHE and scaling; seed=42 | 0.82154 ± 0.01766 | 0.78229 |
| mlp_v5 | MLP: Linear(128) → BatchNorm → GELU → Dropout(0.3) → Linear(1); Adam, learning_rate=0.001, epochs=100, batch_size=64; OHE and scaling; seed=42 | 0.82715 ± 0.02295 | 0.77751 |
| mlp_v6 | MLP: Linear(128) → BatchNorm → GELU → Dropout(0.3) → Linear(1); AdamW, weight_decay=0.01, learning_rate=0.001, cosine scheduler (min_lr=0.00001), epochs=100, batch_size=64; OHE and scaling; seed=42 | 0.82264 ± 0.01752 | 0.78708 |
| mlp_v7_embeddings | MLP: embeddings (dim=4) for Sex/Pclass/Embarked, hidden_size=128, BatchNorm, GELU, dropout=0.3; AdamW, weight_decay=0.01, learning_rate=0.001, cosine scheduler (min_lr=0.00001), epochs=100, batch_size=64; numeric scaling; seed=42 | 0.82377 ± 0.01851 | 0.77990 |
| ensemble_xgb_lgbm_cat_soft_v1 | Equal-weight probability averaging (soft voting): xgboost_v1 + lightgbm_v1_optuna + cat_boost_v1; shared 5-fold OOF; seed=42 | 0.843977 ± 0.014990 | 0.76555 |
| ensemble_xgb_lgbm_cat_hard_v1 | Majority class voting (hard voting): xgboost_v1 + lightgbm_v1_optuna + cat_boost_v1; shared 5-fold OOF; seed=42 | 0.846218 ± 0.015513 | 0.76315 |
| ensemble_xgb_lgbm_cat_stacking_ridge_v1 | Stacking: xgboost_v1 + lightgbm_v1_optuna + cat_boost_v1; Ridge (alpha=1.0); 5 outer folds, 3 inner folds; seed=42 | 0.840625 ± 0.018350 | 0.77033 |
