# MLPipeline

Проект для соревнования Kaggle [Titanic — Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic).

Положите `train.csv` и `test.csv` в папку `raw_data/` в корне проекта.
Для запуска из корня проекта выполните `uv sync --locked`, затем `uv run --locked python src/main.py`.

| Эксперимент | Модель | Обработка | Валидация | Train accuracy | Validation accuracy | Kaggle score |
| --- | --- | --- | --- | ---: | ---: | ---: |
| logreg_baseline_v1 | LogisticRegression, параметры по умолчанию | Медианы Age/Fare, мода Embarked, OHE, scaling; без Name/Ticket/Cabin | Holdout 25%, stratify, seed=42 | 0,8114 | 0,7758 | 0,77272 |
