# food-recs

Система рекомендации товаров для ресторана на основе анализа корзин

## Описание проекта

### Задача

Рекомендация товаров по текущей корзине пользователя (item-to-item recommendations). Когда пользователь добавляет товары в корзину, система предлагает дополнительные позиции на основе анализа исторических заказов

### Данные

- **orders.csv** - ~22M позиций заказов за 14+ месяцев (2025-01 -> 2026-03)
  - Поля: created_at, profile_id, order_oms_id, order_item_id, order_item_oms_id, order_item_title, price, amount, sum, order_status_title
- **products.csv** - справочник товаров (ID, название, описание, категория)

Данные управляются через DVC. После клонирования выполните `dvc pull` для загрузки

### Алгоритмы

#### Baseline модели
1. **TopPopular** - топ популярных товаров
2. **CooccurrenceLift** - ассоциативные правила с метрикой NPMI
3. **Item2Vec** - эмбеддинги товаров через Word2Vec
4. **ImplicitALS** - матричная факторизация (ALS)
5. **ImplicitBPR** - Bayesian Personalized Ranking
6. **SessionCooccurrence** - co-occurrence с учетом истории пользователя
7. **ContentBoost** - гибрид co-occurrence + TF-IDF + категории

#### Production-grade модели (новые)
8. **SentenceTransformerBoost** - замена TF-IDF на dense embeddings (`intfloat/multilingual-e5-base`)
   - Комбинирует co-occurrence, категорийную близость и семантические embeddings
   - Автоматический tuning весов через grid search с кэшированием
   - GPU/CPU auto-fallback
   
9. **LGBMEnsemble** - learned ensemble с LambdaRank
   - Обучается на scores базовых моделей + user/item features
   - Feature engineering: popularity, basket stats, category overlap, co-occurrence diversity
   - Оптимизирует ранжирование напрямую через gradient boosting

10. **DebiasedEvaluator** - честная оценка без popularity bias
    - Стратификация по popularity buckets
    - Macro-averaged метрики для fair comparison
    - Выявляет проблемы на редких товарах

### Разбиение данных (Temporal Split)

Вместо случайного leave-one-out используется временное разбиение (задаётся в `configs/data/default.yaml`):

- **Train** - первые **365** дней данных
- **Test** - следующие **30** дней
- **OOT (Out-of-Time)** - последние **30** дней полного ряда дат - проверка устойчивости к временному дрифту

Оценка качества на test/OOT: **leave-one-out** по корзинам (случайный held-out айтем); см. `make_leave_one_out` в `food_recs/data.py`.

### Метрики

**Test** (leave-one-out):

| Модель             | Hit@5      | Hit@10     | Hit@20     | MRR        | Latency (ms) |
| ------------------ | ---------- | ---------- | ---------- | ---------- | ------------ |
| **LGBMEnsemble**   | **0.2596** | **0.3632** | **0.4695** | **0.1765** | 7.1          |
| STBoost            | 0.1762     | 0.2581     | 0.3584     | 0.1214     | 3.1          |
| TopPopular         | 0.1677     | 0.2683     | 0.3986     | 0.1215     | 0.05         |
| CooccurrenceLift   | 0.1217     | 0.1911     | 0.2957     | 0.0805     | 0.05         |
| Item2Vec           | 0.0705     | 0.1473     | 0.2846     | 0.0506     | -            |
| ImplicitBPR        | 0.0327     | 0.0622     | 0.1156     | 0.0170     | -            |
| ImplicitALS        | 0.0304     | 0.0424     | 0.0665     | 0.0140     | -            |

**OOT** (out-of-time, 30 дней после test):

| Модель             | Hit@5      | Hit@10     | Hit@20     | MRR        | OOT Degradation (Hit@5) |
| ------------------ | ---------- | ---------- | ---------- | ---------- | ----------------------- |
| **LGBMEnsemble**   | **0.2581** | **0.3663** | **0.4685** | **0.1753** | -0.6%                   |
| STBoost            | -          | -          | -          | -          | -                       |
| TopPopular         | 0.1279     | 0.1987     | 0.3186     | 0.0899     | -23.7%                  |
| CooccurrenceLift   | 0.1233     | 0.1762     | 0.2571     | 0.0782     | -1.3%                   |
| Item2Vec           | 0.0441     | 0.0908     | 0.1863     | 0.0326     | -37.4%                  |
| ImplicitBPR        | 0.0167     | 0.0440     | 0.0692     | 0.0110     | -48.9%                  |
| ImplicitALS        | 0.0195     | 0.0245     | 0.0405     | 0.0079     | -35.9%                  |

**Debiased метрики** (macro-averaged по popularity buckets):

| Bucket         | Freq Range   | Hit@5  | Hit@10 | MRR    |
| -------------- | ------------ | ------ | ------ | ------ |
| 0 (редкие)     | 1-10         | 0.0122 | 0.0244 | 0.0076 |
| 1              | 10-47        | 0.1169 | 0.1563 | 0.0739 |
| 2              | 47-227       | 0.1558 | 0.2211 | 0.0906 |
| 3              | 227-1415     | 0.1146 | 0.1820 | 0.0779 |
| 4 (популярные) | 1415-746776  | 0.1782 | 0.2609 | 0.1229 |
| **Debiased**   | macro avg    | 0.1155 | 0.1689 | 0.0746 |
| **Standard**   | micro avg    | 0.1762 | 0.2581 | 0.1214 |

### Выводы по результатам

#### 1. LGBMEnsemble - лучшая модель

**Hit@5 = 25.96%** - прирост **+55%** относительно лучшей baseline (TopPopular 16.8%). Learned ranking через LambdaRank комбинирует scores базовых моделей с user/item features. Самый важный признак - `score_STBoost` (gain 1.9M), далее `item_popularity_rank` (794K) и `item_popularity` (385K). Минимальная деградация на OOT (-0.6%) - модель устойчива к временному дрифту

#### 2. SentenceTransformerBoost - прорыв в семантике

**Hit@5 = 17.62%** - превосходит все baseline модели. Dense embeddings (`intfloat/multilingual-e5-base`, 768-dim) улавливают семантическую близость товаров значительно лучше, чем TF-IDF в ContentBoost. Автоматический tuning весов (co-occurrence + category + text) через grid search с кэшированием embeddings

#### 3. CooccurrenceLift - самая стабильная baseline модель

Минимальная деградация на OOT: Hit@5 упал всего на 1.3%. Паттерны совместных покупок (суши + имбирь, пицца + кола) стабильны во времени, в отличие от "хитов продаж" (TopPopular -23.7%)

#### 4. ImplicitALS/BPR - не подходят для задачи

Hit@5 ~= 3%. Матричная факторизация предназначена для User -> Item с десятками взаимодействий. В нашей постановке каждая корзина - анонимный "пользователь" с 2-3 товарами, матрица слишком разреженная (sparsity > 99.99%)

#### 5. Popularity bias

Debiased Hit@5 (11.55%) значительно ниже standard (17.62%) -> -34.5% gap. На редких товарах (freq 1-10) Hit@5 = 1.22% vs 17.82% на популярных. Для улучшения нужны: fallback на category/text similarity для холодных товаров, reranking с учетом diversity

#### 6. Рекомендация для продакшена

**Оптимальная стратегия:**
1. **LGBMEnsemble** - основная модель (26% Hit@5, latency 7 мс)
2. **STBoost** - fallback для холодных товаров (semantic similarity)
3. **TopPopular** - cold start для пустых корзин

**Мониторинг:** отслеживать debiased метрики для выявления bias drift

## Setup

### Требования

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) или pip

### Установка окружения

```bash
# Клонируем репозиторий
git clone https://github.com/sii21/food-recs.git
cd food-recs

# Создаём виртуальное окружение
python -m venv .venv

# Активируем (Windows)
.\.venv\Scripts\activate

# Устанавливаем зависимости
pip install -e .

# Устанавливаем dev-зависимости (pytest)
pip install -e ".[dev]"

# Устанавливаем pre-commit хуки
pre-commit install

# Загружаем данные через DVC
dvc pull
```

### Данные и DVC (Yandex Cloud S3)

Для хранения данных использовалось S3 хранилище Яндекса

- Remote: `s3://food-dfcvdasd/dvc` с endpoint `https://storage.yandexcloud.net`, регион `ru-central1`
- Ключи HMAC храним локально (не в git):

```bash
dvc remote modify ycs3 access_key_id <ACCESS_KEY> --local
dvc remote modify ycs3 secret_access_key <SECRET_KEY> --local
```

- Забрать данные: `dvc pull` (скачает `data/orders.csv`, `data/products.csv`)
- Залить данные (при необходимости): `dvc push`

### Проверка code quality

```bash
pre-commit run -a
```

### Тесты

```bash
# Запуск всех тестов с подробным выводом
pytest tests/ -v

# Запуск с покрытием кода
pytest tests/ --cov=food_recs --cov-report=term-missing

# Запуск конкретного тестового файла
pytest tests/test_models.py -v

# Запуск с фильтрацией по ключевым словам
pytest tests/ -k "test_recommend" -v
```

## Train

Обучение моделей с логированием в MLflow:

```bash
# Запуск MLflow UI для просмотра экспериментов (в отдельном терминале)
mlflow ui --backend-store-uri mlruns

# Или полноценный MLflow сервер с SQLite
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000

# Обучение с дефолтным конфигом
food-recs train

# Или с кастомными параметрами
food-recs train --config_name=train
```

Модели сохраняются в `artifacts/models/`, метрики логируются в MLflow

### Конфигурация

Гиперпараметры задаются в Hydra конфигах:

- `configs/train.yaml` - основной конфиг
- `configs/data/default.yaml` - настройки данных
- `configs/model/default.yaml` - параметры моделей
- `configs/logging/default.yaml` - настройки MLflow

## Infer

Получение рекомендаций:

```bash
# Рекомендации для корзины с item_id 98, 183
food-recs infer --basket="[98, 183]" --model=cooccurrence --top_k=5
```

## Visualize

Генерация графиков качества:

```bash
food-recs visualize
```

Графики сохраняются в `artifacts/plots/`

## Demo

Интерактивное Streamlit приложение:

```bash
food-recs app
```

Или напрямую:

```bash
streamlit run food_recs/app.py
```

## Структура проекта

```
food-recs/
├── .github/workflows/        # CI (lint + pytest)
│   └── ci.yml
├── configs/                  # Hydra конфиги
│   ├── train.yaml
│   ├── data/
│   ├── model/
│   └── logging/
├── data/                     # Данные (под DVC)
├── food_recs/                # Python пакет
│   ├── __init__.py
│   ├── commands.py           # CLI entry point
│   ├── data.py               # Загрузка данных
│   ├── models.py             # Baseline модели
│   ├── sentence_transformer_model.py  # STBoost (production)
│   ├── lgbm_ensemble.py      # LGBM ensemble (production)
│   ├── evaluation/           # Evaluation модули
│   │   └── debiased_metrics.py  # Debiased evaluation
│   ├── features/             # Feature engineering
│   │   ├── user_features.py  # User-side features
│   │   └── item_features.py  # Item-side features
│   ├── training.py           # Пайплайн обучения
│   ├── inference.py          # Инференс
│   ├── visualization.py      # Визуализация
│   └── app.py                # Streamlit приложение
├── tests/                    # Unit-тесты
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_new_models.py    # Тесты production моделей
│   ├── test_training.py
│   └── test_visualization.py
├── artifacts/                # Артефакты (под DVC)
│   ├── models/
│   └── plots/
├── pyproject.toml            # Зависимости и настройки
├── .pre-commit-config.yaml
└── README.md
```

## Зависимости

**Core:**
- `pandas`, `numpy`, `scipy` - data processing
- `scikit-learn` - TF-IDF, metrics
- `gensim` - Word2Vec (Item2Vec)
- `implicit` - ALS/BPR

**Production models:**
- `sentence-transformers` - dense embeddings (STBoost)
- `lightgbm` - gradient boosting (LGBM ensemble)
- `torch` - backend для sentence transformers

**Infrastructure:**
- `hydra-core` - config management
- `mlflow` - experiment tracking
- `dvc` - data version control
- `streamlit` - demo app
