# food-recs

Система рекомендации товаров для ресторана на основе анализа корзин

## Описание проекта

### Задача

Рекомендация товаров по текущей корзине пользователя (item-to-item recommendations). Когда пользователь добавляет товары в корзину, система предлагает дополнительные позиции на основе анализа исторических заказов

### Данные

- **orders.csv** - ~22M позиций заказов за 14+ месяцев (2025-01 → 2026-03)
  - Поля: created_at, profile_id, order_oms_id, order_item_id, order_item_oms_id, order_item_title, price, amount, sum, order_status_title
- **products.csv** - справочник товаров (ID, название, описание, категория)

Данные управляются через DVC. После клонирования выполните `dvc pull` для загрузки

### Алгоритмы

1. **TopPopular** - baseline, топ популярных товаров
2. **CooccurrenceLift** - ассоциативные правила на корзинах с метрикой Lift
3. **Item2Vec** - эмбеддинги товаров через Word2Vec на последовательностях корзин
4. **ImplicitALS** - матричная факторизация (ALS) из библиотеки [implicit](https://github.com/benfred/implicit)
5. **ImplicitBPR** - Bayesian Personalized Ranking из библиотеки [implicit](https://github.com/benfred/implicit)

### Разбиение данных (Temporal Split)

Вместо случайного leave-one-out используется временное разбиение:

- **Train** - первые 30 дней данных (январь 2025)
- **Test** - следующие 7 дней (начало февраля 2025)
- **OOT (Out-of-Time)** - последние 7 дней данных (март 2026) - проверка устойчивости к временному дрифту

### Метрики (Temporal Split + Leave-One-Out evaluation)

**Test** (следующая неделя после обучения):

| Модель           | Hit@5  | Hit@10 | Hit@20 | MRR    |
| ---------------- | ------ | ------ | ------ | ------ |
| TopPopular       | 0.1677 | 0.2683 | 0.3986 | 0.1215 |
| CooccurrenceLift | 0.1217 | 0.1911 | 0.2957 | 0.0805 |
| Item2Vec         | 0.0759 | 0.1540 | 0.2879 | 0.0524 |
| ImplicitBPR      | 0.0331 | 0.0650 | 0.1058 | 0.0172 |
| ImplicitALS      | 0.0304 | 0.0424 | 0.0665 | 0.0140 |

**OOT** (через 14 месяцев после обучения):

| Модель           | Hit@5  | Hit@10 | Hit@20 | MRR    |
| ---------------- | ------ | ------ | ------ | ------ |
| TopPopular       | 0.1279 | 0.1987 | 0.3186 | 0.0899 |
| CooccurrenceLift | 0.1233 | 0.1762 | 0.2571 | 0.0782 |
| Item2Vec         | 0.0465 | 0.0973 | 0.1901 | 0.0330 |
| ImplicitBPR      | 0.0331 | 0.0440 | 0.0548 | 0.0124 |
| ImplicitALS      | 0.0195 | 0.0245 | 0.0405 | 0.0079 |

**OOT Degradation** (% падения метрик при переходе Test → OOT):

| Модель           | Hit@5  | Hit@10 | MRR    |
| ---------------- | ------ | ------ | ------ |
| CooccurrenceLift | −1.3%  | −7.8%  | −2.8%  |
| TopPopular       | −23.7% | −25.9% | −26.0% |
| Item2Vec         | −38.7% | −36.8% | −36.9% |
| ImplicitBPR      | −0.0%  | −32.3% | −27.8% |
| ImplicitALS      | −35.9% | −42.3% | −43.4% |

### Выводы по результатам

#### 1. Сравнение с библиотечными моделями

`ImplicitALS` (Alternating Least Squares) и `ImplicitBPR` (Bayesian Personalized Ranking) из библиотеки [implicit](https://github.com/benfred/implicit) показали наихудшие результаты (Hit@5 ≈ 3%).

Причина: матричная факторизация предназначена для задачи **User → Item**, где у каждого пользователя десятки-сотни взаимодействий. В нашей постановке каждая корзина - это анонимный "пользователь" с 2-3 товарами. Матрица взаимодействий получается крайне разреженной (sparsity > 99.99%), и латентные факторы не сходятся. Этот результат подтверждает, что для **сессионных (basket-based) рекомендаций** без профиля пользователя коллаборативная фильтрация не применима напрямую

#### 2. TopPopular - лучшая абсолютная модель

`TopPopular` лидирует по всем метрикам на Test (Hit@5 = 16.8%, MRR = 12.2%). В ресторанном домене значительная доля заказов приходится на одни и те же популярные позиции ("хиты продаж"), поэтому рекомендация самого частотного работает лучше сложных алгоритмов

#### 3. CooccurrenceLift - самая стабильная модель

`CooccurrenceLift` демонстрирует минимальную деградацию на OOT: Hit@5 упал всего на 1.3% (с 12.2% до 12.3%), а MRR - на 2.8%. Для сравнения, `TopPopular` потерял 24-26% по всем метрикам, поскольку "хиты продаж" меняются со временем (сезонность, обновление меню). Ассоциативные правила (Lift) устойчивее к дрифту, т.к. паттерны совместных покупок (суши + имбирь, пицца + кола) стабильны во времени

#### 4. Item2Vec - промежуточный результат

`Item2Vec` занял среднюю позицию (Hit@5 = 7.6% на Test). Модель улавливает контекст совместных покупок, но уступает прямому Lift'у - корзины слишком короткие (2-4 товара) для качественного обучения Word2Vec эмбеддингов. При этом на Hit@20 Item2Vec приближается к CooccurrenceLift (28.8% vs 29.6%), что говорит о хорошем recall на дальних позициях

#### 5. Рекомендация для продакшена

Оптимальная стратегия - **гибрид**: `CooccurrenceLift` как основной алгоритм (стабильность + персонализация по корзине) с фоллбэком на `TopPopular` для холодного старта (если товары из корзины отсутствуют в матрице Lift)

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
│   ├── models.py             # Модели рекомендаций
│   ├── training.py           # Пайплайн обучения
│   ├── inference.py          # Инференс
│   ├── visualization.py      # Визуализация
│   └── app.py                # Streamlit приложение
├── tests/                    # Unit-тесты
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_training.py
│   └── test_visualization.py
├── artifacts/                # Артефакты (под DVC)
│   ├── models/
│   └── plots/
├── pyproject.toml            # Зависимости и настройки
├── .pre-commit-config.yaml
└── README.md
```
