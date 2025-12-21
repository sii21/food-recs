# food-recs

Система рекомендации товаров для ресторана на основе анализа корзин

## Описание проекта

### Задача

Рекомендация товаров по текущей корзине пользователя (item-to-item recommendations). Когда пользователь добавляет товары в корзину, система предлагает дополнительные позиции на основе анализа исторических заказов

### Данные

- **orders_with_status.csv** — ~39K позиций заказов за 1 день (8430 пользователей, 825 товаров, 9512 заказов)
  - Поля: дата, profile_id, order_id, item_id, название, цена, количество, статус
- **products.csv** — справочник товаров (ID, название, описание, категория)

Данные управляются через DVC. После клонирования выполните `dvc pull` для загрузки

### Алгоритмы

1. **TopPopular** — baseline, топ популярных товаров
2. **CooccurrenceLift** — ассоциативные правила на корзинах с метрикой Lift
3. **Item2Vec** — эмбеддинги товаров через Word2Vec на последовательностях корзин

### Метрики (leave-one-out evaluation)

| Модель           | Hit@5  | Hit@10  | Hit@20  | MRR    |
| ---------------- | ------ | ------ | ------ | ------ |
| CooccurrenceLift | 0.1531 | 0.2449 | 0.3701 | 0.1053 |
| TopPopular       | 0.1422 | 0.2347 | 0.3442 | 0.1021 |
| Item2Vec         | 0.1007 | 0.1952 | 0.3224 | 0.0695 |

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

- Забрать данные: `dvc pull` (скачает `data/orders_with_status.csv`, `data/products.csv`)
- Залить данные (при необходимости): `dvc push`

### Проверка code quality

```bash
pre-commit run -a
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
├── artifacts/                # Артефакты (под DVC)
│   ├── models/
│   └── plots/
├── pyproject.toml            # Зависимости и настройки
├── .pre-commit-config.yaml
└── README.md
```
