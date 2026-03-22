"""Data loading and preprocessing"""

import pickle
import subprocess
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm


def dvc_pull(targets: list[str] | None = None) -> None:
    """Run dvc pull for specified targets (or all)

    Args:
        targets: Specific files/dirs to pull
    """
    cmd = ["dvc", "pull"]
    if targets:
        cmd.extend(targets)
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("DVC не установлен или не найден в PATH. Установите dvc.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"dvc pull завершился с ошибкой: {exc}") from exc


def ensure_data_available(cfg: DictConfig) -> None:
    """Ensure required data files are present, pulling via DVC if needed

    Args:
        cfg: Hydra config with data settings
    """
    data_path = Path(cfg.data.data_path)
    if data_path.exists():
        return

    dvc_pull([str(data_path)])

    if not data_path.exists():
        raise FileNotFoundError(
            f"Не найден файл данных: {data_path}. "
            "Убедитесь, что настроен DVC remote и выполните dvc pull."
        )


def _validate_columns(df: pd.DataFrame, required: list[str], source: str) -> None:
    """Проверяет наличие обязательных колонок в DataFrame

    Args:
        df: DataFrame для проверки
        required: Список обязательных колонок
        source: Описание источника данных (для сообщения об ошибке)

    Raises:
        ValueError: если обязательные колонки отсутствуют
    """
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"В {source} отсутствуют обязательные колонки: {sorted(missing)}. "
            f"Доступные колонки: {sorted(df.columns)}"
        )


def load_orders_chunked(
    cfg: DictConfig,
) -> tuple[dict[int, list[int]], dict[int, pd.Timestamp], dict[int, str]]:
    """Load orders from CSV by chunks, building baskets incrementally

    Args:
        cfg: Hydra config with data settings

    Returns:
        Tuple of (order_baskets, order_dates, item_mapping) where
        order_baskets maps order_oms_id -> list of item IDs,
        order_dates maps order_oms_id -> earliest timestamp,
        item_mapping maps item_id -> item_title
    """
    data_path = Path(cfg.data.data_path)
    chunksize = cfg.data.get("chunksize", 500_000)
    success_statuses = set(cfg.data.success_statuses)
    item_id_col = cfg.data.get("item_id_column", "order_item_oms_id")
    date_col = cfg.data.get("date_column", "created_at")

    required_columns = [
        "order_status_title",
        "order_oms_id",
        item_id_col,
        "order_item_title",
        date_col,
    ]

    order_baskets: dict[int, list[int]] = defaultdict(list)
    order_dates: dict[int, pd.Timestamp] = {}
    item_mapping: dict[int, str] = {}

    reader = pd.read_csv(data_path, sep=cfg.data.separator, chunksize=chunksize)

    schema_validated = False
    for chunk in tqdm(reader, desc="Loading data"):
        if not schema_validated:
            _validate_columns(chunk, required_columns, str(data_path))
            schema_validated = True
        success = chunk[chunk["order_status_title"].isin(success_statuses)]
        if len(success) == 0:
            continue

        success = success.copy()
        success["_date"] = pd.to_datetime(success[date_col], format="mixed")

        for oid, group in success.groupby("order_oms_id"):
            items = group[item_id_col].dropna().astype(int).tolist()
            order_baskets[oid].extend(items)

            min_dt = group["_date"].min()
            if oid not in order_dates or min_dt < order_dates[oid]:
                order_dates[oid] = min_dt

        # Item mapping
        names = (
            success.drop_duplicates(item_id_col)
            .set_index(item_id_col)["order_item_title"]
            .to_dict()
        )
        for iid, title in names.items():
            if iid not in item_mapping:
                item_mapping[int(iid)] = title

    return dict(order_baskets), order_dates, item_mapping


def temporal_split(
    order_baskets: dict[int, list[int]],
    order_dates: dict[int, pd.Timestamp],
    train_days: int = 30,
    test_days: int = 7,
    oot_days: int = 7,
    min_basket_size: int = 2,
) -> tuple[
    list[list[int]],
    list[list[int]],
    list[list[int]],
    pd.Timestamp,
    pd.Timestamp,
    pd.Timestamp,
    pd.Timestamp,
]:
    """Split baskets by time into train / test / OOT

    Args:
        order_baskets: order_oms_id -> list of item IDs
        order_dates: order_oms_id -> timestamp
        train_days: Number of days for training period
        test_days: Number of days for test period
        oot_days: Number of days for OOT period (from the end of data)
        min_basket_size: Minimum basket size to include

    Returns:
        Tuple of (train_baskets, test_baskets, oot_baskets,
                  train_end, test_end, oot_start, data_end)
    """
    if not order_dates:
        raise ValueError("No order dates provided")

    dates_series = pd.Series(order_dates)
    data_start = dates_series.min()
    data_end = dates_series.max()

    train_end = data_start + timedelta(days=train_days)
    test_end = train_end + timedelta(days=test_days)
    oot_start = data_end - timedelta(days=oot_days)

    train_baskets = []
    test_baskets = []
    oot_baskets = []

    for oid, dt in order_dates.items():
        basket = order_baskets[oid]
        # Deduplicate items within basket
        basket = list(dict.fromkeys(basket))
        if len(basket) < min_basket_size:
            continue

        if dt < train_end:
            train_baskets.append(basket)
        elif dt < test_end:
            test_baskets.append(basket)

        if dt >= oot_start:
            oot_baskets.append(basket)

    print(
        f"Data period: {data_start.date()} → {data_end.date()} ({(data_end - data_start).days} days)"
    )
    print(f"Train: {data_start.date()} → {train_end.date()} | {len(train_baskets):,} baskets")
    print(f"Test:  {train_end.date()} → {test_end.date()} | {len(test_baskets):,} baskets")
    print(f"OOT:   {oot_start.date()} → {data_end.date()} | {len(oot_baskets):,} baskets")

    return (
        train_baskets,
        test_baskets,
        oot_baskets,
        train_end,
        test_end,
        oot_start,
        data_end,
    )


def make_leave_one_out(baskets: list[list[int]], seed: int = 42) -> list[tuple[list[int], int]]:
    """Create leave-one-out test data from baskets

    For each basket with >=2 items, hold out one random item

    Args:
        baskets: List of baskets (each is list of item IDs)
        seed: Random seed

    Returns:
        List of (input_basket, held_out_item) tuples
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    test_data = []

    for basket in baskets:
        if len(basket) < 2:
            continue
        held_out_idx = rng.integers(len(basket))
        held_out_item = basket[held_out_idx]
        input_basket = basket[:held_out_idx] + basket[held_out_idx + 1 :]
        test_data.append((input_basket, held_out_item))

    return test_data


def prepare_data(
    cfg: DictConfig,
) -> tuple[list[list[int]], list[list[int]], list[list[int]], dict[int, str]]:
    """Full data preparation pipeline with temporal split

    Args:
        cfg: Hydra config

    Returns:
        Tuple of (train_baskets, test_baskets, oot_baskets, item_mapping)
    """
    cache_path = Path(cfg.data.get("cache_path", "artifacts/data_cache.pkl"))

    if cache_path.exists():
        print(f"Loading cached data from {cache_path}...")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        return (
            cached["train_baskets"],
            cached["test_baskets"],
            cached["oot_baskets"],
            cached["item_mapping"],
        )

    print("Loading data from CSV (this may take a few minutes for large files)...")
    order_baskets, order_dates, item_mapping = load_orders_chunked(cfg)
    print(f"Loaded {len(order_baskets):,} orders, {len(item_mapping):,} unique items")

    ts_cfg = cfg.data.temporal_split
    train_baskets, test_baskets, oot_baskets, *_ = temporal_split(
        order_baskets,
        order_dates,
        train_days=ts_cfg.train_days,
        test_days=ts_cfg.test_days,
        oot_days=ts_cfg.oot_days,
        min_basket_size=cfg.data.min_basket_size,
    )

    # Cache processed data
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(
            {
                "train_baskets": train_baskets,
                "test_baskets": test_baskets,
                "oot_baskets": oot_baskets,
                "item_mapping": item_mapping,
            },
            f,
        )
    print(f"Cached processed data to {cache_path}")

    return train_baskets, test_baskets, oot_baskets, item_mapping
