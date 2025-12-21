"""Data loading and preprocessing"""

import shutil
import subprocess
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig


def download_data(source_dir: str = "dvc-storage", target_dir: str = "data") -> None:
    """Copy data files from a local storage directory into target data dir

    Args:
        source_dir: Directory where data files are stored (e.g. local DVC remote)
        target_dir: Directory to place files for training/inference
    """
    src = Path(source_dir)
    dst = Path(target_dir)
    dst.mkdir(parents=True, exist_ok=True)

    expected_files = ["orders_with_status.csv", "products.csv"]
    missing = [f for f in expected_files if not (dst / f).exists()]

    if not missing:
        return

    for filename in missing:
        src_file = src / filename
        if src_file.exists():
            shutil.copy2(src_file, dst / filename)
        else:
            raise FileNotFoundError(
                f"Не найден {src_file}. Скопируйте данные в {source_dir} или выполните dvc pull"
            )


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


def ensure_data_available(data_dir: str = "data") -> None:
    """Ensure required data files are present, pulling via DVC if needed

    Args:
        data_dir: Local data directory
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    required = [data_path / "orders_with_status.csv", data_path / "products.csv"]
    missing = [str(p) for p in required if not p.exists()]

    if not missing:
        return

    # Try to pull missing data via DVC
    dvc_pull(missing)

    # If still missing, raise explicit error
    still_missing = [p for p in required if not p.exists()]
    if still_missing:
        raise FileNotFoundError(
            f"Не найдены данные: {', '.join(str(p) for p in still_missing)}. "
            "Убедитесь, что настроен DVC remote и выполните dvc pull."
        )


def load_orders(cfg: DictConfig) -> pd.DataFrame:
    """Load orders data from CSV

    Args:
        cfg: Hydra config with data settings

    Returns:
        DataFrame with order data
    """
    data_path = Path(cfg.data.data_path)
    df = pd.read_csv(data_path, sep=cfg.data.separator)
    return df


def filter_successful_orders(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Filter only successful orders

    Args:
        df: Raw orders DataFrame
        cfg: Config with success statuses

    Returns:
        Filtered DataFrame
    """
    success_statuses = list(cfg.data.success_statuses)
    return df[df["order_status_title"].isin(success_statuses)]


def create_baskets(df: pd.DataFrame, cfg: DictConfig) -> list[list[int]]:
    """Create baskets from orders

    Args:
        df: Filtered orders DataFrame
        cfg: Config with min basket size

    Returns:
        List of baskets (each basket is list of item IDs)
    """
    baskets = df.groupby("order_oms_id")["order_item_id"].apply(list).tolist()
    min_size = cfg.data.min_basket_size
    return [b for b in baskets if len(b) >= min_size]


def create_item_mapping(df: pd.DataFrame) -> dict[int, str]:
    """Create mapping from item_id to item name

    Args:
        df: Orders DataFrame

    Returns:
        Dict mapping item_id -> item_title
    """
    return (
        df.drop_duplicates("order_item_id").set_index("order_item_id")["order_item_title"].to_dict()
    )


def prepare_data(cfg: DictConfig) -> tuple[list[list[int]], dict[int, str], pd.DataFrame]:
    """Full data preparation pipeline

    Args:
        cfg: Hydra config

    Returns:
        Tuple of (baskets, item_mapping, filtered_df)
    """
    df = load_orders(cfg)
    df_filtered = filter_successful_orders(df, cfg)
    baskets = create_baskets(df_filtered, cfg)
    item_mapping = create_item_mapping(df_filtered)
    return baskets, item_mapping, df_filtered
