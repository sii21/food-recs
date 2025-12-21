"""Training pipeline with MLflow logging"""

import pickle
import subprocess
from pathlib import Path

import mlflow
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from food_recs.data import ensure_data_available, prepare_data
from food_recs.models import (
    CooccurrenceLiftRecommender,
    Item2VecRecommender,
    TopPopularRecommender,
)


def get_git_commit_id() -> str:
    """Get current git commit hash"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()[:8]
    except Exception:
        return "unknown"


def leave_one_out_split(
    baskets: list[list[int]], test_ratio: float = 0.2, seed: int = 42
) -> tuple[list[list[int]], list[tuple[list[int], int]]]:
    """Leave-one-out split for evaluation

    Args:
        baskets: List of baskets
        test_ratio: Fraction of baskets to use for test
        seed: Random seed

    Returns:
        Tuple of (train_baskets, test_data) where test_data is list of (input_basket, held_out_item)
    """
    np.random.seed(seed)
    n_test = int(len(baskets) * test_ratio)

    indices = np.random.permutation(len(baskets))
    test_indices = set(indices[:n_test])

    train_baskets = []
    test_data = []

    for idx, basket in enumerate(baskets):
        if idx in test_indices and len(basket) >= 2:
            held_out_idx = np.random.randint(len(basket))
            held_out_item = basket[held_out_idx]
            input_basket = basket[:held_out_idx] + basket[held_out_idx + 1 :]
            test_data.append((input_basket, held_out_item))
            train_baskets.append(basket)
        else:
            train_baskets.append(basket)

    return train_baskets, test_data


def evaluate_model(
    model, test_data: list[tuple[list[int], int]], k_values: list[int]
) -> dict[str, float]:
    """Evaluate model with Hit@K and MRR metrics

    Args:
        model: Trained recommender model
        test_data: List of (input_basket, held_out_item) tuples
        k_values: List of K values for Hit@K

    Returns:
        Dict with metric names and values
    """
    results = {f"hit@{k}": 0.0 for k in k_values}
    mrr_sum = 0.0

    for input_basket, held_out_item in tqdm(test_data, desc="Evaluating"):
        recs = model.recommend(input_basket, k=max(k_values))

        if held_out_item in recs:
            rank = recs.index(held_out_item) + 1
            mrr_sum += 1 / rank

        for k in k_values:
            if held_out_item in recs[:k]:
                results[f"hit@{k}"] += 1

    n_test = len(test_data)
    for key in results:
        results[key] /= n_test
    results["mrr"] = mrr_sum / n_test

    return results


def train_models(cfg: DictConfig) -> dict[str, dict[str, float]]:
    """Main training pipeline

    Args:
        cfg: Hydra config

    Returns:
        Dict with evaluation results for each model
    """
    print("=" * 60)
    print("TRAINING RECOMMENDATION MODELS")
    print("=" * 60)

    # Setup MLflow (optional - skip if server not available)
    mlflow_enabled = False
    try:
        mlflow.set_tracking_uri(cfg.logging.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.logging.mlflow.experiment_name)
        mlflow_enabled = True
        print(f"MLflow tracking enabled: {cfg.logging.mlflow.tracking_uri}")
    except Exception as e:
        print(f"⚠️ MLflow not available ({e}), continuing without logging")

    # Ensure data present via DVC
    print("\nEnsuring data is available (dvc pull if needed)...")
    ensure_data_available()

    # Prepare data
    print("Loading and preparing data...")
    baskets, item_mapping, df = prepare_data(cfg)
    print(f"Loaded {len(baskets)} baskets with {cfg.data.min_basket_size}+ items")

    # Train/test split
    print(f"\nSplitting data ({1-cfg.test_ratio:.0%}/{cfg.test_ratio:.0%})...")
    train_baskets, test_data = leave_one_out_split(baskets, cfg.test_ratio, cfg.seed)
    print(f"Train baskets: {len(train_baskets)}, Test samples: {len(test_data)}")

    # Create models directory
    models_dir = Path(cfg.model.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    k_values = list(cfg.model.evaluation.k_values)
    all_results = {}

    # Start MLflow run if enabled
    run_context = mlflow.start_run() if mlflow_enabled else None
    if run_context:
        run_context.__enter__()
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
        mlflow.log_param("git_commit", get_git_commit_id())
        mlflow.log_param("n_baskets", len(baskets))
        mlflow.log_param("n_items", len(item_mapping))

    # Train and evaluate each model
    models_to_train = []
    if cfg.model.popularity.enabled:
        models_to_train.append(("TopPopular", TopPopularRecommender()))
    if cfg.model.cooccurrence.enabled:
        models_to_train.append(
            (
                "CooccurrenceLift",
                CooccurrenceLiftRecommender(cfg.model.cooccurrence.min_support),
            )
        )
    if cfg.model.item2vec.enabled:
        models_to_train.append(("Item2Vec", Item2VecRecommender(cfg)))

    for name, model in models_to_train:
        print(f"\n{'=' * 40}")
        print(f"Training {name}...")
        model.fit(train_baskets)

        print(f"Evaluating {name}...")
        metrics = evaluate_model(model, test_data, k_values)
        all_results[name] = metrics

        # Log metrics to MLflow (replace @ with _at_ for valid metric names)
        if mlflow_enabled:
            for metric_name, value in metrics.items():
                safe_metric_name = metric_name.replace("@", "_at_")
                mlflow.log_metric(f"{name}_{safe_metric_name}", value)

        print(f"\n{name} Results:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")

        # Save model
        model_path = models_dir / f"{name.lower()}_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    # Save item mapping and results
    with open(models_dir / "item_mapping.pkl", "wb") as f:
        pickle.dump(item_mapping, f)

    with open(models_dir / "evaluation_results.pkl", "wb") as f:
        pickle.dump(all_results, f)

    with open(models_dir / "train_baskets.pkl", "wb") as f:
        pickle.dump(train_baskets, f)

    # Log artifacts and end run
    if mlflow_enabled and run_context:
        if cfg.logging.mlflow.log_models:
            mlflow.log_artifacts(str(models_dir), "models")
        run_context.__exit__(None, None, None)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Model':<20} {'Hit@5':<10} {'Hit@10':<10} {'Hit@20':<10} {'MRR':<10}")
    print("-" * 60)
    for name, metrics in all_results.items():
        print(
            f"{name:<20} {metrics['hit@5']:.4f}     {metrics['hit@10']:.4f}     "
            f"{metrics['hit@20']:.4f}     {metrics['mrr']:.4f}"
        )

    print(f"\nModels saved to: {models_dir}")
    print("Training complete!")

    return all_results
