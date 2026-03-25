"""Training pipeline with MLflow logging"""

import pickle
import subprocess
import time
from pathlib import Path

import mlflow
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from food_recs.data import ensure_data_available, make_leave_one_out, prepare_data
from food_recs.models import (
    ContentBoostRecommender,
    CooccurrenceLiftRecommender,
    ImplicitALSRecommender,
    ImplicitBPRRecommender,
    Item2VecRecommender,
    SessionCooccurrenceRecommender,
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


def evaluate_model(
    model,
    test_data: list[tuple[list[int], int]],
    k_values: list[int],
    split_name: str = "test",
) -> dict[str, float]:
    """Evaluate model with Hit@K and MRR metrics

    Args:
        model: Trained recommender model
        test_data: List of (input_basket, held_out_item) tuples
        k_values: List of K values for Hit@K
        split_name: Prefix for metric names (test or oot)

    Returns:
        Dict with metric names and values
    """
    if not test_data:
        return {f"{split_name}_hit@{k}": 0.0 for k in k_values} | {f"{split_name}_mrr": 0.0}

    results = {f"{split_name}_hit@{k}": 0.0 for k in k_values}
    mrr_sum = 0.0

    for entry in tqdm(test_data, desc=f"Evaluating ({split_name})"):
        input_basket, held_out_item = entry[0], entry[1]
        profile_id = entry[2] if len(entry) > 2 else None

        if isinstance(model, SessionCooccurrenceRecommender) and profile_id is not None:
            recs = model.recommend(input_basket, k=max(k_values), user_id=profile_id)
        else:
            recs = model.recommend(input_basket, k=max(k_values))

        if held_out_item in recs:
            rank = recs.index(held_out_item) + 1
            mrr_sum += 1 / rank

        for k in k_values:
            if held_out_item in recs[:k]:
                results[f"{split_name}_hit@{k}"] += 1

    n_test = len(test_data)
    for key in results:
        results[key] /= n_test
    results[f"{split_name}_mrr"] = mrr_sum / n_test

    return results


def _build_models_list(cfg: DictConfig) -> list[tuple[str, object]]:
    """Build list of (name, model) pairs based on config"""
    models = []
    if cfg.model.popularity.enabled:
        models.append(("TopPopular", TopPopularRecommender()))
    if cfg.model.cooccurrence.enabled:
        models.append(
            (
                "CooccurrenceLift",
                CooccurrenceLiftRecommender(cfg.model.cooccurrence.min_support),
            )
        )
    if cfg.model.item2vec.enabled:
        models.append(("Item2Vec", Item2VecRecommender(cfg)))
    if cfg.model.implicit_als.enabled:
        als_cfg = cfg.model.implicit_als
        models.append(
            (
                "ImplicitALS",
                ImplicitALSRecommender(
                    factors=als_cfg.factors,
                    regularization=als_cfg.regularization,
                    iterations=als_cfg.iterations,
                    alpha=als_cfg.alpha,
                ),
            )
        )
    if cfg.model.implicit_bpr.enabled:
        bpr_cfg = cfg.model.implicit_bpr
        models.append(
            (
                "ImplicitBPR",
                ImplicitBPRRecommender(
                    factors=bpr_cfg.factors,
                    learning_rate=bpr_cfg.learning_rate,
                    regularization=bpr_cfg.regularization,
                    iterations=bpr_cfg.iterations,
                ),
            )
        )
    if cfg.model.get("session_cooccurrence", {}).get("enabled", False):
        sc_cfg = cfg.model.session_cooccurrence
        models.append(
            (
                "SessionCooccurrence",
                SessionCooccurrenceRecommender(
                    history_weight=sc_cfg.get("history_weight", 0.3),
                    min_support=sc_cfg.get("min_support", 2),
                ),
            )
        )
    if cfg.model.get("content_boost", {}).get("enabled", False):
        cb_cfg = cfg.model.content_boost
        models.append(
            (
                "ContentBoost",
                ContentBoostRecommender(
                    cooc_weight=cb_cfg.get("cooc_weight", 0.6),
                    category_weight=cb_cfg.get("category_weight", 0.25),
                    text_weight=cb_cfg.get("text_weight", 0.15),
                    min_support=cb_cfg.get("min_support", 2),
                ),
            )
        )
    return models


def train_models(cfg: DictConfig) -> dict[str, dict[str, float]]:
    """Main training pipeline with temporal split and OOT evaluation

    Args:
        cfg: Hydra config

    Returns:
        Dict with evaluation results for each model
    """
    print("=" * 60)
    print("TRAINING RECOMMENDATION MODELS")
    print("=" * 60)

    # Setup MLflow (optional)
    mlflow_enabled = False
    try:
        mlflow.set_tracking_uri(cfg.logging.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.logging.mlflow.experiment_name)
        mlflow_enabled = True
        print(f"MLflow tracking enabled: {cfg.logging.mlflow.tracking_uri}")
    except Exception as e:
        print(f"MLflow not available ({e}), continuing without logging")

    # Ensure data present via DVC
    print("\nEnsuring data is available...")
    ensure_data_available(cfg)

    # Prepare data with temporal split
    print("Loading and preparing data...")
    (
        train_baskets, test_baskets, oot_baskets, item_mapping,
        user_histories, product_catalog, test_profiles, oot_profiles,
    ) = prepare_data(cfg)
    print(f"Train: {len(train_baskets):,} baskets")
    print(f"Test:  {len(test_baskets):,} baskets")
    print(f"OOT:   {len(oot_baskets):,} baskets")
    print(f"Items: {len(item_mapping):,}")
    if user_histories:
        print(f"User histories: {len(user_histories):,} profiles")
    if not product_catalog.empty:
        print(f"Product catalog: {len(product_catalog):,} items")

    # Create leave-one-out evaluation data from test and OOT baskets
    seed = cfg.seed
    test_data = make_leave_one_out(test_baskets, seed=seed, profile_ids=test_profiles or None)
    oot_data = make_leave_one_out(oot_baskets, seed=seed, profile_ids=oot_profiles or None)
    print(f"\nLeave-one-out samples: test={len(test_data):,}, oot={len(oot_data):,}")

    # Create models directory
    models_dir = Path(cfg.model.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    k_values = list(cfg.model.evaluation.k_values)
    all_results = {}

    # Build models list
    models_to_train = _build_models_list(cfg)

    # MLflow run with proper context manager
    run_ctx = mlflow.start_run() if mlflow_enabled else None
    try:
        if run_ctx:
            run_ctx.__enter__()
            flat_cfg = OmegaConf.to_container(cfg, resolve=True)
            mlflow.log_params(_flatten_dict(flat_cfg))
            mlflow.log_param("git_commit", get_git_commit_id())
            mlflow.log_param("n_train_baskets", len(train_baskets))
            mlflow.log_param("n_test_baskets", len(test_baskets))
            mlflow.log_param("n_oot_baskets", len(oot_baskets))
            mlflow.log_param("n_items", len(item_mapping))

        for name, model in models_to_train:
            print(f"\n{'=' * 40}")
            print(f"Training {name}...")
            t0 = time.time()
            # Special fit for models that need extra data
            if isinstance(model, SessionCooccurrenceRecommender):
                model.fit(train_baskets, user_histories=user_histories)
            elif isinstance(model, ContentBoostRecommender):
                model.fit(train_baskets, product_catalog=product_catalog)
            else:
                model.fit(train_baskets)
            train_time = time.time() - t0
            print(f"Training time: {train_time:.1f}s")

            # Evaluate on test
            print(f"Evaluating {name} on test...")
            test_metrics = evaluate_model(model, test_data, k_values, split_name="test")

            # Evaluate on OOT
            print(f"Evaluating {name} on OOT...")
            oot_metrics = evaluate_model(model, oot_data, k_values, split_name="oot")

            metrics = {**test_metrics, **oot_metrics, "train_time_s": train_time}
            all_results[name] = metrics

            # Log metrics to MLflow
            if mlflow_enabled:
                for metric_name, value in metrics.items():
                    safe_name = metric_name.replace("@", "_at_")
                    mlflow.log_metric(f"{name}_{safe_name}", value)

            print(f"\n{name} Results:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")

            # Save model
            model_path = models_dir / f"{name.lower()}_model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

        # Save artifacts
        with open(models_dir / "item_mapping.pkl", "wb") as f:
            pickle.dump(item_mapping, f)

        with open(models_dir / "evaluation_results.pkl", "wb") as f:
            pickle.dump(all_results, f)

        with open(models_dir / "train_baskets.pkl", "wb") as f:
            pickle.dump(train_baskets, f)

        if mlflow_enabled and cfg.logging.mlflow.log_models:
            mlflow.log_artifacts(str(models_dir), "models")

    finally:
        if run_ctx:
            run_ctx.__exit__(None, None, None)

    # Print summary
    _print_summary(all_results, k_values, models_dir)
    return all_results


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten nested dict for MLflow params"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)


def _print_summary(
    all_results: dict[str, dict[str, float]],
    k_values: list[int],
    models_dir: Path,
) -> None:
    """Print formatted summary table"""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    header = f"{'Model':<20}"
    for _split in ["test", "oot"]:
        header += f" {'Hit@5':>8} {'Hit@10':>8} {'MRR':>8} |"
    print(f"\n{header}")
    print(f"{'':20s} {'--- Test ---':^26s} | {'--- OOT ---':^26s} |")
    print("-" * 80)

    for name, metrics in all_results.items():
        row = f"{name:<20}"
        for split in ["test", "oot"]:
            h5 = metrics.get(f"{split}_hit@5", 0)
            h10 = metrics.get(f"{split}_hit@10", 0)
            mrr = metrics.get(f"{split}_mrr", 0)
            row += f" {h5:>8.4f} {h10:>8.4f} {mrr:>8.4f} |"
        print(row)

    print(f"\nModels saved to: {models_dir}")
    print("Training complete!")
