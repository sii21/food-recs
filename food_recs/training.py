"""Training pipeline with MLflow logging"""

import pickle
import random
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
    EnsembleRecommender,
    ImplicitALSRecommender,
    ImplicitBPRRecommender,
    Item2VecRecommender,
    ItemGraphNode2VecRecommender,
    PopularityRerankRecommender,
    SessionCooccurrenceRecommender,
    TopPopularRecommender,
)
from food_recs.sentence_transformer_model import SentenceTransformerBoostRecommender


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
    n_items_total: int | None = None,
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
    unique_recommended: set[int] = set()
    total_reco_time = 0.0

    for entry in tqdm(test_data, desc=f"Evaluating ({split_name})"):
        input_basket, held_out_item = entry[0], entry[1]
        profile_id = entry[2] if len(entry) > 2 else None

        t0 = time.perf_counter()
        if isinstance(model, SessionCooccurrenceRecommender) and profile_id is not None:
            recs = model.recommend(input_basket, k=max(k_values), user_id=profile_id)
        else:
            recs = model.recommend(input_basket, k=max(k_values))
        total_reco_time += time.perf_counter() - t0
        unique_recommended.update(recs[: max(k_values)])

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
    results[f"{split_name}_avg_latency_ms"] = (total_reco_time / n_test) * 1000.0
    if n_items_total and n_items_total > 0:
        results[f"{split_name}_coverage"] = len(unique_recommended) / n_items_total
    else:
        results[f"{split_name}_coverage"] = 0.0

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
                CooccurrenceLiftRecommender(
                    cfg.model.cooccurrence.min_support,
                    score_metric=cfg.model.cooccurrence.get("score_metric", "lift"),
                ),
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
                    max_features=cb_cfg.get("max_features", 10000),
                    ngram_range=tuple(cb_cfg.get("ngram_range", [1, 2])),
                    sublinear_tf=cb_cfg.get("sublinear_tf", True),
                    use_russian_stopwords=cb_cfg.get("use_russian_stopwords", True),
                    score_metric=cb_cfg.get("score_metric", "lift"),
                ),
            )
        )
    if cfg.model.get("item_graph", {}).get("enabled", False):
        g_cfg = cfg.model.item_graph
        models.append(
            (
                "ItemGraphNode2Vec",
                ItemGraphNode2VecRecommender(
                    min_support=g_cfg.get("min_support", 2),
                    walk_length=g_cfg.get("walk_length", 20),
                    num_walks=g_cfg.get("num_walks", 10),
                    vector_size=g_cfg.get("vector_size", 64),
                    window=g_cfg.get("window", 5),
                    epochs=g_cfg.get("epochs", 10),
                    workers=g_cfg.get("workers", 1),
                ),
            )
        )
    if cfg.model.get("sentence_transformer_boost", {}).get("enabled", False):
        st_cfg = cfg.model.sentence_transformer_boost
        models.append(
            (
                "STBoost",
                SentenceTransformerBoostRecommender(
                    cooc_weight=st_cfg.get("cooc_weight", 0.5),
                    category_weight=st_cfg.get("category_weight", 0.2),
                    text_weight=st_cfg.get("text_weight", 0.3),
                    min_support=st_cfg.get("min_support", 2),
                    score_metric=st_cfg.get("score_metric", "lift"),
                    st_model_name=st_cfg.get("st_model_name", "intfloat/multilingual-e5-base"),
                    st_device=st_cfg.get("st_device", None),
                    st_batch_size=st_cfg.get("st_batch_size", 256),
                    st_prefix=st_cfg.get("st_prefix", "query: "),
                ),
            )
        )
    return models


def _build_profile_level_baskets(user_histories: dict[int, list[int]]) -> list[list[int]]:
    profile_baskets = []
    for _, items in user_histories.items():
        uniq_items = list(dict.fromkeys(items))
        if len(uniq_items) >= 2:
            profile_baskets.append(uniq_items)
    return profile_baskets


def _select_ensemble_models(
    cfg: DictConfig,
    trained_models: dict[str, object],
    all_results: dict[str, dict[str, float]],
) -> list[tuple[str, object, float]]:
    ens_cfg = cfg.model.get("ensemble", {})
    min_hit5 = float(ens_cfg.get("min_hit5", 0.0))
    base_weights = ens_cfg.get(
        "weights",
        {"TopPopular": 1.0, "CooccurrenceLift": 1.0, "ContentBoost": 1.0},
    )
    selected: list[tuple[str, object, float]] = []
    for name, weight in base_weights.items():
        if name not in trained_models or name not in all_results:
            continue
        metrics = all_results[name]
        test_hit5 = metrics.get("test_hit@5", 0.0)
        test_mrr = metrics.get("test_mrr", 0.0)
        if test_hit5 <= 0.0 or test_mrr <= 0.0:
            continue
        if test_hit5 < min_hit5:
            continue
        selected.append((name, trained_models[name], float(weight)))
    return selected


def _tune_val_subset(
    test_data: list[tuple],
    max_samples: int,
    seed: int,
) -> list[tuple]:
    """Subsample L1O rows for faster hyperparameter search (same draw for all grid points)."""
    if max_samples <= 0 or len(test_data) <= max_samples:
        return test_data
    rng = random.Random(seed)
    return rng.sample(test_data, max_samples)


def _tune_st_boost_weights(
    cfg: DictConfig,
    model: SentenceTransformerBoostRecommender,
    train_baskets: list[list[int]],
    test_data: list[tuple],
    product_catalog,
) -> SentenceTransformerBoostRecommender:
    st_cfg = cfg.model.sentence_transformer_boost
    if not st_cfg.get("tune_weights", False):
        return model

    max_tune = int(st_cfg.get("tune_val_max_samples", 0) or 0)
    seed = int(cfg.get("seed", 42))
    tune_data = _tune_val_subset(test_data, max_tune, seed)
    if len(tune_data) < len(test_data):
        print(
            f"Tuning STBoost on val subset: {len(tune_data):,} / {len(test_data):,} L1O examples "
            f"(tune_val_max_samples={max_tune})"
        )

    cooc_grid = st_cfg.get("cooc_grid", [0.3, 0.4, 0.5, 0.6])
    cat_grid = st_cfg.get("category_grid", [0.1, 0.2, 0.3])
    best = (model.cooc_weight, model.category_weight, model.text_weight)
    best_hit5 = -1.0

    # OPTIMIZATION: Pre-fit base model and cache embeddings once
    print("Pre-computing embeddings for grid search (1x instead of Nx)...")
    base_candidate = SentenceTransformerBoostRecommender(
        cooc_weight=0.5,
        category_weight=0.25,
        text_weight=0.25,
        min_support=model.min_support,
        score_metric=model.score_metric,
        st_model_name=model.st_model_name,
        st_device=model.st_device,
        st_batch_size=model.st_batch_size,
        st_prefix=model.st_prefix,
    )
    base_candidate.fit(train_baskets, product_catalog=product_catalog)
    
    # Cache the expensive parts
    cached_base_model = base_candidate.base_model
    cached_item_categories = base_candidate.item_categories
    cached_category_affinity = base_candidate.category_affinity
    cached_embeddings = base_candidate._embeddings
    cached_emb_oms_ids = base_candidate._emb_oms_ids
    cached_emb_id_to_idx = base_candidate._emb_id_to_idx
    cached_st_model = base_candidate._st_model

    total_combos = len([1 for c in cooc_grid for cat in cat_grid if 1.0 - c - cat >= 0])
    print(f"Tuning STBoost weights ({total_combos} combinations)...")
    print(f"{'Combo':<8} {'Cooc':>6} {'Cat':>6} {'Text':>6} {'Hit@5':>8}")
    print("-" * 40)
    
    combo_idx = 0
    for cooc_w in cooc_grid:
        for cat_w in cat_grid:
            text_w = 1.0 - float(cooc_w) - float(cat_w)
            if text_w < 0.0:
                continue
            combo_idx += 1
            
            # Create candidate with cached components (no re-encoding!)
            candidate = SentenceTransformerBoostRecommender(
                cooc_weight=float(cooc_w),
                category_weight=float(cat_w),
                text_weight=float(text_w),
                min_support=model.min_support,
                score_metric=model.score_metric,
                st_model_name=model.st_model_name,
                st_device=model.st_device,
                st_batch_size=model.st_batch_size,
                st_prefix=model.st_prefix,
            )
            # Inject cached components
            candidate.base_model = cached_base_model
            candidate.item_categories = cached_item_categories
            candidate.category_affinity = cached_category_affinity
            candidate._embeddings = cached_embeddings
            candidate._emb_oms_ids = cached_emb_oms_ids
            candidate._emb_id_to_idx = cached_emb_id_to_idx
            candidate._st_model = cached_st_model
            
            metrics = evaluate_model(
                candidate,
                tune_data,
                [5],
                split_name="val",
            )
            hit5 = metrics.get("val_hit@5", 0.0)
            print(f"{combo_idx}/{total_combos:<4} {cooc_w:>6.2f} {cat_w:>6.2f} {text_w:>6.2f} {hit5:>8.4f}")
            
            if hit5 > best_hit5:
                best_hit5 = hit5
                best = (float(cooc_w), float(cat_w), float(text_w))

    print("-" * 40)
    print(
        f"Best weights: cooc={best[0]:.2f}, category={best[1]:.2f}, text={best[2]:.2f}, val_hit@5={best_hit5:.4f}"
    )
    
    # Return final model with best weights and cached components
    final_model = SentenceTransformerBoostRecommender(
        cooc_weight=best[0],
        category_weight=best[1],
        text_weight=best[2],
        min_support=model.min_support,
        score_metric=model.score_metric,
        st_model_name=model.st_model_name,
        st_device=model.st_device,
        st_batch_size=model.st_batch_size,
        st_prefix=model.st_prefix,
    )
    final_model.base_model = cached_base_model
    final_model.item_categories = cached_item_categories
    final_model.category_affinity = cached_category_affinity
    final_model._embeddings = cached_embeddings
    final_model._emb_oms_ids = cached_emb_oms_ids
    final_model._emb_id_to_idx = cached_emb_id_to_idx
    final_model._st_model = cached_st_model
    
    return final_model


def _tune_content_boost_weights(
    cfg: DictConfig,
    model: ContentBoostRecommender,
    train_baskets: list[list[int]],
    test_data: list[tuple],
    product_catalog,
) -> ContentBoostRecommender:
    cb_cfg = cfg.model.content_boost
    if not cb_cfg.get("tune_weights", False):
        return model

    max_tune = int(cb_cfg.get("tune_val_max_samples", 0) or 0)
    seed = int(cfg.get("seed", 42))
    tune_data = _tune_val_subset(test_data, max_tune, seed)
    if len(tune_data) < len(test_data):
        print(
            f"Tuning ContentBoost on val subset: {len(tune_data):,} / {len(test_data):,} L1O examples "
            f"(tune_val_max_samples={max_tune})"
        )

    cooc_grid = cb_cfg.get("cooc_grid", [0.4, 0.5, 0.6, 0.7])
    cat_grid = cb_cfg.get("category_grid", [0.1, 0.2, 0.3])
    best = (model.cooc_weight, model.category_weight, model.text_weight)
    best_hit5 = -1.0

    print("Tuning ContentBoost weights...")
    for cooc_w in cooc_grid:
        for cat_w in cat_grid:
            text_w = 1.0 - float(cooc_w) - float(cat_w)
            if text_w < 0.0:
                continue
            candidate = ContentBoostRecommender(
                cooc_weight=float(cooc_w),
                category_weight=float(cat_w),
                text_weight=float(text_w),
                min_support=model.min_support,
                max_features=model.max_features,
                ngram_range=model.ngram_range,
                sublinear_tf=model.sublinear_tf,
                use_russian_stopwords=model.use_russian_stopwords,
                score_metric=model.base_model.score_metric,
            )
            candidate.fit(train_baskets, product_catalog=product_catalog)
            metrics = evaluate_model(
                candidate,
                tune_data,
                [5],
                split_name="val",
            )
            hit5 = metrics.get("val_hit@5", 0.0)
            if hit5 > best_hit5:
                best_hit5 = hit5
                best = (float(cooc_w), float(cat_w), float(text_w))

    print(
        "Best ContentBoost weights:",
        f"cooc={best[0]:.2f}, category={best[1]:.2f}, text={best[2]:.2f}, val_hit@5={best_hit5:.4f}",
    )
    return ContentBoostRecommender(
        cooc_weight=best[0],
        category_weight=best[1],
        text_weight=best[2],
        min_support=model.min_support,
        max_features=model.max_features,
        ngram_range=model.ngram_range,
        sublinear_tf=model.sublinear_tf,
        use_russian_stopwords=model.use_russian_stopwords,
        score_metric=model.base_model.score_metric,
    )


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
        train_baskets,
        test_baskets,
        oot_baskets,
        item_mapping,
        user_histories,
        product_catalog,
        test_profiles,
        oot_profiles,
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
    trained_models: dict[str, object] = {}

    profile_train_baskets = _build_profile_level_baskets(user_histories)
    use_profile_agg = cfg.model.get("profile_aggregation", {}).get("enabled", False)
    profile_models = set(cfg.model.get("profile_aggregation", {}).get("models", []))
    if use_profile_agg:
        print(f"Profile aggregation enabled: {len(profile_train_baskets):,} profile-level baskets")

    # Build models list
    models_to_train = _build_models_list(cfg)
    train_only_raw = cfg.model.get("train_only")
    train_only = str(train_only_raw).strip() if train_only_raw not in (None, False) else ""
    if train_only:
        models_to_train = [p for p in models_to_train if p[0] == train_only]
        if not models_to_train:
            known = ", ".join(n for n, _ in _build_models_list(cfg))
            raise ValueError(
                f"model.train_only={train_only!r} matches no enabled model. "
                f"Enable it in config or pick one of: {known}"
            )
        print(f"\nSingle-model mode (train_only): {train_only}")

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
            model_train_baskets = train_baskets
            if use_profile_agg and name in profile_models and profile_train_baskets:
                model_train_baskets = profile_train_baskets
                print(f"Using profile-level baskets for {name}: {len(model_train_baskets):,}")

            # Special fit for models that need extra data
            if isinstance(model, SessionCooccurrenceRecommender):
                model.fit(model_train_baskets, user_histories=user_histories)
            elif isinstance(model, ContentBoostRecommender):
                tuned = _tune_content_boost_weights(
                    cfg,
                    model,
                    model_train_baskets,
                    test_data,
                    product_catalog,
                )
                tuned.fit(model_train_baskets, product_catalog=product_catalog)
                model = tuned
            elif isinstance(model, SentenceTransformerBoostRecommender):
                tuned = _tune_st_boost_weights(
                    cfg,
                    model,
                    model_train_baskets,
                    test_data,
                    product_catalog,
                )
                tuned.fit(model_train_baskets, product_catalog=product_catalog)
                model = tuned
            else:
                model.fit(model_train_baskets)
            train_time = time.time() - t0
            print(f"Training time: {train_time:.1f}s")

            # Evaluate on test
            print(f"Evaluating {name} on test...")
            test_metrics = evaluate_model(
                model,
                test_data,
                k_values,
                split_name="test",
                n_items_total=len(item_mapping),
            )

            # Evaluate on OOT
            print(f"Evaluating {name} on OOT...")
            oot_metrics = evaluate_model(
                model,
                oot_data,
                k_values,
                split_name="oot",
                n_items_total=len(item_mapping),
            )

            metrics = {**test_metrics, **oot_metrics, "train_time_s": train_time}
            all_results[name] = metrics
            trained_models[name] = model

            # Log metrics to MLflow
            if mlflow_enabled:
                for metric_name, value in metrics.items():
                    safe_name = metric_name.replace("@", "_at_")
                    mlflow.log_metric(f"{name}_{safe_name}", value)

            print(f"\n{name} Results:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")

            disable_zero = cfg.model.get("disable_zero_metric_models", True)
            is_zero = metrics.get("test_hit@5", 0.0) <= 0.0 or metrics.get("test_mrr", 0.0) <= 0.0
            if disable_zero and is_zero:
                print(f"Skipping save for {name}: zero metrics on test split")
            else:
                model_path = models_dir / f"{name.lower()}_model.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)

        # Optional popularity re-rank model built on top of ContentBoost
        if (
            not train_only
            and cfg.model.get("popularity_rerank", {}).get("enabled", False)
            and "ContentBoost" in trained_models
        ):
            print(f"\n{'=' * 40}")
            print("Training PopularityRerank...")
            pr_cfg = cfg.model.popularity_rerank
            pr_model = PopularityRerankRecommender(
                base_model=trained_models["ContentBoost"],
                pop_weight=pr_cfg.get("pop_weight", 0.2),
            )
            t0 = time.time()
            pr_model.fit(train_baskets)
            train_time = time.time() - t0
            test_metrics = evaluate_model(
                pr_model, test_data, k_values, split_name="test", n_items_total=len(item_mapping)
            )
            oot_metrics = evaluate_model(
                pr_model, oot_data, k_values, split_name="oot", n_items_total=len(item_mapping)
            )
            metrics = {**test_metrics, **oot_metrics, "train_time_s": train_time}
            all_results["PopularityRerank"] = metrics
            trained_models["PopularityRerank"] = pr_model
            with open(models_dir / "popularityrerank_model.pkl", "wb") as f:
                pickle.dump(pr_model, f)

        # Optional ensemble model (RRF) with zero-metric filtering
        if not train_only and cfg.model.get("ensemble", {}).get("enabled", False):
            selected_models = _select_ensemble_models(cfg, trained_models, all_results)
            if len(selected_models) >= 2:
                print(f"\n{'=' * 40}")
                print("Training EnsembleRRF...")
                ens_model = EnsembleRecommender(selected_models)
                t0 = time.time()
                ens_model.fit(train_baskets)
                train_time = time.time() - t0
                test_metrics = evaluate_model(
                    ens_model,
                    test_data,
                    k_values,
                    split_name="test",
                    n_items_total=len(item_mapping),
                )
                oot_metrics = evaluate_model(
                    ens_model, oot_data, k_values, split_name="oot", n_items_total=len(item_mapping)
                )
                metrics = {**test_metrics, **oot_metrics, "train_time_s": train_time}
                all_results["EnsembleRRF"] = metrics
                trained_models["EnsembleRRF"] = ens_model
                with open(models_dir / "ensemblerrf_model.pkl", "wb") as f:
                    pickle.dump(ens_model, f)
            else:
                print("Skipping EnsembleRRF: less than 2 eligible component models")

        # Optional LightGBM learned ensemble
        if not train_only and cfg.model.get("lgbm_ensemble", {}).get("enabled", False):
            lgbm_cfg = cfg.model.lgbm_ensemble
            # Select base models with non-zero metrics
            lgbm_base = {
                name: m
                for name, m in trained_models.items()
                if all_results.get(name, {}).get("test_hit@5", 0) > 0
                and name not in ("EnsembleRRF", "PopularityRerank", "LGBMEnsemble")
            }
            if len(lgbm_base) >= 2:
                print(f"\n{'=' * 40}")
                print(f"Training LGBMEnsemble with {len(lgbm_base)} base models...")

                from food_recs.features.item_features import ItemFeatureExtractor
                from food_recs.features.user_features import UserFeatureExtractor
                from food_recs.lgbm_ensemble import LGBMEnsembleRecommender

                user_fe = UserFeatureExtractor()
                item_categories = {}
                if not product_catalog.empty:
                    for _, row in product_catalog.iterrows():
                        item_categories[int(row["oms_id"])] = row["category"]
                user_fe.fit(train_baskets, item_categories=item_categories)

                item_fe = ItemFeatureExtractor()
                item_fe.fit(train_baskets, product_catalog=product_catalog)

                lgbm_model = LGBMEnsembleRecommender(
                    base_models=lgbm_base,
                    user_feature_extractor=user_fe,
                    item_feature_extractor=item_fe,
                    n_candidates=lgbm_cfg.get("n_candidates", 50),
                    n_estimators=lgbm_cfg.get("n_estimators", 200),
                    learning_rate=lgbm_cfg.get("learning_rate", 0.05),
                )

                # Split test_data into lgbm_train / lgbm_val to avoid data leakage
                lgbm_val_ratio = lgbm_cfg.get("val_ratio", 0.3)
                rng_lgbm = random.Random(seed)
                lgbm_indices = list(range(len(test_data)))
                rng_lgbm.shuffle(lgbm_indices)
                n_val = max(1, int(len(lgbm_indices) * lgbm_val_ratio))
                lgbm_val_idx = set(lgbm_indices[:n_val])
                lgbm_train_data = [test_data[i] for i in range(len(test_data)) if i not in lgbm_val_idx]
                lgbm_val_data = [test_data[i] for i in lgbm_val_idx]
                print(
                    f"LGBMEnsemble split: train={len(lgbm_train_data):,}, "
                    f"val={len(lgbm_val_data):,} (val_ratio={lgbm_val_ratio})"
                )

                t0 = time.time()
                lgbm_model.fit(
                    lgbm_train_data,
                    user_histories=user_histories,
                    max_train_samples=lgbm_cfg.get("max_train_samples", 50000),
                    seed=seed,
                )
                train_time = time.time() - t0

                # Evaluate on held-out validation portion (no leakage)
                test_metrics = evaluate_model(
                    lgbm_model, lgbm_val_data, k_values,
                    split_name="test", n_items_total=len(item_mapping),
                )
                oot_metrics = evaluate_model(
                    lgbm_model, oot_data, k_values,
                    split_name="oot", n_items_total=len(item_mapping),
                )
                metrics = {**test_metrics, **oot_metrics, "train_time_s": train_time}
                all_results["LGBMEnsemble"] = metrics
                trained_models["LGBMEnsemble"] = lgbm_model
                with open(models_dir / "lgbmensemble_model.pkl", "wb") as f:
                    pickle.dump(lgbm_model, f)

                print(f"\nLGBMEnsemble Results:")
                for metric_name, value in metrics.items():
                    print(f"  {metric_name}: {value:.4f}")
            else:
                print("Skipping LGBMEnsemble: less than 2 eligible base models")

        # Debiased evaluation (runs on all trained models)
        if cfg.model.get("debiased_evaluation", {}).get("enabled", False):
            from food_recs.evaluation.debiased_metrics import DebiasedEvaluator

            db_cfg = cfg.model.debiased_evaluation
            n_buckets = db_cfg.get("n_buckets", 5)
            debiased_eval = DebiasedEvaluator(train_baskets, n_buckets=n_buckets)

            print(f"\n{'=' * 60}")
            print("DEBIASED EVALUATION")
            print(f"{'=' * 60}")
            for name, model in trained_models.items():
                db_metrics = debiased_eval.evaluate(
                    model, test_data, k_values,
                    split_name="test", n_items_total=len(item_mapping),
                )
                # Merge debiased metrics into all_results
                for mk, mv in db_metrics.items():
                    if "bucket" in mk or "debiased" in mk:
                        all_results[name][mk] = mv
                debiased_eval.print_bucket_summary(db_metrics, split_name="test")

        # Save artifacts
        with open(models_dir / "item_mapping.pkl", "wb") as f:
            pickle.dump(item_mapping, f)

        with open(models_dir / "evaluation_results.pkl", "wb") as f:
            pickle.dump(all_results, f)

        with open(models_dir / "train_baskets.pkl", "wb") as f:
            pickle.dump(train_baskets, f)
        with open(models_dir / "profile_train_baskets.pkl", "wb") as f:
            pickle.dump(profile_train_baskets, f)

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

    # Debiased summary (if available)
    has_debiased = any(
        "test_debiased_hit@5" in m for m in all_results.values()
    )
    if has_debiased:
        print(f"\n{'Model':<20} {'Debiased Hit@5':>14} {'Debiased MRR':>12}")
        print("-" * 50)
        for name, metrics in all_results.items():
            dh5 = metrics.get("test_debiased_hit@5", 0)
            dmrr = metrics.get("test_debiased_mrr", 0)
            if dh5 > 0 or dmrr > 0:
                print(f"{name:<20} {dh5:>14.4f} {dmrr:>12.4f}")

    print(f"\nModels saved to: {models_dir}")
    print("Training complete!")
