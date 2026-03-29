"""Inference module for getting recommendations"""

import pickle
from pathlib import Path


def load_model(model_name: str, models_dir: str = "artifacts/models"):
    """Load a trained model

    Args:
        model_name: Name of model (popularity, cooccurrence, item2vec, implicit_als, implicit_bpr)
        models_dir: Directory with saved models

    Returns:
        Loaded model
    """
    models_dir = Path(models_dir)
    model_map = {
        "popularity": "toppopular_model.pkl",
        "cooccurrence": "cooccurrencelift_model.pkl",
        "item2vec": "item2vec_model.pkl",
        "implicit_als": "implicitals_model.pkl",
        "implicit_bpr": "implicitbpr_model.pkl",
        "content_boost": "contentboost_model.pkl",
        "popularity_rerank": "popularityrerank_model.pkl",
        "item_graph": "itemgraphnode2vec_model.pkl",
        "ensemble": "ensemblerrf_model.pkl",
    }

    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_map.keys())}")

    model_path = models_dir / model_map[model_name]
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run training first.")

    with open(model_path, "rb") as f:
        return pickle.load(f)


def load_item_mapping(models_dir: str = "artifacts/models") -> dict[int, str]:
    """Load item ID to name mapping

    Args:
        models_dir: Directory with saved models

    Returns:
        Dict mapping item_id -> item_name
    """
    mapping_path = Path(models_dir) / "item_mapping.pkl"
    if not mapping_path.exists():
        return {}

    with open(mapping_path, "rb") as f:
        return pickle.load(f)


def run_inference(
    basket: list[int],
    model_name: str = "cooccurrence",
    top_k: int = 5,
    models_dir: str = "artifacts/models",
) -> list[tuple[int, str]]:
    """Get recommendations for a basket

    Args:
        basket: List of item IDs in the basket
        model_name: Model to use
        top_k: Number of recommendations
        models_dir: Directory with models

    Returns:
        List of (item_id, item_name) tuples
    """
    model = load_model(model_name, models_dir)
    item_mapping = load_item_mapping(models_dir)

    recommendations = model.recommend(basket, k=top_k)

    results = []
    print(f"\nRecommendations from {model_name} model:")
    print("-" * 50)
    for idx, item_id in enumerate(recommendations, 1):
        item_name = item_mapping.get(item_id, f"Item #{item_id}")
        results.append((item_id, item_name))
        print(f"{idx}. {item_name} (ID: {item_id})")

    return results
