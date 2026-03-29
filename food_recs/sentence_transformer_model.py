"""ContentBoost recommender with Sentence Transformer embeddings instead of TF-IDF

Replaces sparse TF-IDF vectors with dense 384-dim multilingual embeddings
for better semantic similarity between product descriptions
"""

from __future__ import annotations

import logging
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from food_recs.models import CooccurrenceLiftRecommender

logger = logging.getLogger(__name__)


def _load_sentence_model(model_name: str, device: str | None = None):
    """Lazy-load SentenceTransformer with GPU/CPU fallback

    Args:
        model_name: HuggingFace model name
        device: Force device ('cuda', 'cpu', or None for auto)

    Returns:
        SentenceTransformer model instance
    """
    from sentence_transformers import SentenceTransformer

    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading SentenceTransformer %s on %s", model_name, device)
    return SentenceTransformer(model_name, device=device)


class SentenceTransformerBoostRecommender:
    """Content-boosted recommender using Sentence Transformer embeddings

    Drop-in replacement for ContentBoostRecommender that uses dense
    multilingual embeddings (e.g. intfloat/multilingual-e5-base, 384-dim)
    instead of sparse TF-IDF vectors for text similarity scoring

    Architecture:
    - Co-occurrence scores from CooccurrenceLiftRecommender (same as ContentBoost)
    - Category affinity from basket co-occurrence (same as ContentBoost)
    - Text similarity via Sentence Transformer cosine similarity (NEW)
    """

    def __init__(
        self,
        cooc_weight: float = 0.5,
        category_weight: float = 0.2,
        text_weight: float = 0.3,
        min_support: int = 2,
        score_metric: str = "lift",
        st_model_name: str = "intfloat/multilingual-e5-base",
        st_device: str | None = None,
        st_batch_size: int = 256,
        st_prefix: str = "query: ",
    ):
        self.cooc_weight = cooc_weight
        self.category_weight = category_weight
        self.text_weight = text_weight
        self.min_support = min_support
        self.score_metric = score_metric
        self.st_model_name = st_model_name
        self.st_device = st_device
        self.st_batch_size = st_batch_size
        self.st_prefix = st_prefix

        self.base_model = CooccurrenceLiftRecommender(
            min_support=min_support,
            score_metric=score_metric,
        )
        self.item_categories: dict[int, str] = {}
        self.category_affinity: dict[str, dict[str, float]] = {}

        # Sentence Transformer embeddings (dense)
        self._embeddings: np.ndarray | None = None
        self._emb_oms_ids: list[int] = []
        self._emb_id_to_idx: dict[int, int] = {}
        self._st_model = None

    def fit(
        self,
        baskets: list[list[int]],
        product_catalog: pd.DataFrame | None = None,
    ) -> SentenceTransformerBoostRecommender:
        self.base_model.fit(baskets)

        if product_catalog is None or product_catalog.empty:
            return self

        # Build item -> category mapping
        for _, row in product_catalog.iterrows():
            self.item_categories[int(row["oms_id"])] = row["category"]

        # Build category co-occurrence affinity from baskets
        cat_pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        cat_counts: dict[str, int] = defaultdict(int)
        total = 0
        for basket_items in tqdm(baskets, desc="Building category affinity (ST)"):
            cats = list(
                {self.item_categories[it] for it in basket_items if it in self.item_categories}
            )
            for cat in cats:
                cat_counts[cat] += 1
            total += 1
            for c1, c2 in combinations(sorted(cats), 2):
                cat_pair_counts[(c1, c2)] += 1

        self.category_affinity = defaultdict(dict)
        for (c1, c2), count in cat_pair_counts.items():
            if total > 0 and cat_counts[c1] > 0 and cat_counts[c2] > 0:
                p1 = cat_counts[c1] / total
                p2 = cat_counts[c2] / total
                p_both = count / total
                lift = p_both / (p1 * p2) if p1 * p2 > 0 else 0
                self.category_affinity[c1][c2] = lift
                self.category_affinity[c2][c1] = lift
        # Same-category affinity = 1.5 (boost)
        for cat in cat_counts:
            self.category_affinity[cat][cat] = 1.5

        # Build Sentence Transformer embeddings for items with descriptions
        items_with_desc = product_catalog[product_catalog["description"].str.len() > 0].copy()
        if len(items_with_desc) > 0:
            self._st_model = _load_sentence_model(self.st_model_name, self.st_device)

            descriptions = items_with_desc["description"].tolist()
            # Add prefix for e5 models (query: / passage:)
            if self.st_prefix:
                descriptions = [self.st_prefix + d for d in descriptions]

            print(f"Encoding {len(descriptions)} product descriptions with {self.st_model_name}...")
            embeddings = self._st_model.encode(
                descriptions,
                batch_size=self.st_batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
            )

            oms_ids = items_with_desc["oms_id"].astype(int).tolist()
            self._embeddings = np.array(embeddings, dtype=np.float32)
            self._emb_oms_ids = oms_ids
            self._emb_id_to_idx = {oid: idx for idx, oid in enumerate(oms_ids)}
            print(f"Embeddings shape: {self._embeddings.shape}")

        return self

    def _text_similarity_scores(self, basket: list[int], candidates: set[int]) -> dict[int, float]:
        """Compute average cosine similarity between basket items and candidates
        using Sentence Transformer embeddings
        """
        if self._embeddings is None or not candidates:
            return {}

        basket_idxs = [self._emb_id_to_idx[it] for it in basket if it in self._emb_id_to_idx]
        if not basket_idxs:
            return {}

        basket_vec = self._embeddings[basket_idxs].mean(axis=0, keepdims=True)

        cand_list = [c for c in candidates if c in self._emb_id_to_idx]
        if not cand_list:
            return {}

        cand_idxs = [self._emb_id_to_idx[c] for c in cand_list]
        cand_vecs = self._embeddings[cand_idxs]

        sims = cosine_similarity(basket_vec, cand_vecs).flatten()
        return dict(zip(cand_list, sims, strict=False))

    def _category_scores(self, basket: list[int], candidates: set[int]) -> dict[int, float]:
        """Compute category affinity scores for candidates"""
        basket_cats = {self.item_categories[it] for it in basket if it in self.item_categories}
        if not basket_cats:
            return {}

        scores = {}
        for cand in candidates:
            if cand not in self.item_categories:
                continue
            cand_cat = self.item_categories[cand]
            max_affinity = 0.0
            for bcat in basket_cats:
                aff = self.category_affinity.get(bcat, {}).get(cand_cat, 0.0)
                max_affinity = max(max_affinity, aff)
            scores[cand] = max_affinity
        return scores

    def recommend(self, basket: list[int], k: int = 10) -> list[int]:
        basket_set = set(basket)

        # Step 1: Get co-occurrence candidates (wider pool)
        cooc_scores: dict[int, float] = defaultdict(float)
        for item in basket:
            if item in self.base_model.lift_matrix:
                for other_item, lift in self.base_model.lift_matrix[item].items():
                    if other_item not in basket_set:
                        cooc_scores[other_item] += lift

        # Get top candidates from co-occurrence (wider than k)
        top_cooc = sorted(cooc_scores.items(), key=lambda x: x[1], reverse=True)
        candidates = {item for item, _ in top_cooc[: k * 5]}

        if not candidates:
            return [item for item, _ in top_cooc[:k]]

        # Step 2: Normalize co-occurrence scores
        max_cooc = max(cooc_scores[c] for c in candidates) if candidates else 1.0
        norm_cooc = {c: cooc_scores.get(c, 0) / max_cooc for c in candidates}

        # Step 3: Category affinity scores
        cat_scores = self._category_scores(basket, candidates)
        max_cat = max(cat_scores.values()) if cat_scores else 1.0
        norm_cat = {c: cat_scores.get(c, 0) / max_cat if max_cat > 0 else 0 for c in candidates}

        # Step 4: Text similarity scores (Sentence Transformer)
        text_scores = self._text_similarity_scores(basket, candidates)
        max_text = max(text_scores.values()) if text_scores else 1.0
        norm_text = {c: text_scores.get(c, 0) / max_text if max_text > 0 else 0 for c in candidates}

        # Step 5: Combine scores
        final_scores = {}
        for c in candidates:
            final_scores[c] = (
                self.cooc_weight * norm_cooc.get(c, 0)
                + self.category_weight * norm_cat.get(c, 0)
                + self.text_weight * norm_text.get(c, 0)
            )

        sorted_candidates = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in sorted_candidates[:k]]
