"""Recommendation models"""

import re
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from gensim.models import Word2Vec
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from omegaconf import DictConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


class TopPopularRecommender:
    """Baseline: recommends most popular items"""

    def __init__(self):
        self.item_counts: dict[int, int] = {}
        self.top_items: list[int] = []

    def fit(self, baskets: list[list[int]]) -> "TopPopularRecommender":
        self.item_counts = defaultdict(int)
        for basket in baskets:
            for item in basket:
                self.item_counts[item] += 1
        self.top_items = sorted(
            self.item_counts.keys(),
            key=lambda x: self.item_counts[x],
            reverse=True,
        )
        return self

    def recommend(self, basket: list[int], k: int = 10) -> list[int]:
        basket_set = set(basket)
        recs = [item for item in self.top_items if item not in basket_set]
        return recs[:k]


class CooccurrenceLiftRecommender:
    """Co-occurrence with Lift metric for ranking"""

    def __init__(self, min_support: int = 2):
        self.min_support = min_support
        self.item_counts: dict[int, int] = {}
        self.pair_counts: dict[tuple[int, int], int] = {}
        self.total_baskets = 0
        self.lift_matrix: dict[int, dict[int, float]] = {}

    def fit(self, baskets: list[list[int]]) -> "CooccurrenceLiftRecommender":
        self.item_counts = defaultdict(int)
        self.pair_counts = defaultdict(int)
        self.total_baskets = len(baskets)

        for basket in tqdm(baskets, desc="Building co-occurrence"):
            unique_items = list(set(basket))
            for item in unique_items:
                self.item_counts[item] += 1
            for item1, item2 in combinations(sorted(unique_items), 2):
                self.pair_counts[(item1, item2)] += 1

        self.lift_matrix = defaultdict(dict)
        for (item1, item2), count in tqdm(self.pair_counts.items(), desc="Computing lift"):
            if count >= self.min_support:
                p_item1 = self.item_counts[item1] / self.total_baskets
                p_item2 = self.item_counts[item2] / self.total_baskets
                p_both = count / self.total_baskets
                lift = p_both / (p_item1 * p_item2) if p_item1 * p_item2 > 0 else 0
                self.lift_matrix[item1][item2] = lift
                self.lift_matrix[item2][item1] = lift

        return self

    def recommend(self, basket: list[int], k: int = 10) -> list[int]:
        basket_set = set(basket)
        candidate_scores: dict[int, float] = defaultdict(float)

        for item in basket:
            if item in self.lift_matrix:
                for other_item, lift in self.lift_matrix[item].items():
                    if other_item not in basket_set:
                        candidate_scores[other_item] += lift

        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in sorted_candidates[:k]]


class Item2VecRecommender:
    """Item2Vec: Word2Vec on baskets"""

    def __init__(self, cfg: DictConfig | None = None):
        if cfg is not None:
            self.vector_size = cfg.model.item2vec.vector_size
            self.window = cfg.model.item2vec.window
            self.min_count = cfg.model.item2vec.min_count
            self.epochs = cfg.model.item2vec.epochs
            self.sg = cfg.model.item2vec.sg
            self.workers = cfg.model.item2vec.workers
        else:
            self.vector_size = 64
            self.window = 5
            self.min_count = 2
            self.epochs = 20
            self.sg = 1
            self.workers = 4
        self.model: Word2Vec | None = None
        self.all_items: list[str] = []

    def fit(self, baskets: list[list[int]]) -> "Item2VecRecommender":
        sentences = [[str(item) for item in basket] for basket in baskets]

        workers = self.workers
        if workers > 1:
            import warnings

            warnings.warn(
                f"Item2Vec workers={workers}: результаты будут недетерминированы. "
                "Установите workers=1 для полной воспроизводимости",
                stacklevel=2,
            )

        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            sg=self.sg,
            workers=workers,
            seed=42,
        )
        self.all_items = list(self.model.wv.key_to_index.keys())
        return self

    def recommend(self, basket: list[int], k: int = 10) -> list[int]:
        if self.model is None:
            return []

        basket_set = set(basket)
        basket_strs = [str(item) for item in basket if str(item) in self.model.wv]

        if not basket_strs:
            return []

        basket_vector = np.mean([self.model.wv[item] for item in basket_strs], axis=0)
        similar = self.model.wv.similar_by_vector(basket_vector, topn=k + len(basket))

        recs = []
        for item_str, _ in similar:
            item_id = int(item_str)
            if item_id not in basket_set:
                recs.append(item_id)
                if len(recs) >= k:
                    break

        return recs


class ImplicitALSRecommender:
    """ALS collaborative filtering via implicit library

    Each basket is treated as a separate "user" in the user-item matrix
    """

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.01,
        iterations: int = 15,
        alpha: float = 1.0,
    ):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.model: AlternatingLeastSquares | None = None
        self.item_ids: list[int] = []
        self.item_to_idx: dict[int, int] = {}

    def _build_matrix(self, baskets: list[list[int]]) -> sp.csr_matrix:
        """Build user-item sparse matrix from baskets"""
        all_items = sorted({item for basket in baskets for item in basket})
        self.item_ids = all_items
        self.item_to_idx = {item: idx for idx, item in enumerate(all_items)}

        rows, cols, data = [], [], []
        for user_idx, basket in enumerate(baskets):
            for item in basket:
                if item in self.item_to_idx:
                    rows.append(user_idx)
                    cols.append(self.item_to_idx[item])
                    data.append(1.0)

        return sp.csr_matrix(
            (data, (rows, cols)),
            shape=(len(baskets), len(all_items)),
        )

    def fit(self, baskets: list[list[int]]) -> "ImplicitALSRecommender":
        user_item = self._build_matrix(baskets)
        # implicit expects item-user matrix for ALS
        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=42,
        )
        # Confidence matrix = alpha * interaction
        self.model.fit(user_item * self.alpha)
        # Store user-item for recommend calls
        self._user_item = user_item
        return self

    def recommend(self, basket: list[int], k: int = 10) -> list[int]:
        if self.model is None:
            return []

        basket_set = set(basket)
        # Build a sparse vector for this basket as a new "user"
        cols = [self.item_to_idx[item] for item in basket if item in self.item_to_idx]
        if not cols:
            return []

        user_vector = sp.csr_matrix(
            ([1.0] * len(cols), ([0] * len(cols), cols)),
            shape=(1, len(self.item_ids)),
        )

        # Get recommendations
        item_indices, scores = self.model.recommend(
            userid=0,
            user_items=user_vector,
            N=k + len(basket),
            filter_already_liked_items=False,
        )

        recs = []
        for idx in item_indices:
            item_id = self.item_ids[idx]
            if item_id not in basket_set:
                recs.append(item_id)
                if len(recs) >= k:
                    break
        return recs


class ImplicitBPRRecommender:
    """BPR (Bayesian Personalized Ranking) via implicit library

    Each basket is treated as a separate "user" in the user-item matrix
    Optimizes pairwise ranking loss (BPR)
    """

    def __init__(
        self,
        factors: int = 64,
        learning_rate: float = 0.01,
        regularization: float = 0.01,
        iterations: int = 100,
    ):
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.iterations = iterations
        self.model: BayesianPersonalizedRanking | None = None
        self.item_ids: list[int] = []
        self.item_to_idx: dict[int, int] = {}

    def _build_matrix(self, baskets: list[list[int]]) -> sp.csr_matrix:
        """Build user-item sparse matrix from baskets"""
        all_items = sorted({item for basket in baskets for item in basket})
        self.item_ids = all_items
        self.item_to_idx = {item: idx for idx, item in enumerate(all_items)}

        rows, cols, data = [], [], []
        for user_idx, basket in enumerate(baskets):
            for item in basket:
                if item in self.item_to_idx:
                    rows.append(user_idx)
                    cols.append(self.item_to_idx[item])
                    data.append(1.0)

        return sp.csr_matrix(
            (data, (rows, cols)),
            shape=(len(baskets), len(all_items)),
        )

    def fit(self, baskets: list[list[int]]) -> "ImplicitBPRRecommender":
        user_item = self._build_matrix(baskets)
        self.model = BayesianPersonalizedRanking(
            factors=self.factors,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=42,
        )
        self.model.fit(user_item)
        self._user_item = user_item
        return self

    def recommend(self, basket: list[int], k: int = 10) -> list[int]:
        if self.model is None:
            return []

        basket_set = set(basket)
        cols = [self.item_to_idx[item] for item in basket if item in self.item_to_idx]
        if not cols:
            return []

        user_vector = sp.csr_matrix(
            ([1.0] * len(cols), ([0] * len(cols), cols)),
            shape=(1, len(self.item_ids)),
        )

        item_indices, scores = self.model.recommend(
            userid=0,
            user_items=user_vector,
            N=k + len(basket),
            filter_already_liked_items=False,
        )

        recs = []
        for idx in item_indices:
            item_id = self.item_ids[idx]
            if item_id not in basket_set:
                recs.append(item_id)
                if len(recs) >= k:
                    break
        return recs


class SessionCooccurrenceRecommender:
    """Session-based recommender that combines current basket with user purchase history

    Uses co-occurrence Lift from the base model, but enriches the query
    with items from the user's historical purchases (previous orders)
    """

    def __init__(self, history_weight: float = 0.3, min_support: int = 2):
        self.history_weight = history_weight
        self.min_support = min_support
        self.base_model = CooccurrenceLiftRecommender(min_support=min_support)
        self.user_histories: dict[int, list[int]] = {}

    def fit(
        self,
        baskets: list[list[int]],
        user_histories: dict[int, list[int]] | None = None,
    ) -> "SessionCooccurrenceRecommender":
        self.base_model.fit(baskets)
        self.user_histories = user_histories or {}
        return self

    def recommend(
        self,
        basket: list[int],
        k: int = 10,
        user_id: int | None = None,
    ) -> list[int]:
        basket_set = set(basket)
        candidate_scores: dict[int, float] = defaultdict(float)

        # Scores from current basket (weight = 1.0)
        for item in basket:
            if item in self.base_model.lift_matrix:
                for other_item, lift in self.base_model.lift_matrix[item].items():
                    if other_item not in basket_set:
                        candidate_scores[other_item] += lift

        # Scores from user history (weight = history_weight)
        if user_id is not None and user_id in self.user_histories:
            history_items = [
                it for it in self.user_histories[user_id] if it not in basket_set
            ]
            for item in history_items:
                if item in self.base_model.lift_matrix:
                    for other_item, lift in self.base_model.lift_matrix[item].items():
                        if other_item not in basket_set:
                            candidate_scores[other_item] += lift * self.history_weight

        sorted_candidates = sorted(
            candidate_scores.items(), key=lambda x: x[1], reverse=True
        )
        return [item for item, _ in sorted_candidates[:k]]


class ContentBoostRecommender:
    """Content-boosted recommender that combines co-occurrence with product metadata

    Uses category affinity and TF-IDF text similarity on product descriptions
    to re-rank and enrich co-occurrence recommendations
    """

    def __init__(
        self,
        cooc_weight: float = 0.6,
        category_weight: float = 0.25,
        text_weight: float = 0.15,
        min_support: int = 2,
    ):
        self.cooc_weight = cooc_weight
        self.category_weight = category_weight
        self.text_weight = text_weight
        self.min_support = min_support
        self.base_model = CooccurrenceLiftRecommender(min_support=min_support)
        self.item_categories: dict[int, str] = {}
        self.category_affinity: dict[str, dict[str, float]] = {}
        self._tfidf_matrix = None
        self._tfidf_oms_ids: list[int] = []
        self._tfidf_id_to_idx: dict[int, int] = {}

    def fit(
        self,
        baskets: list[list[int]],
        product_catalog: pd.DataFrame | None = None,
    ) -> "ContentBoostRecommender":
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
        for basket_items in tqdm(baskets, desc="Building category affinity"):
            cats = list({
                self.item_categories[it]
                for it in basket_items
                if it in self.item_categories
            })
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

        # Build TF-IDF vectors for items with descriptions
        items_with_desc = product_catalog[product_catalog["description"].str.len() > 0]
        if len(items_with_desc) > 0:
            tfidf = TfidfVectorizer(max_features=5000, stop_words=None)
            vectors = tfidf.fit_transform(items_with_desc["description"])
            oms_ids = items_with_desc["oms_id"].astype(int).tolist()
            self._tfidf_matrix = vectors
            self._tfidf_oms_ids = oms_ids
            self._tfidf_id_to_idx = {oid: idx for idx, oid in enumerate(oms_ids)}

        return self

    def _text_similarity_scores(
        self, basket: list[int], candidates: set[int]
    ) -> dict[int, float]:
        """Compute average text similarity between basket items and candidates"""
        if self._tfidf_matrix is None or not candidates:
            return {}

        basket_idxs = [
            self._tfidf_id_to_idx[it]
            for it in basket
            if it in self._tfidf_id_to_idx
        ]
        if not basket_idxs:
            return {}

        basket_vec = np.asarray(self._tfidf_matrix[basket_idxs].mean(axis=0))

        cand_list = [c for c in candidates if c in self._tfidf_id_to_idx]
        if not cand_list:
            return {}

        cand_idxs = [self._tfidf_id_to_idx[c] for c in cand_list]
        cand_vecs = self._tfidf_matrix[cand_idxs]

        sims = cosine_similarity(basket_vec, cand_vecs).flatten()
        return dict(zip(cand_list, sims))

    def _category_scores(
        self, basket: list[int], candidates: set[int]
    ) -> dict[int, float]:
        """Compute category affinity scores for candidates"""
        basket_cats = {
            self.item_categories[it]
            for it in basket
            if it in self.item_categories
        }
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
        norm_cat = {
            c: cat_scores.get(c, 0) / max_cat if max_cat > 0 else 0
            for c in candidates
        }

        # Step 4: Text similarity scores
        text_scores = self._text_similarity_scores(basket, candidates)
        max_text = max(text_scores.values()) if text_scores else 1.0
        norm_text = {
            c: text_scores.get(c, 0) / max_text if max_text > 0 else 0
            for c in candidates
        }

        # Step 5: Combine scores
        final_scores = {}
        for c in candidates:
            final_scores[c] = (
                self.cooc_weight * norm_cooc.get(c, 0)
                + self.category_weight * norm_cat.get(c, 0)
                + self.text_weight * norm_text.get(c, 0)
            )

        sorted_candidates = sorted(
            final_scores.items(), key=lambda x: x[1], reverse=True
        )
        return [item for item, _ in sorted_candidates[:k]]
