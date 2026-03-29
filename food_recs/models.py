"""Recommendation models"""

import math
import random
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
    """Co-occurrence recommender with configurable association metric"""

    def __init__(self, min_support: int = 2, score_metric: str = "lift"):
        self.min_support = min_support
        self.score_metric = score_metric.lower()
        self.item_counts: dict[int, int] = {}
        self.pair_counts: dict[tuple[int, int], int] = {}
        self.total_baskets = 0
        self.lift_matrix: dict[int, dict[int, float]] = {}

    def _pair_score(self, p_item1: float, p_item2: float, p_both: float) -> float:
        if p_item1 <= 0 or p_item2 <= 0 or p_both <= 0:
            return 0.0
        if self.score_metric == "pmi":
            return float(math.log2(p_both / (p_item1 * p_item2)))
        if self.score_metric == "npmi":
            pmi = math.log2(p_both / (p_item1 * p_item2))
            denom = -math.log2(p_both)
            return float(pmi / denom) if denom > 0 else 0.0
        # Default: lift
        return float(p_both / (p_item1 * p_item2))

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
        for (item1, item2), count in tqdm(
            self.pair_counts.items(), desc=f"Computing {self.score_metric}"
        ):
            if count >= self.min_support:
                p_item1 = self.item_counts[item1] / self.total_baskets
                p_item2 = self.item_counts[item2] / self.total_baskets
                p_both = count / self.total_baskets
                assoc = self._pair_score(p_item1, p_item2, p_both)
                if assoc > 0:
                    self.lift_matrix[item1][item2] = assoc
                    self.lift_matrix[item2][item1] = assoc

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
            history_items = [it for it in self.user_histories[user_id] if it not in basket_set]
            for item in history_items:
                if item in self.base_model.lift_matrix:
                    for other_item, lift in self.base_model.lift_matrix[item].items():
                        if other_item not in basket_set:
                            candidate_scores[other_item] += lift * self.history_weight

        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
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
        max_features: int = 10000,
        ngram_range: tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
        use_russian_stopwords: bool = True,
        score_metric: str = "lift",
    ):
        self.cooc_weight = cooc_weight
        self.category_weight = category_weight
        self.text_weight = text_weight
        self.min_support = min_support
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.use_russian_stopwords = use_russian_stopwords
        self.base_model = CooccurrenceLiftRecommender(
            min_support=min_support,
            score_metric=score_metric,
        )
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

        # Build TF-IDF vectors for items with descriptions
        items_with_desc = product_catalog[product_catalog["description"].str.len() > 0]
        if len(items_with_desc) > 0:
            russian_stops = None
            if self.use_russian_stopwords:
                russian_stops = [
                    "и",
                    "в",
                    "во",
                    "не",
                    "что",
                    "он",
                    "на",
                    "я",
                    "с",
                    "со",
                    "как",
                    "а",
                    "то",
                    "все",
                    "она",
                    "так",
                    "его",
                    "но",
                    "да",
                    "ты",
                    "к",
                    "у",
                    "же",
                    "вы",
                    "за",
                    "бы",
                    "по",
                    "только",
                    "ее",
                    "мне",
                    "было",
                    "вот",
                    "от",
                    "меня",
                    "еще",
                    "нет",
                    "о",
                    "из",
                    "ему",
                    "теперь",
                    "когда",
                    "даже",
                    "ну",
                    "вдруг",
                    "ли",
                    "если",
                    "уже",
                    "или",
                    "ни",
                    "быть",
                    "был",
                    "него",
                    "до",
                    "вас",
                    "нибудь",
                ]
            tfidf = TfidfVectorizer(
                max_features=self.max_features,
                stop_words=russian_stops,
                ngram_range=self.ngram_range,
                sublinear_tf=self.sublinear_tf,
            )
            vectors = tfidf.fit_transform(items_with_desc["description"])
            oms_ids = items_with_desc["oms_id"].astype(int).tolist()
            self._tfidf_matrix = vectors
            self._tfidf_oms_ids = oms_ids
            self._tfidf_id_to_idx = {oid: idx for idx, oid in enumerate(oms_ids)}

        return self

    def _text_similarity_scores(self, basket: list[int], candidates: set[int]) -> dict[int, float]:
        """Compute average text similarity between basket items and candidates"""
        if self._tfidf_matrix is None or not candidates:
            return {}

        basket_idxs = [self._tfidf_id_to_idx[it] for it in basket if it in self._tfidf_id_to_idx]
        if not basket_idxs:
            return {}

        basket_vec = np.asarray(self._tfidf_matrix[basket_idxs].mean(axis=0))

        cand_list = [c for c in candidates if c in self._tfidf_id_to_idx]
        if not cand_list:
            return {}

        cand_idxs = [self._tfidf_id_to_idx[c] for c in cand_list]
        cand_vecs = self._tfidf_matrix[cand_idxs]

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

        # Step 4: Text similarity scores
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


class PopularityRerankRecommender:
    """Wraps a base model and re-ranks with popularity prior"""

    def __init__(self, base_model, pop_weight: float = 0.2):
        self.base_model = base_model
        self.pop_weight = pop_weight
        self.item_popularity: dict[int, int] = {}

    def fit(self, baskets: list[list[int]]) -> "PopularityRerankRecommender":
        if hasattr(self.base_model, "fit"):
            self.base_model.fit(baskets)
        item_counts = defaultdict(int)
        for basket in baskets:
            for item in basket:
                item_counts[item] += 1
        self.item_popularity = dict(item_counts)
        return self

    def recommend(self, basket: list[int], k: int = 10) -> list[int]:
        pool_k = max(k * 5, 50)
        recs = self.base_model.recommend(basket, k=pool_k)
        basket_set = set(basket)
        scored = []
        for rank, item in enumerate(recs):
            if item in basket_set:
                continue
            model_score = 1.0 / (rank + 1)
            pop_score = math.log1p(self.item_popularity.get(item, 0))
            scored.append((item, model_score + self.pop_weight * pop_score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in scored[:k]]


class EnsembleRecommender:
    """Weighted reciprocal rank fusion over multiple recommenders"""

    def __init__(self, models_with_weights: list[tuple[str, object, float]]):
        self.models = models_with_weights

    def fit(self, baskets: list[list[int]]) -> "EnsembleRecommender":
        # Components are expected to be pre-fitted in the training pipeline.
        _ = baskets
        return self

    def recommend(self, basket: list[int], k: int = 10) -> list[int]:
        basket_set = set(basket)
        all_scores: dict[int, float] = defaultdict(float)
        for _name, model, weight in self.models:
            recs = model.recommend(basket, k=max(k * 3, 30))
            for rank, item in enumerate(recs):
                if item not in basket_set:
                    all_scores[item] += weight / (rank + 1)
        return sorted(all_scores.keys(), key=lambda x: all_scores[x], reverse=True)[:k]


class ItemGraphNode2VecRecommender:
    """Graph-based item recommender using random walks + Word2Vec"""

    def __init__(
        self,
        min_support: int = 2,
        walk_length: int = 20,
        num_walks: int = 10,
        vector_size: int = 64,
        window: int = 5,
        epochs: int = 10,
        workers: int = 1,
    ):
        self.min_support = min_support
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.vector_size = vector_size
        self.window = window
        self.epochs = epochs
        self.workers = workers
        self.graph: dict[int, dict[int, int]] = defaultdict(dict)
        self.model: Word2Vec | None = None

    def fit(self, baskets: list[list[int]]) -> "ItemGraphNode2VecRecommender":
        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        for basket in baskets:
            unique_items = list(set(basket))
            for i1, i2 in combinations(sorted(unique_items), 2):
                pair_counts[(i1, i2)] += 1

        self.graph = defaultdict(dict)
        for (i1, i2), c in pair_counts.items():
            if c >= self.min_support:
                self.graph[i1][i2] = c
                self.graph[i2][i1] = c

        walks: list[list[str]] = []
        nodes = list(self.graph.keys())
        rng = random.Random(42)
        if not nodes:
            return self

        for _ in range(self.num_walks):
            rng.shuffle(nodes)
            for start in nodes:
                walk = [start]
                current = start
                for _step in range(self.walk_length - 1):
                    neighbors = self.graph.get(current, {})
                    if not neighbors:
                        break
                    nbr_ids = list(neighbors.keys())
                    weights = list(neighbors.values())
                    current = rng.choices(nbr_ids, weights=weights, k=1)[0]
                    walk.append(current)
                walks.append([str(i) for i in walk])

        self.model = Word2Vec(
            sentences=walks,
            vector_size=self.vector_size,
            window=self.window,
            min_count=1,
            sg=1,
            epochs=self.epochs,
            workers=self.workers,
            seed=42,
        )
        return self

    def recommend(self, basket: list[int], k: int = 10) -> list[int]:
        if self.model is None:
            return []
        basket_set = set(basket)
        basket_strs = [str(it) for it in basket if str(it) in self.model.wv]
        if not basket_strs:
            return []

        basket_vec = np.mean([self.model.wv[item] for item in basket_strs], axis=0)
        similar = self.model.wv.similar_by_vector(basket_vec, topn=k + len(basket))
        recs = []
        for item_str, _score in similar:
            item_id = int(item_str)
            if item_id not in basket_set:
                recs.append(item_id)
                if len(recs) >= k:
                    break
        return recs
