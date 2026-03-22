"""Recommendation models"""

from collections import defaultdict
from itertools import combinations

import numpy as np
import scipy.sparse as sp
from gensim.models import Word2Vec
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from omegaconf import DictConfig
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
