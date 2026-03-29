"""User-side feature extraction for ensemble learning

Extracts features from user purchase history for LightGBM ranker
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


class UserFeatureExtractor:
    """Extract user-level features from purchase history and current basket

    Features:
    - basket_size: number of items in current basket
    - n_unique_categories: number of distinct categories in basket
    - avg_item_popularity: average popularity of basket items
    - max_item_popularity: max popularity of basket items
    - history_length: total items in user's purchase history
    - history_unique_items: unique items in history
    - repeat_ratio: fraction of basket items that appear in history
    - preferred_category_match: whether basket top category matches user's preferred category
    """

    def __init__(self):
        self.item_popularity: dict[int, int] = {}
        self.item_categories: dict[int, str] = {}

    def fit(
        self,
        train_baskets: list[list[int]],
        item_categories: dict[int, str] | None = None,
    ) -> "UserFeatureExtractor":
        """Compute item popularity from training baskets

        Args:
            train_baskets: Training baskets
            item_categories: Optional item->category mapping
        """
        counts: dict[int, int] = defaultdict(int)
        for basket in train_baskets:
            for item in basket:
                counts[item] += 1
        self.item_popularity = dict(counts)
        self.item_categories = item_categories or {}
        return self

    def extract(
        self,
        basket: list[int],
        user_history: list[int] | None = None,
    ) -> dict[str, float]:
        """Extract features for a basket + optional user history

        Args:
            basket: Current basket items
            user_history: Historical purchases for this user

        Returns:
            Dict of feature name -> value
        """
        features: dict[str, float] = {}

        # Basket-level features
        features["basket_size"] = float(len(basket))

        basket_pops = [self.item_popularity.get(it, 0) for it in basket]
        features["avg_item_popularity"] = float(np.mean(basket_pops)) if basket_pops else 0.0
        features["max_item_popularity"] = float(np.max(basket_pops)) if basket_pops else 0.0
        features["min_item_popularity"] = float(np.min(basket_pops)) if basket_pops else 0.0

        basket_cats = [self.item_categories.get(it, "unknown") for it in basket]
        features["n_unique_categories"] = float(len(set(basket_cats)))

        # Category distribution
        cat_counter = Counter(basket_cats)
        if cat_counter:
            most_common_cat, most_common_count = cat_counter.most_common(1)[0]
            features["dominant_category_ratio"] = float(most_common_count) / len(basket_cats)
        else:
            features["dominant_category_ratio"] = 0.0

        # History features
        if user_history:
            history_set = set(user_history)
            features["history_length"] = float(len(user_history))
            features["history_unique_items"] = float(len(history_set))

            repeat_count = sum(1 for it in basket if it in history_set)
            features["repeat_ratio"] = float(repeat_count) / len(basket) if basket else 0.0

            # Category overlap between basket and history
            history_cats = {self.item_categories.get(it, "unknown") for it in user_history}
            basket_cat_set = set(basket_cats)
            cat_overlap = len(basket_cat_set & history_cats)
            features["category_overlap"] = float(cat_overlap)
            features["category_overlap_ratio"] = (
                float(cat_overlap) / len(basket_cat_set) if basket_cat_set else 0.0
            )
        else:
            features["history_length"] = 0.0
            features["history_unique_items"] = 0.0
            features["repeat_ratio"] = 0.0
            features["category_overlap"] = 0.0
            features["category_overlap_ratio"] = 0.0

        return features
