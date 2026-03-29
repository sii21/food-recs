"""Item-side feature extraction for ensemble learning

Extracts features from product catalog and item statistics for LightGBM ranker
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


class ItemFeatureExtractor:
    """Extract item-level features from catalog and training data

    Features per candidate item:
    - item_popularity: purchase count in training data
    - item_popularity_log: log(1 + popularity)
    - item_popularity_rank: rank by popularity (0 = most popular)
    - description_length: character length of product description
    - category_size: number of items in the same category
    - avg_basket_size: average basket size this item appears in
    - cooccurrence_diversity: number of distinct co-occurring items
    """

    def __init__(self):
        self.item_popularity: dict[int, int] = {}
        self.item_popularity_rank: dict[int, int] = {}
        self.item_descriptions: dict[int, str] = {}
        self.item_categories: dict[int, str] = {}
        self.category_sizes: dict[str, int] = {}
        self.avg_basket_size: dict[int, float] = {}
        self.cooccurrence_diversity: dict[int, int] = {}

    def fit(
        self,
        train_baskets: list[list[int]],
        product_catalog: pd.DataFrame | None = None,
    ) -> "ItemFeatureExtractor":
        """Compute item statistics from training data and catalog

        Args:
            train_baskets: Training baskets
            product_catalog: Product catalog DataFrame with oms_id, description, category
        """
        # Item popularity
        counts: dict[int, int] = defaultdict(int)
        basket_sizes: dict[int, list[int]] = defaultdict(list)
        cooc_items: dict[int, set[int]] = defaultdict(set)

        for basket in train_baskets:
            basket_set = set(basket)
            for item in basket:
                counts[item] += 1
                basket_sizes[item].append(len(basket))
                cooc_items[item].update(basket_set - {item})

        self.item_popularity = dict(counts)

        # Popularity rank (0 = most popular)
        sorted_items = sorted(counts.keys(), key=lambda x: counts[x], reverse=True)
        self.item_popularity_rank = {item: rank for rank, item in enumerate(sorted_items)}

        # Average basket size per item
        self.avg_basket_size = {
            item: float(np.mean(sizes)) for item, sizes in basket_sizes.items()
        }

        # Co-occurrence diversity
        self.cooccurrence_diversity = {
            item: len(cooc_set) for item, cooc_set in cooc_items.items()
        }

        # Catalog features
        if product_catalog is not None and not product_catalog.empty:
            for _, row in product_catalog.iterrows():
                oms_id = int(row["oms_id"])
                self.item_descriptions[oms_id] = str(row.get("description", ""))
                self.item_categories[oms_id] = str(row.get("category", "unknown"))

            # Category sizes
            cat_counts: dict[str, int] = defaultdict(int)
            for cat in self.item_categories.values():
                cat_counts[cat] += 1
            self.category_sizes = dict(cat_counts)

        return self

    def extract(self, item_id: int) -> dict[str, float]:
        """Extract features for a single candidate item

        Args:
            item_id: Item ID

        Returns:
            Dict of feature name -> value
        """
        features: dict[str, float] = {}

        pop = self.item_popularity.get(item_id, 0)
        features["item_popularity"] = float(pop)
        features["item_popularity_log"] = float(np.log1p(pop))
        features["item_popularity_rank"] = float(
            self.item_popularity_rank.get(item_id, len(self.item_popularity))
        )

        desc = self.item_descriptions.get(item_id, "")
        features["description_length"] = float(len(desc))
        features["has_description"] = 1.0 if len(desc) > 0 else 0.0

        cat = self.item_categories.get(item_id, "unknown")
        features["category_size"] = float(self.category_sizes.get(cat, 0))

        features["avg_basket_size"] = self.avg_basket_size.get(item_id, 0.0)
        features["cooccurrence_diversity"] = float(
            self.cooccurrence_diversity.get(item_id, 0)
        )

        return features

    def extract_batch(self, item_ids: list[int]) -> list[dict[str, float]]:
        """Extract features for multiple candidate items

        Args:
            item_ids: List of item IDs

        Returns:
            List of feature dicts
        """
        return [self.extract(item_id) for item_id in item_ids]
