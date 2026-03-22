"""Tests for recommendation models"""

import pytest

from food_recs.models import (
    CooccurrenceLiftRecommender,
    Item2VecRecommender,
    TopPopularRecommender,
)

# Toy dataset: 10 baskets with items from 1..20
BASKETS = [
    [1, 2, 3],
    [1, 2, 4],
    [1, 3, 5],
    [2, 3, 6],
    [1, 2, 3, 7],
    [4, 5, 6],
    [1, 2, 8],
    [3, 5, 9],
    [1, 2, 3, 10],
    [2, 3, 4, 5],
]


class TestTopPopular:
    def setup_method(self):
        self.model = TopPopularRecommender()
        self.model.fit(BASKETS)

    def test_fit_builds_counts(self):
        assert len(self.model.item_counts) > 0
        assert len(self.model.top_items) > 0

    def test_recommend_returns_list(self):
        recs = self.model.recommend([1, 2], k=5)
        assert isinstance(recs, list)
        assert len(recs) <= 5

    def test_recommend_excludes_basket_items(self):
        basket = [1, 2]
        recs = self.model.recommend(basket, k=10)
        for item in basket:
            assert item not in recs

    def test_recommend_respects_k(self):
        recs = self.model.recommend([1], k=3)
        assert len(recs) <= 3

    def test_recommend_empty_basket(self):
        recs = self.model.recommend([], k=5)
        assert len(recs) == 5


class TestCooccurrenceLift:
    def setup_method(self):
        self.model = CooccurrenceLiftRecommender(min_support=1)
        self.model.fit(BASKETS)

    def test_fit_builds_lift_matrix(self):
        assert len(self.model.lift_matrix) > 0

    def test_recommend_returns_list(self):
        recs = self.model.recommend([1, 2], k=5)
        assert isinstance(recs, list)
        assert len(recs) <= 5

    def test_recommend_excludes_basket_items(self):
        basket = [1, 2]
        recs = self.model.recommend(basket, k=10)
        for item in basket:
            assert item not in recs

    def test_recommend_unknown_items(self):
        recs = self.model.recommend([9999], k=5)
        assert isinstance(recs, list)


class TestItem2Vec:
    def setup_method(self):
        self.model = Item2VecRecommender()
        self.model.workers = 1
        self.model.min_count = 1
        self.model.epochs = 5
        self.model.vector_size = 16
        self.model.fit(BASKETS)

    def test_fit_builds_model(self):
        assert self.model.model is not None
        assert len(self.model.all_items) > 0

    def test_recommend_returns_list(self):
        recs = self.model.recommend([1, 2], k=5)
        assert isinstance(recs, list)
        assert len(recs) <= 5

    def test_recommend_excludes_basket_items(self):
        basket = [1, 2]
        recs = self.model.recommend(basket, k=10)
        for item in basket:
            assert item not in recs

    def test_recommend_unknown_items_returns_empty(self):
        recs = self.model.recommend([99999], k=5)
        assert recs == []

    def test_recommend_before_fit_returns_empty(self):
        model = Item2VecRecommender()
        recs = model.recommend([1, 2], k=5)
        assert recs == []
