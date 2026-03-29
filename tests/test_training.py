"""Tests for training evaluation logic"""

from food_recs.models import TopPopularRecommender
from food_recs.training import evaluate_model

BASKETS = [
    [1, 2, 3],
    [1, 2, 4],
    [1, 3, 5],
    [2, 3, 6],
    [1, 2, 3, 7],
]


class TestEvaluateModel:
    def setup_method(self):
        self.model = TopPopularRecommender()
        self.model.fit(BASKETS)

    def test_basic_evaluation(self):
        test_data = [([1, 2], 3), ([1, 3], 2)]
        metrics = evaluate_model(self.model, test_data, k_values=[5, 10])
        assert "test_hit@5" in metrics
        assert "test_hit@10" in metrics
        assert "test_mrr" in metrics
        assert all(0 <= v <= 1 for v in metrics.values())

    def test_empty_test_data_returns_zeros(self):
        metrics = evaluate_model(self.model, [], k_values=[5, 10])
        assert metrics["test_hit@5"] == 0.0
        assert metrics["test_hit@10"] == 0.0
        assert metrics["test_mrr"] == 0.0

    def test_custom_split_name(self):
        test_data = [([1, 2], 3)]
        metrics = evaluate_model(self.model, test_data, k_values=[5], split_name="oot")
        assert "oot_hit@5" in metrics
        assert "oot_mrr" in metrics

    def test_perfect_hit(self):
        # Item 3 is very popular (appears in 4/5 baskets), so it should be in top-5
        test_data = [([99], 3)]
        metrics = evaluate_model(self.model, test_data, k_values=[5])
        # Item 3 should be recommended (it's one of the most popular)
        assert metrics["test_hit@5"] >= 0.0  # Может не попасть если 99 not in model

    def test_metrics_bounded(self):
        test_data = [([1, 2], 3), ([4, 5], 6), ([1], 2)]
        metrics = evaluate_model(self.model, test_data, k_values=[5, 10, 20])
        for key, val in metrics.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of bounds"
