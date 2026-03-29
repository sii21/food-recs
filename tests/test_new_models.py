"""Tests for new production-grade recommender components"""

import numpy as np
import pandas as pd
import pytest

# Toy dataset
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

PRODUCT_CATALOG = pd.DataFrame(
    {
        "oms_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "name": [f"Item {i}" for i in range(1, 11)],
        "description": [
            "Свежий лосось филе",
            "Рис для суши",
            "Соевый соус классический",
            "Имбирь маринованный",
            "Васаби паста",
            "Нори листы для роллов",
            "Авокадо спелое",
            "Сливочный сыр филадельфия",
            "Огурец свежий",
            "Кунжут белый",
        ],
        "category": [
            "Рыба",
            "Крупы",
            "Соусы",
            "Приправы",
            "Приправы",
            "Морепродукты",
            "Овощи",
            "Молочные",
            "Овощи",
            "Приправы",
        ],
    }
)


class TestDebiasedEvaluator:
    def setup_method(self):
        from food_recs.evaluation.debiased_metrics import DebiasedEvaluator

        self.evaluator = DebiasedEvaluator(BASKETS, n_buckets=3)

    def test_bucket_assignment(self):
        assert len(self.evaluator.item_to_bucket) > 0
        for item, bucket in self.evaluator.item_to_bucket.items():
            assert 0 <= bucket < 3

    def test_evaluate_returns_standard_metrics(self):
        from food_recs.models import TopPopularRecommender

        model = TopPopularRecommender()
        model.fit(BASKETS)

        test_data = [([1, 2], 3, None), ([4, 5], 6, None), ([1, 3], 2, None)]
        results = self.evaluator.evaluate(model, test_data, [5, 10], split_name="test")

        assert "test_hit@5" in results
        assert "test_hit@10" in results
        assert "test_mrr" in results
        assert 0.0 <= results["test_hit@5"] <= 1.0

    def test_evaluate_returns_debiased_metrics(self):
        from food_recs.models import TopPopularRecommender

        model = TopPopularRecommender()
        model.fit(BASKETS)

        test_data = [([1, 2], 3, None), ([4, 5], 6, None), ([1, 3], 2, None)]
        results = self.evaluator.evaluate(model, test_data, [5, 10], split_name="test")

        assert "test_debiased_hit@5" in results
        assert "test_debiased_mrr" in results

    def test_evaluate_empty_test_data(self):
        from food_recs.models import TopPopularRecommender

        model = TopPopularRecommender()
        model.fit(BASKETS)
        results = self.evaluator.evaluate(model, [], [5], split_name="test")
        assert results["test_hit@5"] == 0.0


class TestStratifiedLeaveOneOut:
    def test_stratified_sampling(self):
        from food_recs.evaluation.debiased_metrics import stratified_leave_one_out

        test_data = [([1], 2, None), ([3], 4, None), ([5], 6, None), ([1], 3, None)]
        stratified, item_to_bucket = stratified_leave_one_out(
            test_data, BASKETS, n_buckets=3, seed=42
        )
        assert len(stratified) <= len(test_data)
        assert len(item_to_bucket) > 0


class TestUserFeatureExtractor:
    def setup_method(self):
        from food_recs.features.user_features import UserFeatureExtractor

        self.extractor = UserFeatureExtractor()
        item_cats = {i: PRODUCT_CATALOG.set_index("oms_id").loc[i, "category"] for i in range(1, 11)}
        self.extractor.fit(BASKETS, item_categories=item_cats)

    def test_extract_returns_dict(self):
        features = self.extractor.extract([1, 2, 3])
        assert isinstance(features, dict)
        assert "basket_size" in features
        assert features["basket_size"] == 3.0

    def test_extract_with_history(self):
        features = self.extractor.extract([1, 2], user_history=[3, 4, 5])
        assert features["history_length"] == 3.0
        assert features["history_unique_items"] == 3.0
        assert features["repeat_ratio"] == 0.0

    def test_extract_without_history(self):
        features = self.extractor.extract([1, 2])
        assert features["history_length"] == 0.0
        assert features["repeat_ratio"] == 0.0

    def test_popularity_features(self):
        features = self.extractor.extract([1, 2, 3])
        assert features["avg_item_popularity"] > 0
        assert features["max_item_popularity"] >= features["avg_item_popularity"]


class TestItemFeatureExtractor:
    def setup_method(self):
        from food_recs.features.item_features import ItemFeatureExtractor

        self.extractor = ItemFeatureExtractor()
        self.extractor.fit(BASKETS, product_catalog=PRODUCT_CATALOG)

    def test_extract_returns_dict(self):
        features = self.extractor.extract(1)
        assert isinstance(features, dict)
        assert "item_popularity" in features
        assert "item_popularity_log" in features
        assert "description_length" in features

    def test_extract_unknown_item(self):
        features = self.extractor.extract(9999)
        assert features["item_popularity"] == 0.0
        assert features["has_description"] == 0.0

    def test_extract_batch(self):
        batch = self.extractor.extract_batch([1, 2, 3])
        assert len(batch) == 3
        assert all(isinstance(f, dict) for f in batch)

    def test_popularity_rank(self):
        features_popular = self.extractor.extract(1)  # item 1 appears often
        features_rare = self.extractor.extract(10)  # item 10 appears rarely
        assert features_popular["item_popularity_rank"] < features_rare["item_popularity_rank"]


class TestSentenceTransformerBoostRecommender:
    """Tests for STBoost that don't require downloading the actual model"""

    def test_init_defaults(self):
        from food_recs.sentence_transformer_model import SentenceTransformerBoostRecommender

        model = SentenceTransformerBoostRecommender()
        assert model.cooc_weight == 0.5
        assert model.text_weight == 0.3
        assert model.st_model_name == "intfloat/multilingual-e5-base"

    def test_fit_without_catalog_uses_cooc_only(self):
        from food_recs.sentence_transformer_model import SentenceTransformerBoostRecommender

        model = SentenceTransformerBoostRecommender()
        model.fit(BASKETS, product_catalog=None)
        assert len(model.base_model.lift_matrix) > 0
        assert model._embeddings is None

    def test_recommend_without_catalog(self):
        from food_recs.sentence_transformer_model import SentenceTransformerBoostRecommender

        model = SentenceTransformerBoostRecommender()
        model.fit(BASKETS, product_catalog=None)
        recs = model.recommend([1, 2], k=5)
        assert isinstance(recs, list)
        assert len(recs) <= 5
        assert 1 not in recs
        assert 2 not in recs


class TestLGBMEnsembleRecommender:
    """Tests for LGBMEnsemble that use simple base models"""

    def setup_method(self):
        from food_recs.models import CooccurrenceLiftRecommender, TopPopularRecommender

        self.pop_model = TopPopularRecommender()
        self.pop_model.fit(BASKETS)
        self.cooc_model = CooccurrenceLiftRecommender(min_support=1)
        self.cooc_model.fit(BASKETS)

    def test_init(self):
        from food_recs.lgbm_ensemble import LGBMEnsembleRecommender

        model = LGBMEnsembleRecommender(
            base_models={"TopPopular": self.pop_model, "Cooc": self.cooc_model}
        )
        assert len(model.base_models) == 2

    def test_recommend_before_fit_uses_fallback(self):
        from food_recs.lgbm_ensemble import LGBMEnsembleRecommender

        model = LGBMEnsembleRecommender(
            base_models={"TopPopular": self.pop_model, "Cooc": self.cooc_model}
        )
        recs = model.recommend([1, 2], k=5)
        assert isinstance(recs, list)
        assert len(recs) <= 5

    def test_fit_and_recommend(self):
        pytest.importorskip("lightgbm")
        from food_recs.lgbm_ensemble import LGBMEnsembleRecommender

        model = LGBMEnsembleRecommender(
            base_models={"TopPopular": self.pop_model, "Cooc": self.cooc_model},
            n_candidates=10,
            n_estimators=10,
        )
        test_data = [([1, 2], 3, None), ([4, 5], 6, None), ([1, 3], 2, None)]
        model.fit(test_data, max_train_samples=10, seed=42)

        recs = model.recommend([1, 2], k=5)
        assert isinstance(recs, list)
        assert len(recs) <= 5
        assert 1 not in recs
        assert 2 not in recs
