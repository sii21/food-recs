"""LightGBM-based learned ensemble ranker

Combines scores from multiple base recommenders with user/item features
using a LightGBM ranker to learn optimal blending weights
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _try_import_lightgbm():
    try:
        import lightgbm as lgb

        return lgb
    except ImportError:
        raise ImportError(
            "lightgbm is required for LGBMEnsembleRecommender. "
            "Install with: pip install lightgbm>=3.3.2"
        )


class LGBMEnsembleRecommender:
    """Learned ensemble using LightGBM ranker over base model scores + features

    Training:
    1. For each L1O test example, generate candidates from base models
    2. Extract features: base model scores + user features + item features
    3. Train LightGBM ranker with binary relevance (1 if candidate == held_out)

    Inference:
    1. Get candidates from all base models
    2. Extract features for each candidate
    3. Score with LightGBM and return top-k
    """

    def __init__(
        self,
        base_models: dict[str, object],
        user_feature_extractor=None,
        item_feature_extractor=None,
        n_candidates: int = 50,
        lgbm_params: dict | None = None,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
    ):
        self.base_models = base_models
        self.user_feature_extractor = user_feature_extractor
        self.item_feature_extractor = item_feature_extractor
        self.n_candidates = n_candidates
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.lgbm_params = lgbm_params or {}
        self.ranker = None
        self.feature_names: list[str] = []

    def _get_candidates_with_scores(
        self, basket: list[int]
    ) -> dict[int, dict[str, float]]:
        """Get candidate items and their scores from all base models

        Returns:
            Dict mapping item_id -> {model_name: score, ...}
        """
        basket_set = set(basket)
        candidate_scores: dict[int, dict[str, float]] = defaultdict(dict)

        for model_name, model in self.base_models.items():
            recs = model.recommend(basket, k=self.n_candidates)
            for rank, item_id in enumerate(recs):
                if item_id not in basket_set:
                    # Reciprocal rank as score
                    candidate_scores[item_id][f"score_{model_name}"] = 1.0 / (rank + 1)
                    candidate_scores[item_id][f"rank_{model_name}"] = float(rank)

        return candidate_scores

    def _build_feature_vector(
        self,
        item_id: int,
        model_scores: dict[str, float],
        basket: list[int],
        user_history: list[int] | None = None,
    ) -> dict[str, float]:
        """Build full feature vector for one candidate item"""
        features = dict(model_scores)

        # Fill missing model scores with 0
        for model_name in self.base_models:
            if f"score_{model_name}" not in features:
                features[f"score_{model_name}"] = 0.0
            if f"rank_{model_name}" not in features:
                features[f"rank_{model_name}"] = float(self.n_candidates)

        # User features
        if self.user_feature_extractor is not None:
            user_feats = self.user_feature_extractor.extract(basket, user_history)
            features.update(user_feats)

        # Item features
        if self.item_feature_extractor is not None:
            item_feats = self.item_feature_extractor.extract(item_id)
            features.update(item_feats)

        return features

    def fit(
        self,
        train_data: list[tuple[list[int], int, int | None]],
        user_histories: dict[int, list[int]] | None = None,
        max_train_samples: int = 50000,
        seed: int = 42,
    ) -> "LGBMEnsembleRecommender":
        """Train the LightGBM ranker on leave-one-out data

        Args:
            train_data: L1O tuples (input_basket, held_out_item, profile_id)
            user_histories: profile_id -> list of historical item purchases
            max_train_samples: Cap on number of L1O examples to use
            seed: Random seed
        """
        lgb = _try_import_lightgbm()

        rng = np.random.default_rng(seed)
        if len(train_data) > max_train_samples:
            indices = rng.choice(len(train_data), size=max_train_samples, replace=False)
            train_data = [train_data[i] for i in indices]
            print(f"LGBMEnsemble: subsampled to {max_train_samples:,} training examples")

        all_features: list[dict[str, float]] = []
        all_labels: list[int] = []
        group_sizes: list[int] = []

        user_histories = user_histories or {}

        print(f"Building LGBMEnsemble training data from {len(train_data):,} L1O examples...")
        for entry in train_data:
            input_basket, held_out_item = entry[0], entry[1]
            profile_id = entry[2] if len(entry) > 2 else None

            user_hist = user_histories.get(profile_id) if profile_id is not None else None

            candidate_scores = self._get_candidates_with_scores(input_basket)
            if not candidate_scores:
                continue

            group_size = 0
            for item_id, scores in candidate_scores.items():
                fv = self._build_feature_vector(item_id, scores, input_basket, user_hist)
                all_features.append(fv)
                all_labels.append(1 if item_id == held_out_item else 0)
                group_size += 1

            # If held_out not in candidates, add it with zero scores
            if held_out_item not in candidate_scores:
                fv = self._build_feature_vector(
                    held_out_item, {}, input_basket, user_hist
                )
                all_features.append(fv)
                all_labels.append(1)
                group_size += 1

            group_sizes.append(group_size)

        if not all_features:
            print("LGBMEnsemble: no training data generated, skipping")
            return self

        # Convert to DataFrame
        df = pd.DataFrame(all_features).fillna(0.0)
        self.feature_names = list(df.columns)
        X = df.values.astype(np.float32)
        y = np.array(all_labels, dtype=np.int32)

        print(
            f"LGBMEnsemble training: {X.shape[0]:,} rows, "
            f"{X.shape[1]} features, {len(group_sizes):,} groups"
        )

        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [5, 10],
            "learning_rate": self.learning_rate,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbose": -1,
            "seed": seed,
            **self.lgbm_params,
        }

        train_set = lgb.Dataset(
            X, label=y, group=group_sizes, feature_name=self.feature_names
        )

        self.ranker = lgb.train(
            params,
            train_set,
            num_boost_round=self.n_estimators,
        )

        # Feature importance
        importance = self.ranker.feature_importance(importance_type="gain")
        feat_imp = sorted(
            zip(self.feature_names, importance),
            key=lambda x: x[1],
            reverse=True,
        )
        print("\nTop-10 feature importances (gain):")
        for fname, imp in feat_imp[:10]:
            print(f"  {fname}: {imp:.1f}")

        return self

    def recommend(self, basket: list[int], k: int = 10) -> list[int]:
        if self.ranker is None:
            # Fallback: use first available base model
            for model in self.base_models.values():
                return model.recommend(basket, k=k)
            return []

        candidate_scores = self._get_candidates_with_scores(basket)
        if not candidate_scores:
            for model in self.base_models.values():
                return model.recommend(basket, k=k)
            return []

        features_list = []
        item_ids = []
        for item_id, scores in candidate_scores.items():
            fv = self._build_feature_vector(item_id, scores, basket)
            features_list.append(fv)
            item_ids.append(item_id)

        df = pd.DataFrame(features_list, columns=self.feature_names).fillna(0.0)
        X = df.values.astype(np.float32)
        preds = self.ranker.predict(X)

        ranked = sorted(zip(item_ids, preds), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in ranked[:k]]
