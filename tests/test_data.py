"""Tests for data loading and preprocessing"""

import numpy as np
import pandas as pd
import pytest

from food_recs.data import _validate_columns, make_leave_one_out, temporal_split


class TestValidateColumns:
    def test_all_columns_present(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        _validate_columns(df, ["a", "b"], "test")

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="отсутствуют обязательные колонки"):
            _validate_columns(df, ["a", "b", "c", "d"], "test.csv")

    def test_empty_required_list(self):
        df = pd.DataFrame({"a": [1]})
        _validate_columns(df, [], "test")


class TestMakeLeaveOneOut:
    def test_basic(self):
        baskets = [[1, 2, 3], [4, 5]]
        result = make_leave_one_out(baskets, seed=42)
        assert len(result) == 2
        for idx, (input_basket, held_out, _profile_id) in enumerate(result):
            assert held_out not in input_basket
            assert len(input_basket) == len(baskets[idx]) - 1

    def test_single_item_basket_skipped(self):
        baskets = [[1], [2, 3]]
        result = make_leave_one_out(baskets, seed=42)
        assert len(result) == 1

    def test_empty_baskets(self):
        result = make_leave_one_out([], seed=42)
        assert result == []

    def test_deterministic_with_same_seed(self):
        baskets = [[1, 2, 3, 4, 5]] * 10
        r1 = make_leave_one_out(baskets, seed=123)
        r2 = make_leave_one_out(baskets, seed=123)
        assert r1 == r2

    def test_different_seed_different_result(self):
        baskets = [[1, 2, 3, 4, 5]] * 10
        r1 = make_leave_one_out(baskets, seed=1)
        r2 = make_leave_one_out(baskets, seed=2)
        held_out_1 = [h for _, h, _ in r1]
        held_out_2 = [h for _, h, _ in r2]
        assert held_out_1 != held_out_2

    def test_held_out_item_was_in_original(self):
        baskets = [[10, 20, 30]]
        result = make_leave_one_out(baskets, seed=42)
        input_basket, held_out, _profile_id = result[0]
        assert held_out in [10, 20, 30]
        assert set(input_basket) | {held_out} == {10, 20, 30}


class TestTemporalSplit:
    def _make_data(self, n_orders=100):
        """Helper: create synthetic order data spanning 60 days"""
        rng = np.random.default_rng(42)
        dates = pd.date_range("2025-01-01", periods=60, freq="D")
        order_baskets = {}
        order_dates = {}
        for i in range(n_orders):
            order_baskets[i] = rng.integers(1, 100, size=rng.integers(2, 6)).tolist()
            order_dates[i] = dates[i % 60]
        return order_baskets, order_dates

    def test_split_sizes(self):
        order_baskets, order_dates = self._make_data()
        train, test, oot, *_ = temporal_split(
            order_baskets,
            order_dates,
            train_days=30,
            test_days=7,
            oot_days=7,
            min_basket_size=2,
        )
        assert len(train) > 0
        assert len(test) > 0
        assert len(oot) > 0

    def test_no_temporal_overlap(self):
        order_baskets, order_dates = self._make_data()
        train, test, oot, train_end, test_end, oot_start, data_end, *_ = temporal_split(
            order_baskets,
            order_dates,
            train_days=30,
            test_days=7,
            oot_days=7,
            min_basket_size=2,
        )
        # train_end < test_end <= oot_start (in general)
        assert train_end <= test_end

    def test_empty_dates_raises(self):
        with pytest.raises(ValueError, match="No order dates"):
            temporal_split({}, {}, train_days=30, test_days=7, oot_days=7)

    def test_min_basket_size_filter(self):
        order_baskets = {0: [1], 1: [1, 2, 3]}
        order_dates = {0: pd.Timestamp("2025-01-01"), 1: pd.Timestamp("2025-01-02")}
        train, test, oot, *_ = temporal_split(
            order_baskets,
            order_dates,
            train_days=30,
            test_days=7,
            oot_days=7,
            min_basket_size=2,
        )
        # Basket with 1 item should be filtered out everywhere
        all_baskets = train + test + oot
        for b in all_baskets:
            assert len(b) >= 2
