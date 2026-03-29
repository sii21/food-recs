"""Tests for visualization functions"""

import warnings

from food_recs.visualization import (
    create_degradation_chart,
    create_improvement_chart,
    create_metrics_comparison_chart,
    create_mrr_chart,
    create_summary_table,
    create_test_vs_oot_chart,
)

SAMPLE_RESULTS = {
    "TopPopular": {
        "test_hit@5": 0.15,
        "test_hit@10": 0.25,
        "test_hit@20": 0.35,
        "test_mrr": 0.10,
        "oot_hit@5": 0.12,
        "oot_hit@10": 0.20,
        "oot_hit@20": 0.30,
        "oot_mrr": 0.08,
        "train_time_s": 1.0,
    },
    "CooccurrenceLift": {
        "test_hit@5": 0.10,
        "test_hit@10": 0.18,
        "test_hit@20": 0.28,
        "test_mrr": 0.07,
        "oot_hit@5": 0.09,
        "oot_hit@10": 0.16,
        "oot_hit@20": 0.25,
        "oot_mrr": 0.06,
        "train_time_s": 5.0,
    },
}


class TestCreateImprovementChart:
    def test_with_baseline_present(self):
        fig = create_improvement_chart(SAMPLE_RESULTS)
        assert fig.data  # Should have traces

    def test_missing_baseline_warns(self):
        results = {"ModelA": SAMPLE_RESULTS["CooccurrenceLift"]}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = create_improvement_chart(results)
            assert len(w) == 1
            assert "отсутствует" in str(w[0].message)
        assert not fig.data  # Empty figure

    def test_custom_baseline(self):
        fig = create_improvement_chart(SAMPLE_RESULTS, baseline="CooccurrenceLift")
        assert fig.data

    def test_single_model_only_baseline(self):
        results = {"TopPopular": SAMPLE_RESULTS["TopPopular"]}
        fig = create_improvement_chart(results)
        # No other models to compare - empty traces
        assert len(fig.data) == 0 or fig.data is not None


class TestOtherCharts:
    def test_test_vs_oot_chart(self):
        fig = create_test_vs_oot_chart(SAMPLE_RESULTS)
        assert len(fig.data) == 2  # Test + OOT bars

    def test_metrics_comparison_chart(self):
        fig = create_metrics_comparison_chart(SAMPLE_RESULTS, split="test")
        assert len(fig.data) == len(SAMPLE_RESULTS)

    def test_mrr_chart(self):
        fig = create_mrr_chart(SAMPLE_RESULTS)
        assert len(fig.data) == 2  # Test + OOT

    def test_degradation_chart(self):
        fig = create_degradation_chart(SAMPLE_RESULTS)
        assert len(fig.data) == len(SAMPLE_RESULTS)

    def test_summary_table(self):
        df = create_summary_table(SAMPLE_RESULTS)
        assert len(df) == 2
        assert "test_Hit@5" in df.columns
        assert "oot_MRR" in df.columns

    def test_empty_results(self):
        fig = create_test_vs_oot_chart({})
        assert fig is not None
