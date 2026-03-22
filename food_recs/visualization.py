"""Visualization of model evaluation results"""

import contextlib
import pickle
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from omegaconf import DictConfig

COLORS = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]


def _save_figure(fig: go.Figure, output_dir: Path, name: str) -> None:
    """Save figure as HTML and optionally PNG (if kaleido works)"""
    fig.write_html(output_dir / f"{name}.html")
    with contextlib.suppress(Exception):
        fig.write_image(output_dir / f"{name}.png", scale=2)


def load_results(models_dir: str = "artifacts/models") -> dict | None:
    """Load evaluation results"""
    results_path = Path(models_dir) / "evaluation_results.pkl"
    if results_path.exists():
        with open(results_path, "rb") as f:
            return pickle.load(f)
    return None


def create_test_vs_oot_chart(results: dict, metric: str = "hit@5") -> go.Figure:
    """Create grouped bar chart comparing Test vs OOT for all models"""
    models = list(results.keys())

    test_vals = [results[m].get(f"test_{metric}", 0) for m in models]
    oot_vals = [results[m].get(f"oot_{metric}", 0) for m in models]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Test",
            x=models,
            y=test_vals,
            text=[f"{v:.4f}" for v in test_vals],
            textposition="outside",
            marker_color="#2196F3",
        )
    )
    fig.add_trace(
        go.Bar(
            name="OOT",
            x=models,
            y=oot_vals,
            text=[f"{v:.4f}" for v in oot_vals],
            textposition="outside",
            marker_color="#FF9800",
        )
    )

    max_y = max(max(test_vals, default=0), max(oot_vals, default=0))
    fig.update_layout(
        title=f"Test vs OOT: {metric.upper()}",
        xaxis_title="Модель",
        yaxis_title=metric.upper(),
        yaxis_range=[0, max_y * 1.3 if max_y > 0 else 1],
        barmode="group",
        template="plotly_white",
        height=450,
    )
    return fig


def create_metrics_comparison_chart(results: dict, split: str = "test") -> go.Figure:
    """Create Hit@K comparison bar chart for a given split"""
    models = list(results.keys())
    k_values = [5, 10, 20]

    fig = go.Figure()
    max_y = 0

    for idx, model in enumerate(models):
        hit_values = [results[model].get(f"{split}_hit@{k}", 0) for k in k_values]
        max_y = max(max_y, max(hit_values, default=0))
        fig.add_trace(
            go.Bar(
                name=model,
                x=[f"Hit@{k}" for k in k_values],
                y=hit_values,
                text=[f"{v:.3f}" for v in hit_values],
                textposition="outside",
                marker_color=COLORS[idx % len(COLORS)],
            )
        )

    fig.update_layout(
        title=f"Сравнение Hit@K метрик ({split.upper()})",
        xaxis_title="Метрика",
        yaxis_title="Значение",
        yaxis_range=[0, max_y * 1.2 if max_y > 0 else 1],
        barmode="group",
        legend_title="Модель",
        template="plotly_white",
        height=500,
    )
    return fig


def create_mrr_chart(results: dict) -> go.Figure:
    """Create MRR comparison bar chart (test vs OOT)"""
    models = list(results.keys())
    test_mrr = [results[m].get("test_mrr", 0) for m in models]
    oot_mrr = [results[m].get("oot_mrr", 0) for m in models]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Test MRR",
            x=models,
            y=test_mrr,
            text=[f"{v:.4f}" for v in test_mrr],
            textposition="outside",
            marker_color="#2196F3",
        )
    )
    fig.add_trace(
        go.Bar(
            name="OOT MRR",
            x=models,
            y=oot_mrr,
            text=[f"{v:.4f}" for v in oot_mrr],
            textposition="outside",
            marker_color="#FF9800",
        )
    )

    max_y = max(max(test_mrr, default=0), max(oot_mrr, default=0))
    fig.update_layout(
        title="MRR: Test vs OOT",
        xaxis_title="Модель",
        yaxis_title="MRR",
        yaxis_range=[0, max_y * 1.3 if max_y > 0 else 1],
        barmode="group",
        template="plotly_white",
        height=400,
    )
    return fig


def create_improvement_chart(results: dict, baseline: str = "TopPopular") -> go.Figure:
    """Create improvement over baseline chart (test split)

    Args:
        results: Dict with model evaluation results
        baseline: Name of the baseline model to compare against
    """
    if baseline not in results:
        import warnings

        warnings.warn(
            f"Baseline модель '{baseline}' отсутствует в результатах, "
            f"график улучшений пропущен. Доступные модели: {list(results.keys())}",
            stacklevel=2,
        )
        return go.Figure()

    ml_models = [m for m in results if m != baseline]
    metrics = ["test_hit@5", "test_hit@10", "test_hit@20", "test_mrr"]
    metric_labels = ["Hit@5", "Hit@10", "Hit@20", "MRR"]

    improvements: dict[str, dict[str, float]] = {}
    for model in ml_models:
        improvements[model] = {}
        for metric, label in zip(metrics, metric_labels, strict=False):
            baseline_val = results[baseline].get(metric, 0)
            model_val = results[model].get(metric, 0)
            if baseline_val > 0:
                improvements[model][label] = ((model_val - baseline_val) / baseline_val) * 100
            else:
                improvements[model][label] = 0.0

    fig = go.Figure()
    for idx, model in enumerate(ml_models):
        imp_values = [improvements[model][label] for label in metric_labels]
        fig.add_trace(
            go.Bar(
                name=model,
                x=metric_labels,
                y=imp_values,
                text=[f"{v:+.1f}%" for v in imp_values],
                textposition="outside",
                marker_color=COLORS[(idx + 1) % len(COLORS)],
            )
        )

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    all_imp = [improvements[m][label] for m in ml_models for label in metric_labels]
    if all_imp:
        ymin, ymax = min(all_imp), max(all_imp)
        pad = (ymax - ymin) * 0.2 if ymax != ymin else 10
        fig.update_yaxes(range=[ymin - pad, ymax + pad])

    fig.update_layout(
        title="Улучшение над TopPopular baseline (%, Test split)",
        xaxis_title="Метрика",
        yaxis_title="Улучшение (%)",
        barmode="group",
        legend_title="Модель",
        template="plotly_white",
        height=450,
    )
    return fig


def create_degradation_chart(results: dict) -> go.Figure:
    """Create OOT degradation chart: (test - oot) / test * 100"""
    models = list(results.keys())
    metrics = [("hit@5", "Hit@5"), ("hit@10", "Hit@10"), ("mrr", "MRR")]

    fig = go.Figure()
    for idx, model in enumerate(models):
        degradations = []
        for metric_key, _ in metrics:
            test_val = results[model].get(f"test_{metric_key}", 0)
            oot_val = results[model].get(f"oot_{metric_key}", 0)
            if test_val > 0:
                degradations.append((test_val - oot_val) / test_val * 100)
            else:
                degradations.append(0.0)

        fig.add_trace(
            go.Bar(
                name=model,
                x=[label for _, label in metrics],
                y=degradations,
                text=[f"{v:.1f}%" for v in degradations],
                textposition="outside",
                marker_color=COLORS[idx % len(COLORS)],
            )
        )

    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="OOT Degradation (%, ниже = стабильнее)",
        xaxis_title="Метрика",
        yaxis_title="Деградация (%)",
        barmode="group",
        legend_title="Модель",
        template="plotly_white",
        height=450,
    )
    return fig


def create_summary_table(results: dict) -> pd.DataFrame:
    """Create summary DataFrame with test and OOT metrics"""
    rows = []
    for model, metrics in results.items():
        row = {"Model": model}
        for split in ["test", "oot"]:
            for k in [5, 10, 20]:
                row[f"{split}_Hit@{k}"] = metrics.get(f"{split}_hit@{k}", 0)
            row[f"{split}_MRR"] = metrics.get(f"{split}_mrr", 0)
        row["train_time_s"] = metrics.get("train_time_s", 0)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model").round(4)
    return df


def generate_plots(cfg: DictConfig) -> None:
    """Generate all visualization plots

    Args:
        cfg: Hydra config
    """
    print("=" * 60)
    print("GENERATING VISUALIZATION")
    print("=" * 60)

    results = load_results(cfg.model.models_dir)
    if results is None:
        print("Results not found! Run training first.")
        return

    output_dir = Path(cfg.logging.plots_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating charts...")

    charts = [
        (
            "hit_comparison_test",
            create_metrics_comparison_chart(results, split="test"),
            "Hit@K comparison (test)",
        ),
        (
            "hit_comparison_oot",
            create_metrics_comparison_chart(results, split="oot"),
            "Hit@K comparison (OOT)",
        ),
        (
            "test_vs_oot_hit5",
            create_test_vs_oot_chart(results, metric="hit@5"),
            "Test vs OOT Hit@5",
        ),
        ("mrr_comparison", create_mrr_chart(results), "MRR comparison"),
        ("improvement_chart", create_improvement_chart(results), "Improvement over baseline"),
        ("oot_degradation", create_degradation_chart(results), "OOT degradation"),
    ]

    for name, fig, label in charts:
        _save_figure(fig, output_dir, name)
        print(f"  + {label}")

    # Summary table
    summary_df = create_summary_table(results)
    summary_df.to_csv(output_dir / "summary_table.csv")

    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(summary_df.to_string())
    print(f"\n+ All charts saved to: {output_dir}")
