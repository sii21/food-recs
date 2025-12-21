"""Visualization of model evaluation results"""

import pickle
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from omegaconf import DictConfig


def load_results(models_dir: str = "artifacts/models") -> dict | None:
    """Load evaluation results"""
    results_path = Path(models_dir) / "evaluation_results.pkl"
    if results_path.exists():
        with open(results_path, "rb") as f:
            return pickle.load(f)
    return None


def create_metrics_comparison_chart(results: dict) -> go.Figure:
    """Create Hit@K comparison bar chart"""
    models = list(results.keys())
    k_values = [5, 10, 20]

    fig = go.Figure()
    colors = ["#636EFA", "#EF553B", "#00CC96"]
    max_y = 0

    for idx, model in enumerate(models):
        hit_values = [results[model][f"hit@{k}"] for k in k_values]
        max_y = max(max_y, max(hit_values))
        fig.add_trace(
            go.Bar(
                name=model,
                x=[f"Hit@{k}" for k in k_values],
                y=hit_values,
                text=[f"{v:.3f}" for v in hit_values],
                textposition="outside",
                marker_color=colors[idx],
            )
        )

    fig.update_layout(
        title="Сравнение Hit@K метрик по моделям",
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
    """Create MRR comparison bar chart"""
    models = list(results.keys())
    mrr_values = [results[model]["mrr"] for model in models]
    colors = ["#636EFA", "#EF553B", "#00CC96"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=models,
                y=mrr_values,
                text=[f"{v:.4f}" for v in mrr_values],
                textposition="outside",
                marker_color=colors,
            )
        ]
    )

    fig.update_layout(
        title="Mean Reciprocal Rank (MRR) по моделям",
        xaxis_title="Модель",
        yaxis_title="MRR",
        yaxis_range=[0, max(mrr_values) * 1.2 if mrr_values else 1],
        template="plotly_white",
        height=400,
    )

    return fig


def create_improvement_chart(results: dict) -> go.Figure:
    """Create improvement over baseline chart"""
    baseline = "TopPopular"
    ml_models = ["CooccurrenceLift", "Item2Vec"]
    metrics = ["hit@5", "hit@10", "hit@20", "mrr"]

    improvements: dict[str, dict[str, float]] = {}
    for model in ml_models:
        improvements[model] = {}
        for metric in metrics:
            baseline_val = results[baseline][metric]
            model_val = results[model][metric]
            if baseline_val > 0:
                improvement = ((model_val - baseline_val) / baseline_val) * 100
            else:
                improvement = 0.0
            improvements[model][metric] = improvement

    fig = go.Figure()
    colors = ["#EF553B", "#00CC96"]

    for idx, model in enumerate(ml_models):
        imp_values = [improvements[model][m] for m in metrics]
        fig.add_trace(
            go.Bar(
                name=model,
                x=["Hit@5", "Hit@10", "Hit@20", "MRR"],
                y=imp_values,
                text=[f"{v:+.1f}%" for v in imp_values],
                textposition="outside",
                marker_color=colors[idx],
            )
        )

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    all_improvements = [improvements[m][metric] for m in ml_models for metric in metrics]
    if all_improvements:
        ymin = min(all_improvements)
        ymax = max(all_improvements)
        pad = (ymax - ymin) * 0.2 if ymax != ymin else 10
        fig.update_yaxes(range=[ymin - pad, ymax + pad])

    fig.update_layout(
        title="Улучшение ML моделей над TopPopular baseline (%)",
        xaxis_title="Метрика",
        yaxis_title="Улучшение (%)",
        barmode="group",
        legend_title="Модель",
        template="plotly_white",
        height=450,
    )

    return fig


def create_summary_table(results: dict) -> pd.DataFrame:
    """Create summary DataFrame with ranks"""
    df = pd.DataFrame(results).T
    df = df[["hit@5", "hit@10", "hit@20", "mrr"]]
    df.columns = ["Hit@5", "Hit@10", "Hit@20", "MRR"]
    df = df.round(4)

    for col in df.columns:
        df[f"{col}_rank"] = df[col].rank(ascending=False).astype(int)

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
        print("❌ Results not found! Run training first.")
        return

    output_dir = Path(cfg.logging.plots_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating charts...")

    fig1 = create_metrics_comparison_chart(results)
    fig1.write_html(output_dir / "hit_comparison.html")
    fig1.write_image(output_dir / "hit_comparison.png", scale=2)
    print("  ✓ Hit@K comparison chart saved")

    fig2 = create_mrr_chart(results)
    fig2.write_html(output_dir / "mrr_comparison.html")
    fig2.write_image(output_dir / "mrr_comparison.png", scale=2)
    print("  ✓ MRR comparison chart saved")

    fig3 = create_improvement_chart(results)
    fig3.write_html(output_dir / "improvement_chart.html")
    fig3.write_image(output_dir / "improvement_chart.png", scale=2)
    print("  ✓ Improvement chart saved")

    summary_df = create_summary_table(results)
    summary_df.to_csv(output_dir / "summary_table.csv")

    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(summary_df.to_string())
    print(f"\n✓ All charts saved to: {output_dir}")
