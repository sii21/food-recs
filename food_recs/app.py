"""Streamlit demo application for food recommendations"""

import pickle
import random
from pathlib import Path

import pandas as pd
import streamlit as st

from food_recs.models import (
    CooccurrenceLiftRecommender,
    ImplicitALSRecommender,
    ImplicitBPRRecommender,
    Item2VecRecommender,
    TopPopularRecommender,
)

MODELS_DIR = Path("artifacts/models")


@st.cache_resource
def load_models() -> dict:
    """Load trained models"""
    models = {}
    model_classes = {
        "toppopular": TopPopularRecommender,
        "cooccurrencelift": CooccurrenceLiftRecommender,
        "item2vec": Item2VecRecommender,
        "implicitals": ImplicitALSRecommender,
        "implicitbpr": ImplicitBPRRecommender,
    }

    for name in model_classes:
        model_path = MODELS_DIR / f"{name}_model.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                models[name] = pickle.load(f)

    return models


@st.cache_resource
def load_item_mapping() -> dict[int, str]:
    """Load item ID to name mapping"""
    mapping_path = MODELS_DIR / "item_mapping.pkl"
    if mapping_path.exists():
        with open(mapping_path, "rb") as f:
            return pickle.load(f)
    return {}


@st.cache_resource
def load_train_baskets() -> list[list[int]]:
    """Load training baskets for examples"""
    baskets_path = MODELS_DIR / "train_baskets.pkl"
    if baskets_path.exists():
        with open(baskets_path, "rb") as f:
            return pickle.load(f)
    return []


@st.cache_resource
def load_evaluation_results() -> dict:
    """Load evaluation results"""
    results_path = MODELS_DIR / "evaluation_results.pkl"
    if results_path.exists():
        with open(results_path, "rb") as f:
            return pickle.load(f)
    return {}


def get_item_name(item_id: int, mapping: dict[int, str]) -> str:
    """Get item name by ID"""
    return mapping.get(item_id, f"Item #{item_id}")


def main() -> None:
    st.set_page_config(page_title="Food Recommender", page_icon="🍣", layout="wide")

    st.title("🍣 Рекомендация товаров для ресторана")
    st.markdown("**Демонстрация работы алгоритмов рекомендаций**")

    models = load_models()
    item_mapping = load_item_mapping()
    train_baskets = load_train_baskets()
    eval_results = load_evaluation_results()

    if not models:
        st.error("❌ Модели не найдены! Сначала запустите обучение: food-recs train")
        return

    with st.sidebar:
        st.header("📊 Качество моделей")

        if eval_results:
            rows = []
            for model, metrics in eval_results.items():
                rows.append(
                    {
                        "Model": model,
                        "Test Hit@5": metrics.get("test_hit@5", 0),
                        "Test MRR": metrics.get("test_mrr", 0),
                        "OOT Hit@5": metrics.get("oot_hit@5", 0),
                        "OOT MRR": metrics.get("oot_mrr", 0),
                    }
                )
            metrics_df = pd.DataFrame(rows).set_index("Model").round(4)
            st.dataframe(metrics_df, width=350)

            st.markdown("---")
            st.markdown(
                """
            **Метрики:**
            - **Hit@K** - доля случаев, когда скрытый товар попал в топ-K
            - **MRR** - средняя обратная позиция правильного товара
            - **Test** - неделя сразу после обучения
            - **OOT** - данные через 14 месяцев
            """
            )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🛒 Корзина пользователя")

        if st.button("🎲 Случайная корзина", use_container_width=True):
            valid_baskets = [b for b in train_baskets if len(b) >= 2]
            if valid_baskets:
                random_basket = random.choice(valid_baskets)
                st.session_state["selected_items"] = random_basket[:3]

        all_items = list(item_mapping.keys())
        item_options = {get_item_name(item_id, item_mapping): item_id for item_id in all_items}

        selected_names = st.multiselect(
            "Выберите товары в корзине:",
            options=list(item_options.keys()),
            default=None,
            max_selections=10,
        )

        selected_items = [item_options[name] for name in selected_names]

        if "selected_items" in st.session_state and not selected_items:
            selected_items = st.session_state["selected_items"]
            st.info("Загружена случайная корзина:")
            for item_id in selected_items:
                st.write(f"• {get_item_name(item_id, item_mapping)}")

        top_k = st.slider("Количество рекомендаций:", 3, 20, 5)

    with col2:
        st.subheader("💡 Рекомендации")

        if not selected_items:
            st.info("👈 Выберите товары или нажмите 'Случайная корзина'")
        else:
            tabs = st.tabs(
                [
                    "TopPopular (Baseline)",
                    "Co-occurrence + Lift",
                    "Item2Vec",
                    "ImplicitALS",
                    "ImplicitBPR",
                ]
            )
            model_keys = [
                "toppopular",
                "cooccurrencelift",
                "item2vec",
                "implicitals",
                "implicitbpr",
            ]
            model_display_names = [
                "TopPopular",
                "CooccurrenceLift",
                "Item2Vec",
                "ImplicitALS",
                "ImplicitBPR",
            ]

            for tab, model_key, display_name in zip(
                tabs, model_keys, model_display_names, strict=True
            ):
                with tab:
                    if model_key in models:
                        recs = models[model_key].recommend(selected_items, k=top_k)
                        if recs:
                            for idx, item_id in enumerate(recs, 1):
                                item_name = get_item_name(item_id, item_mapping)
                                st.write(f"**{idx}.** {item_name}")
                        else:
                            st.warning("Нет рекомендаций для данной корзины")
                    else:
                        st.error(f"Модель {display_name} не загружена")

            st.markdown("---")
            st.subheader("📈 Сравнение рекомендаций")

            comparison_data = []
            for model_key, display_name in zip(model_keys, model_display_names, strict=True):
                if model_key in models:
                    recs = models[model_key].recommend(selected_items, k=top_k)
                    for idx, item_id in enumerate(recs, 1):
                        comparison_data.append(
                            {
                                "Модель": display_name,
                                "Ранг": idx,
                                "Товар": get_item_name(item_id, item_mapping),
                            }
                        )

            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                pivot_df = comparison_df.pivot(index="Ранг", columns="Модель", values="Товар")
                available_cols = [c for c in model_display_names if c in pivot_df.columns]
                if available_cols:
                    pivot_df = pivot_df[available_cols]
                st.dataframe(pivot_df, width=1000)


if __name__ == "__main__":
    main()
