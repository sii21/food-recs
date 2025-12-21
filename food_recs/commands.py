"""CLI entry point for food-recs package"""

import fire
import hydra

from food_recs.inference import run_inference
from food_recs.training import train_models
from food_recs.visualization import generate_plots


class Commands:
    """CLI commands for food recommendation system"""

    def train(self, config_path: str = "configs", config_name: str = "train"):
        """Train recommendation models

        Args:
            config_path: Path to config directory
            config_name: Name of config file (without .yaml)
        """
        with hydra.initialize(version_base=None, config_path=f"../{config_path}"):
            cfg = hydra.compose(config_name=config_name)
            train_models(cfg)

    def infer(self, basket: list[int] | None = None, model: str = "cooccurrence", top_k: int = 5):
        """Get recommendations for a basket

        Args:
            basket: List of item IDs in basket
            model: Model to use (popularity, cooccurrence, item2vec)
            top_k: Number of recommendations
        """
        if basket is None:
            basket = []
        run_inference(basket, model, top_k)

    def visualize(self, config_path: str = "configs", config_name: str = "train"):
        """Generate visualization plots for model evaluation"""
        with hydra.initialize(version_base=None, config_path=f"../{config_path}"):
            cfg = hydra.compose(config_name=config_name)
            generate_plots(cfg)

    def app(self):
        """Run Streamlit demo app"""
        import subprocess
        import sys

        subprocess.run([sys.executable, "-m", "streamlit", "run", "food_recs/app.py"], check=True)


def main():
    """Main entry point"""
    fire.Fire(Commands)


if __name__ == "__main__":
    main()
