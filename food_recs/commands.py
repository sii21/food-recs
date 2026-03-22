"""CLI entry point for food-recs package"""

from pathlib import Path

import fire
import hydra

from food_recs.inference import run_inference
from food_recs.training import train_models
from food_recs.visualization import generate_plots

# Resolve config directory relative to the project root (one level up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_DIR = str(_PROJECT_ROOT / "configs")


class Commands:
    """CLI commands for food recommendation system"""

    def train(self, config_path: str = _DEFAULT_CONFIG_DIR, config_name: str = "train"):
        """Train recommendation models

        Args:
            config_path: Absolute path to config directory
            config_name: Name of config file (without .yaml)
        """
        config_path = str(Path(config_path).resolve())
        with hydra.initialize_config_dir(version_base=None, config_dir=config_path):
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

    def visualize(self, config_path: str = _DEFAULT_CONFIG_DIR, config_name: str = "train"):
        """Generate visualization plots for model evaluation"""
        config_path = str(Path(config_path).resolve())
        with hydra.initialize_config_dir(version_base=None, config_dir=config_path):
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
