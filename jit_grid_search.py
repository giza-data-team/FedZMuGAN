import torch
from copy import deepcopy
import logging
import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product
import matplotlib.pyplot as plt

from client_utils.load_client_data import LoadClientData
from client_utils.weights_controller import WeightsController
from config_manager import ConfigurationManager
from Machine_Unlearning.Unlearning.metrics.all_metrics import get_metrics
from data_split import DataSplitter

from Machine_Unlearning.Unlearning.train.lipschitz_unlearning import LipschitzUnlearning


def aggregate_results(results: list[dict], n_clients):
    combined_metric = {}
    for result in results:
        for key, value in result.items():
            values = combined_metric.get(key, [])
            values.append(value)
            combined_metric[key] = values

    average_metrics = {}
    for key, total in combined_metric.items():
        average_metrics[key] = (sum(total) / n_clients) if n_clients > 0 else 0.0
    return average_metrics


class LipschitzGridSearch:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Configuration from config manager
        self.forget_class = config_manager.get_forget_class()
        self.dataset_name = config_manager.get_dataset_name()
        self.model_name = config_manager.get_model_name()
        self.batch_size = config_manager.get_batch_size_unlearn()
        self.num_classes = config_manager.get_num_classes()
        self.n_clients = config_manager.get_n_clients()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Results storage
        self.results = []
        self.experiment_id = (
            f"lipschitz_grid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.results_dir = f"./experiment_results/{self.experiment_id}"
        os.makedirs(self.results_dir, exist_ok=True)

        data_splitter = DataSplitter(
            num_clients=self.n_clients,
            min_instances_per_client=1000,
            dataset_name=self.dataset_name,
        )
        data_splitter.generate_and_split_data()
        self.load_model_and_data()

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        log_file = os.path.join(self.results_dir, "experiment.log")
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def define_parameter_grid(self):
        """
        Define the parameter grid for grid search (JIT-only).
        Only includes parameters actually used by jit_unlearn().
        """
        base_grid = {
            "learning_rate": [1e-7, 5e-7, 1e-6, 5e-6, 1e-5],
            "noise_std": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
            "num_variants": [40, 60, 80, 100, 120],
            "method": ["jit"],  # <-- JIT only
        }

        # Create parameter combinations
        base_keys = list(base_grid.keys())
        base_values = list(base_grid.values())
        param_combinations = []

        for combination in product(*base_values):
            params = dict(zip(base_keys, combination))
            param_combinations.append(params)

        # Optional: a couple of curated presets
        best_practice_combinations = [
            {
                "learning_rate": 1e-7,
                "noise_std": 0.10,
                "num_variants": 60,
                "method": "jit",
            },  # conservative
            {
                "learning_rate": 5e-7,
                "noise_std": 0.20,
                "num_variants": 80,
                "method": "jit",
            },  # standard
            {
                "learning_rate": 1e-6,
                "noise_std": 0.35,
                "num_variants": 120,
                "method": "jit",
            },  # aggressive
        ]
        param_combinations.extend(best_practice_combinations)

        self.logger.info(
            f"Generated {len(param_combinations)} parameter combinations (JIT)"
        )
        return param_combinations

    def load_model_and_data(self, client_id=0):
        """Load the original model and prepare data loaders"""
        model_path = os.path.join(
            self.config_manager.get_models_path_original(),
            f"model_{self.model_name.upper()}_{self.n_clients}_{self.dataset_name}.pth",
        )

        if not os.path.exists(model_path):
            self.logger.error(f"Model not found at {model_path}")
            raise FileNotFoundError(f"Model not found at {model_path}")

        weight_controller = WeightsController()
        self.original_model = weight_controller.load_model(model_path, self.device)

        # Load client data
        client_loader = LoadClientData(client_id=client_id)

        # Get data loaders
        self.train_loader, _, self.test_loaders = client_loader.get_normal_loaders(
            self.batch_size
        )

        (
            self.forget_train_loader,
            self.retain_train_loader,
        ) = client_loader.get_forget_retain_loaders(
            forget_class=self.forget_class, split="train", batch_size=self.batch_size
        )

        (
            self.forget_test_loader,
            self.retain_test_loader,
        ) = client_loader.get_forget_retain_loaders(
            forget_class=self.forget_class, split="test", batch_size=self.batch_size
        )

        self.logger.info("Model and data loaded successfully")

    def run_single_experiment(self, params, experiment_idx, total_experiments):
        """Run a single experiment with given parameters"""
        self.logger.info(f"Running experiment {experiment_idx + 1}/{total_experiments}")
        self.logger.info(f"Parameters: {params}")

        start_time = time.time()

        # Create a fresh copy of the model
        model = deepcopy(self.original_model)
        model.train()

        # Extract parameters
        method = params.pop("method")

        # Evaluate on all clients and aggregate
        all_metrics = []
        for client_id in range(self.n_clients):
            # --- CHANGED: JIT-only branch ---
            if method == "jit":
                jit = LipschitzUnlearning(
                    model=model,
                    device=self.device,
                    forget_dataloader=self.forget_train_loader,
                    opt_func=torch.optim.Adam,
                    learning_rate=params.get("learning_rate", 1e-6),
                    noise_std=params.get("noise_std", 0.2),
                    num_variants=params.get("num_variants", 100),
                )
                unlearned_model = jit.lipschitz_unlearn()
            else:
                raise ValueError(f"Unknown method: {method} (expected 'jit')")

            # Save unlearned weights (state_dict) for evaluation
            unlearned_model_path = os.path.join(
                self.config_manager.get_models_path_unlearn(),
                f"unlearned_{self.model_name}_{self.dataset_name}_forget_class_{self.config_manager.get_forget_class()}_jit.pth",
            )
            torch.save(unlearned_model.state_dict(), unlearned_model_path)
            metrics = self._evaluate_model(unlearned_model_path)
            self.load_model_and_data(client_id=client_id)
            all_metrics.append(metrics)

        metrics = aggregate_results(results=all_metrics, n_clients=self.n_clients)
        training_time = time.time() - start_time

        # Store results
        result = {
            "experiment_idx": experiment_idx,
            "method": method,
            "training_time": training_time,
            **params,  # lr, noise_std, num_variants remain here
            **metrics,
        }

        self.results.append(result)
        self._save_intermediate_results(result, experiment_idx)

        self.logger.info(
            f"Experiment {experiment_idx + 1} completed in {training_time:.2f}s"
        )
        self.logger.info(
            f"Forget accuracy: {metrics.get('forget_accuracy', 'N/A'):.4f}"
        )
        self.logger.info(
            f"Retain accuracy: {metrics.get('retain_accuracy', 'N/A'):.4f}"
        )

        return result

    def _evaluate_model(self, unlearned_model_path):
        metrics = {}
        final_metrics = get_metrics(
            original_model_path=os.path.join(
                self.config_manager.get_models_path_original(),
                f"model_{self.model_name.upper()}_{self.n_clients}_{self.dataset_name}.pth",
            ),
            unlearned_model_path=unlearned_model_path,
            scratch_model_path=os.path.join(
                self.config_manager.get_models_path_original(),
                f"scratch_model_{str(self.model_name).upper()}_{self.n_clients}_{self.dataset_name}.pth",
            ),
            forget_test_loader=self.forget_test_loader,
            retain_test_loader=self.retain_test_loader,
            device=self.device,  # --- CHANGED: respect runtime device ---
            forget_train_loader=self.forget_train_loader,
            retain_train_loader=self.retain_train_loader,
            test_loader=self.test_loaders,
            alpha=self.config_manager.get_alpha_unlearn_eval(),
        )

        metrics.update(
            {
                "mia": final_metrics["mia"],
                "anamnesis_index": final_metrics["anamnesis_index"],
                "forget_accuracy": final_metrics["forget_acc"],
                "retain_accuracy": final_metrics["retain_acc"],
                "original_forget_acc": final_metrics["original_forget_acc"],
                "original_retain_acc": final_metrics["original_retain_acc"],
                "original_accuracy": final_metrics["original_accuracy"],
            }
        )

        metrics["forget_acc_num_examples"] = len(self.forget_test_loader.dataset)
        metrics["retain_acc_num_examples"] = len(self.retain_test_loader.dataset)
        metrics["mia_num_examples"] = (
            len(self.retain_train_loader.dataset)
            + len(self.forget_train_loader.dataset)
            + len(self.test_loaders.dataset)
        )
        metrics["anamnesis_index_num_example"] = len(self.forget_train_loader)
        metrics["original_accuracy_num_examples"] = len(self.test_loaders.dataset)
        metrics.update({"n_clients": self.n_clients})
        return metrics

    def _save_intermediate_results(self, result, experiment_idx):
        result_file = os.path.join(
            self.results_dir, f"result_{experiment_idx:04d}.json"
        )
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)

        results_file = os.path.join(self.results_dir, "all_results.json")
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)

        df = pd.DataFrame(self.results)
        csv_file = os.path.join(self.results_dir, "results.csv")
        df.to_csv(csv_file, index=False)

    def run_grid_search(self, max_experiments=None):
        self.logger.info("Starting Lipschitz Grid Search Experiment (JIT)")
        self.logger.info(f"Experiment ID: {self.experiment_id}")

        param_combinations = self.define_parameter_grid()

        if max_experiments:
            param_combinations = param_combinations[:max_experiments]
            self.logger.info(f"Limited to {max_experiments} experiments")

        total_experiments = len(param_combinations)
        self.logger.info(f"Running {total_experiments} experiments")

        for idx, params in enumerate(param_combinations):
            self.run_single_experiment(params, idx, total_experiments)

        self.analyze_results()
        self.logger.info("Grid search experiment completed!")

    def analyze_results(self):
        if not self.results:
            self.logger.warning("No results to analyze")
            return

        df = pd.DataFrame(self.results)
        successful_df = df[~df.get("error", pd.Series()).notna()].copy()

        if successful_df.empty:
            self.logger.warning("No successful experiments to analyze")
            return

        best_experiments = self._find_best_experiments(successful_df)
        # self._create_visualizations(successful_df)
        self._save_analysis_summary(successful_df, best_experiments)
        self.logger.info("Results analysis completed")

    def _find_best_experiments(self, df):
        best_experiments = {}
        if "combined_score" in df.columns:
            best_overall_idx = df["combined_score"].idxmax()
            best_experiments["overall"] = df.loc[best_overall_idx].to_dict()
        if "unlearning_effectiveness" in df.columns:
            best_unlearning_idx = df["unlearning_effectiveness"].idxmax()
            best_experiments["unlearning"] = df.loc[best_unlearning_idx].to_dict()
        if "utility_preservation" in df.columns:
            best_utility_idx = df["utility_preservation"].idxmax()
            best_experiments["utility"] = df.loc[best_utility_idx].to_dict()
        if "training_time" in df.columns:
            fastest_idx = df["training_time"].idxmin()
            best_experiments["fastest"] = df.loc[fastest_idx].to_dict()
        return best_experiments

    def _create_visualizations(self, df):
        plt.style.use("seaborn-v0_8")
        # (kept as-is) ...

    def _save_analysis_summary(self, df, best_experiments):
        summary = {
            "experiment_overview": {
                "total_experiments": len(df),
                "successful_experiments": len(
                    df[~df.get("error", pd.Series()).notna()]
                ),
                "failed_experiments": len(df[df.get("error", pd.Series()).notna()])
                if "error" in df.columns
                else 0,
            },
            "performance_statistics": {},
            "best_experiments": best_experiments,
            "parameter_analysis": {},
        }

        for metric in [
            "forget_accuracy",
            "retain_accuracy",
            "training_time",
            "combined_score",
        ]:
            if metric in df.columns:
                summary["performance_statistics"][metric] = {
                    "mean": float(df[metric].mean()),
                    "std": float(df[metric].std()),
                    "min": float(df[metric].min()),
                    "max": float(df[metric].max()),
                    "median": float(df[metric].median()),
                }

        for param in ["learning_rate", "noise_std", "num_variants"]:
            if param in df.columns and "combined_score" in df.columns:
                best_params = df.nlargest(5, "combined_score")[param].tolist()
                summary["parameter_analysis"][param] = {
                    "best_values": best_params,
                    "mean_of_best": float(np.mean(best_params)),
                    "std_of_best": float(np.std(best_params)),
                }

        summary_file = os.path.join(self.results_dir, "analysis_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        self._create_readable_report(summary)

    def _create_readable_report(self, summary):
        report_file = os.path.join(self.results_dir, "experiment_report.txt")
        with open(report_file, "w") as f:
            f.write(f"Lipschitz Unlearning Grid Search Experiment Report\n")
            f.write(f"=" * 60 + "\n\n")
            f.write(f"Experiment ID: {self.experiment_id}\n")
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Forget Class: {self.forget_class}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("EXPERIMENT OVERVIEW\n")
            f.write("-" * 20 + "\n")
            overview = summary["experiment_overview"]
            f.write(f"Total experiments: {overview['total_experiments']}\n")
            f.write(f"Successful experiments: {overview['successful_experiments']}\n")
            f.write(f"Failed experiments: {overview['failed_experiments']}\n\n")

            f.write("BEST EXPERIMENTS\n")
            f.write("-" * 20 + "\n")
            for category, experiment in summary["best_experiments"].items():
                f.write(f"\nBest {category}:\n")
                f.write(f"  Method: {experiment.get('method', 'N/A')}\n")
                f.write(f"  Learning Rate: {experiment.get('learning_rate', 'N/A')}\n")
                f.write(f"  Noise Std: {experiment.get('noise_std', 'N/A')}\n")
                f.write(f"  Num Variants: {experiment.get('num_variants', 'N/A')}\n")
                f.write(
                    f"  Forget Accuracy: {experiment.get('forget_accuracy', 'N/A'):.4f}\n"
                )
                f.write(
                    f"  Retain Accuracy: {experiment.get('retain_accuracy', 'N/A'):.4f}\n"
                )
                f.write(
                    f"  Training Time: {experiment.get('training_time', 'N/A'):.2f}s\n"
                )

            f.write("\nPERFORMANCE STATISTICS\n")
            f.write("-" * 20 + "\n")
            for metric, stats in summary["performance_statistics"].items():
                f.write(f"\n{metric}:\n")
                f.write(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                f.write(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n")
                f.write(f"  Median: {stats['median']:.4f}\n")


def main():
    config_manager = ConfigurationManager()
    grid_search = LipschitzGridSearch(config_manager)
    grid_search.run_grid_search(max_experiments=100)  # adjust as needed
    print(f"Experiment completed! Results saved in: {grid_search.results_dir}")


if __name__ == "__main__":
    main()
