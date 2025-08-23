import os
import csv
import numpy as np
from dotenv import load_dotenv
import subprocess
import time
import signal
import sys
import logging
from config_manager import ConfigurationManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("experiment_runner.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.results = []

    def update_env_file(self, lr_unlearn, std):
        """Update the .env file with new parameters"""
        # Use the correct .env file path for HPC
        env_path = ".env"
        logger.info(f"Updating .env file at: {env_path}")

        try:
            # Read the current content
            with open(env_path, "r") as file:
                lines = file.readlines()

            # Update the values
            new_lines = []
            for line in lines:
                line = line.strip()
                if line.startswith("LR_LIPSCHITZ"):
                    new_lines.append(f"LR_LIPSCHITZ = {lr_unlearn}")
                elif line.startswith("NOISE_STD_LIPSCHITZ"):
                    new_lines.append(f"NOISE_STD_LIPSCHITZ = {std}")
                else:
                    new_lines.append(line)

            # Write back the updated content
            with open(env_path, "w") as file:
                file.write("\n".join(new_lines) + "\n")

            logger.info(
                f"Updated values: LR_UNLEARN={lr_unlearn}, noise_std={std}"
            )

            # Force reload of environment variables
            load_dotenv(env_path, override=True)

            # Reset the ConfigurationManager singleton
            ConfigurationManager._instance = None
            ConfigurationManager._initialized = False

            # Create a new ConfigurationManager instance
            self.config_manager = ConfigurationManager()

        except Exception as e:
            logger.error(f"Error updating .env file: {e}")
            raise

    def run_experiment(self, lr_unlearn, std):
        """Run a single experiment with given parameters"""
        logger.info(
            f"Running experiment with LR_UNLEARN={lr_unlearn}, " f"noise_std={std},"
        )

        # Update .env file with new parameters
        self.update_env_file(lr_unlearn, std)

        # Get the model name and dataset name from config
        model_name = self.config_manager.get_model_name()
        dataset_name = self.config_manager.get_dataset_name()
        n_clients = self.config_manager.get_n_clients()

        # Construct the expected results file path
        results_path = os.path.join(
            "server_results", f"server_{model_name}_{n_clients}_{dataset_name}.csv"
        )

        # Clear the previous results file if it exists
        if os.path.exists(results_path):
            logger.info(f"Clearing previous results file: {results_path}")
            os.remove(results_path)

        # Run the experiment
        try:
            # Run the experiment - don't use check=True since we expect an exception
            result = subprocess.run(
                ["python", "run.py"], capture_output=True, text=True
            )

            # Check if the experiment completed successfully (even with the expected exception)
            # The unlearning process intentionally raises an exception "learning and unlearning completed"
            if result.returncode != 0:
                # Check if this is the expected unlearning completion exception
                if "learning and unlearning completed" in result.stderr:
                    logger.info(
                        "Experiment completed successfully (unlearning phase finished as expected)"
                    )
                else:
                    logger.error(f"Experiment failed with unexpected error:")
                    logger.error(f"STDOUT: {result.stdout}")
                    logger.error(f"STDERR: {result.stderr}")
                    return  # Skip processing results for this failed experiment
            else:
                logger.info("Experiment completed successfully")

            logger.info(f"Looking for results file at: {results_path}")

            # Wait a bit to ensure file is written
            time.sleep(5)  # Increased wait time

            if os.path.exists(results_path):
                # Read the CSV file
                with open(results_path, "r") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        last_row = rows[-1]  # Get the last row
                        logger.info(f"Found results row: {last_row}")

                        # Extract metrics
                        metrics = {
                            "lr_unlearn": lr_unlearn,
                            "forget_accuracy": last_row.get("avg_forget_acc", None),
                            "retain_accuracy": last_row.get("avg_retain_acc", None),
                            "mia": last_row.get("avg_mia", None),
                            "anamnesis_index": last_row.get(
                                "avg_anamnesis_index", None
                            ),
                            "noise_std": std,
                            "original_forget_accuracy": last_row.get(
                                "avg_original_forget_acc", None
                            ),
                            "original_retain_accuracy": last_row.get(
                                "avg_original_retain_acc", None
                            ),
                            "dataset_name": dataset_name,
                            "n_clients": n_clients,
                        }

                        self.results.append(metrics)
                        logger.info(
                            f"Added metrics to results list. Current results count: {len(self.results)}"
                        )

                        # Save results after each run
                        self.save_results()
                    else:
                        logger.error(f"Results CSV file is empty at {results_path}")
                        # Try to read the file content for debugging
                        with open(results_path, "r") as f:
                            content = f.read()
                            logger.info(f"File content: {content}")
            else:
                logger.error(f"Results file not found at {results_path}")
                # List contents of server_results directory
                server_results_dir = "server_results"
                if os.path.exists(server_results_dir):
                    logger.info(f"Contents of {server_results_dir}:")
                    for file in os.listdir(server_results_dir):
                        logger.info(f"- {file}")
                else:
                    logger.error(f"Directory {server_results_dir} does not exist")

        except Exception as e:
            logger.error(f"Error running or processing experiment: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")

    def save_results(self, output_file="experiment_results.csv"):
        """Save experiment results to a CSV file, appending to existing results"""
        if self.results:
            # Create results directory if it doesn't exist
            results_dir = "experiment_results"
            os.makedirs(results_dir, exist_ok=True)

            output_path = os.path.join(results_dir, output_file)
            logger.info(f"Saving results to: {output_path}")

            try:
                # Get fieldnames from the first result
                fieldnames = self.results[0].keys()

                # Check if file exists to determine if we need to write header
                file_exists = os.path.exists(output_path)

                # Append to existing file or create new one
                with open(output_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if not file_exists:
                        writer.writeheader()
                    # Write only the latest result
                    writer.writerow(self.results[-1])

                logger.info(f"Successfully saved latest result to {output_path}")

            except Exception as e:
                logger.error(f"Error saving results: {e}")
                import traceback

                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
        else:
            logger.warning("No results to save")


def main():
    # Define parameter ranges
    learning_rates = [0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
    noise_std = [64, 128, 256]

    runner = ExperimentRunner()

    # Run experiments with all combinations
    for lr in learning_rates:
        for std in noise_std:
            runner.run_experiment(lr, std)
            time.sleep(5)  # Small delay between experiments

    # Save all results
    runner.save_results()


if __name__ == "__main__":
    main()
