import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
import seaborn as sns

from client_utils.load_client_data import LoadClientData
from client_utils.weights_controller import WeightsController
from config_manager import ConfigurationManager
from Machine_Unlearning.Unlearning.metrics.all_metrics import get_metrics
from data_split import DataSplitter
from test_jit import DeviceDataLoader


def aggregate_results(results: list[dict], n_clients):
    """Aggregates experiment results with better statistical handling"""
    if not results:
        return {}

    combined_metrics = {}
    for result in results:
        for key, value in result.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                combined_metrics.setdefault(key, []).append(value)

    aggregated = {}
    for key, values in combined_metrics.items():
        if values:  # Only process non-empty lists
            aggregated[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values)
            }

    return aggregated


class UNSIR_noise(torch.nn.Module):
    """
    UNSIR noise module for generating adversarial noise (from second script).
    """
    def __init__(self, *dim):
        super().__init__()
        self.noise = torch.nn.Parameter(torch.randn(*dim), requires_grad=True)

    def forward(self):
        return self.noise


class EMMNGridSearch:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Configuration
        self.forget_class = config_manager.get_forget_class()
        self.dataset_name = config_manager.get_dataset_name()
        self.model_name = config_manager.get_model_name()
        self.batch_size = config_manager.get_batch_size_unlearn()
        self.num_classes = config_manager.get_num_classes()
        self.n_clients = config_manager.get_n_clients()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Results storage
        self.results = []
        self.experiment_id = f"emmn_grid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir = f"./emmn_grid_experiment_results/{self.experiment_id}"
        os.makedirs(self.results_dir, exist_ok=True)

        # Initialize data
        self._initialize_data()
        self._setup_logging()

    def _initialize_data(self):
        """Initialize data splitting and loading"""
        data_splitter = DataSplitter(
            num_clients=self.n_clients,
            min_instances_per_client=1000,
            dataset_name=self.dataset_name,
        )
        data_splitter.generate_and_split_data()
        self.load_model_and_data()

    def _setup_logging(self):
        """Setup comprehensive logging"""
        log_file = os.path.join(self.results_dir, 'experiment.log')
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def define_parameter_grid(self):
        """Define parameter grid based on basic EMMN implementation"""

        # Simplified parameter ranges based on the basic EMMN implementation
        grid = {
            'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3],  # For final fine-tuning
            'unsir_epochs': [15, 20, 25, 30],            # UNSIR noise training epochs
            'emmn_epochs': [20, 25, 30, 35],             # EMMN noise training epochs
            'finetune_epochs': [1, 2, 3, 5],             # Final model fine-tuning epochs
            'noise_batch_size': [4, 8, 16],              # Noise batch size
            'num_noise_batches': [60, 80, 100],          # Number of noise batches
            'unsir_lr': [0.05, 0.1, 0.15, 0.2],         # UNSIR learning rate (fixed in original)
            'emmn_lr': [0.05, 0.1, 0.15, 0.2],          # EMMN learning rate (fixed in original)
        }

        # Generate all combinations
        param_combinations = []

        for lr in grid['learning_rate']:
            for unsir_epochs in grid['unsir_epochs']:
                for emmn_epochs in grid['emmn_epochs']:
                    for finetune_epochs in grid['finetune_epochs']:
                        for noise_batch_size in grid['noise_batch_size']:
                            for num_noise_batches in grid['num_noise_batches']:
                                for unsir_lr in grid['unsir_lr']:
                                    for emmn_lr in grid['emmn_lr']:
                                        params = {
                                            'learning_rate': lr,
                                            'unsir_epochs': unsir_epochs,
                                            'emmn_epochs': emmn_epochs,
                                            'finetune_epochs': finetune_epochs,
                                            'noise_batch_size': noise_batch_size,
                                            'num_noise_batches': num_noise_batches,
                                            'unsir_lr': unsir_lr,
                                            'emmn_lr': emmn_lr
                                        }
                                        param_combinations.append(params)

        # Shuffle and limit to manageable size
        np.random.shuffle(param_combinations)
        param_combinations = param_combinations[:150]  # Limit to 150 experiments

        print(f"Generated {len(param_combinations)} parameter combinations")
        return param_combinations

    def load_model_and_data(self, client_id=0):
        """Load model and data with better error handling"""
        try:
            # Load model
            model_path = os.path.join(
                self.config_manager.get_models_path_original(),
                f"model_{self.model_name.upper()}_{self.n_clients}_{self.dataset_name}.pth"
            )

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")

            weight_controller = WeightsController()
            self.original_model = weight_controller.load_model(model_path, self.device)

            # Load client data
            client_loader = LoadClientData(client_id=client_id)

            # Get data loaders
            self.train_loader, _, self.test_loaders = client_loader.get_normal_loaders(self.batch_size)

            self.forget_train_loader, self.retain_train_loader = client_loader.get_forget_retain_loaders(
                forget_class=self.forget_class, split="train", batch_size=self.batch_size
            )
            self.forget_test_loader, self.retain_test_loader = client_loader.get_forget_retain_loaders(
                forget_class=self.forget_class, split="test", batch_size=self.batch_size
            )

            print(f"Model and data loaded successfully for client {client_id}")

        except Exception as e:
            print(f"Error loading model and data: {e}")
            raise

    def UNSIR_noise_train(self, noise, model, forget_class_label, num_epochs, noise_batch_size, lr=0.1):
        """
        Train UNSIR noise for the forget class (from second script).
        """
        opt = torch.optim.Adam(noise.parameters(), lr=lr)

        for epoch in range(num_epochs):
            total_loss = []
            inputs = noise()
            labels = torch.zeros(noise_batch_size).to(self.device) + forget_class_label

            # Handle different model output formats
            model_output = model(inputs)
            if isinstance(model_output, tuple):
                outputs = model_output[0]
            else:
                outputs = model_output

            # Ensure labels are integers
            labels = labels.long()

            loss = -F.cross_entropy(outputs, labels) + 0.1 * torch.mean(
                torch.sum(inputs**2, [1, 2, 3])
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss.append(loss.cpu().detach().numpy())

            if epoch % 10 == 0:
                print(f"UNSIR Epoch {epoch}, Loss: {np.mean(total_loss):.4f}")

        return noise

    def emmn_noise_train(self, noise, model, target_class_label, num_epochs, noise_batch_size, lr=0.1):
        """
        Train EMMN noise for retain classes (from second script).
        """
        opt = torch.optim.Adam(noise.parameters(), lr=lr)

        for epoch in range(num_epochs):
            total_loss = []
            inputs = noise()
            labels = torch.zeros(noise_batch_size).to(self.device) + target_class_label

            # Handle different model output formats
            model_output = model(inputs)
            if isinstance(model_output, tuple):
                outputs = model_output[0]
            else:
                outputs = model_output

            # Ensure labels are integers
            labels = labels.long()

            loss = F.cross_entropy(outputs, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss.append(loss.cpu().detach().numpy())

            if epoch % 10 == 0:
                print(f"EMMN Epoch {epoch}, Loss: {np.mean(total_loss):.4f}")

        return noise

    def emmc_create_noisy_loader(self, noise_dict, batch_size=64, num_noise_batches=80):
        """
        Create a DataLoader with noisy data for all classes (from second script).
        """
        noisy_data = []

        for i in range(num_noise_batches):
            for class_label, noise_module in noise_dict.items():
                batch = noise_module()
                for k in range(batch.size(0)):
                    noisy_data.append(
                        (
                            batch[k].detach().cpu(),
                            torch.tensor(class_label),
                            torch.tensor(class_label)
                        )
                    )

        noisy_loader = DataLoader(noisy_data, batch_size=batch_size, shuffle=True)
        return noisy_loader

    def fit(self, epochs, lr, model, train_loader, opt_func=torch.optim.Adam):
        """
        Train model with noisy data (from second script).
        """
        optimizer = opt_func(model.parameters(), lr)

        for epoch in range(epochs):
            model.train()
            train_losses = []

            for batch in train_loader:
                if hasattr(model, 'training_step'):
                    # If model has training_step method
                    loss = model.training_step(batch)
                else:
                    # Standard training step
                    inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = F.cross_entropy(outputs, labels)

                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Log training loss
            if len(train_losses) > 0:
                avg_train_loss = torch.stack(train_losses).mean().item()
                if epoch % 1 == 0:
                    print(f"Finetune Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}")

    def run_basic_emmn_unlearning(self, model, params):
        """
        Basic EMMN implementation (based on second script)
        """
        # Extract parameters
        noise_batch_size = params['noise_batch_size']
        unsir_epochs = params['unsir_epochs']
        emmn_epochs = params['emmn_epochs']
        finetune_epochs = params['finetune_epochs']
        learning_rate = params['learning_rate']
        num_noise_batches = params['num_noise_batches']
        unsir_lr = params['unsir_lr']
        emmn_lr = params['emmn_lr']

        # Get image configuration
        img_shape = self.config_manager.get_image_size()
        num_channels = self.config_manager.get_img_channels()

        try:
            print("Training UNSIR noise...")
            # Create and train UNSIR noise for forget class
            forget_class_label = self.forget_class
            noise = UNSIR_noise(noise_batch_size, num_channels, img_shape, img_shape).to(self.device)
            noise = self.UNSIR_noise_train(
                noise, model, forget_class_label, unsir_epochs, noise_batch_size, lr=unsir_lr
            )

            print("Training EMMN noises...")
            # Create and train EMMN noises for retain classes
            retain_classes = [i for i in range(self.num_classes) if i != self.forget_class]
            retain_noises = {
                i_: UNSIR_noise(noise_batch_size, num_channels, img_shape, img_shape).to(self.device)
                for i_ in retain_classes
            }

            for i_, n_ in retain_noises.items():
                retain_noises[i_] = self.emmn_noise_train(n_, model, i_, emmn_epochs, noise_batch_size, lr=emmn_lr)

            # Combine all noises
            noises = retain_noises.copy()
            noises[self.forget_class] = noise

            # Create noisy data loader
            noisy_loader = self.emmc_create_noisy_loader(
                noises,
                batch_size=noise_batch_size,
                num_noise_batches=num_noise_batches
            )

            # Convert to DeviceDataLoader
            noisy_loader = DeviceDataLoader(noisy_loader, self.device)

            print("Fine-tuning model...")
            # Fine-tune the model with noisy data
            self.fit(
                epochs=finetune_epochs,
                lr=learning_rate,
                model=model,
                train_loader=noisy_loader,
                opt_func=torch.optim.Adam
            )

            return model

        except Exception as e:
            print(f"Error in EMMN unlearning: {e}")
            raise

    def run_single_experiment(self, params, experiment_idx, total_experiments):
        """Run a single experiment with the basic EMMN implementation"""
        print(f"\nRunning experiment {experiment_idx + 1}/{total_experiments}")
        print(f"Parameters: {params}")

        start_time = time.time()

        try:
            # Create fresh model copy
            model = deepcopy(self.original_model)

            # Run basic EMMN unlearning
            unlearned_model = self.run_basic_emmn_unlearning(model, params)

            # Save model
            unlearned_model_path = os.path.join(
                self.config_manager.get_models_path_unlearn(),
                f"unlearned_{self.model_name}_{self.dataset_name}_forget_{self.forget_class}_exp_{experiment_idx}.pth"
            )
            torch.save(unlearned_model.state_dict(), unlearned_model_path)

            # Evaluate across all clients
            all_metrics = []
            for client_id in range(self.n_clients):
                try:
                    self.load_model_and_data(client_id=client_id)
                    metrics = self._evaluate_model(unlearned_model_path)
                    all_metrics.append(metrics)
                except Exception as e:
                    print(f"Error evaluating client {client_id}: {e}")
                    continue

            if not all_metrics:
                raise Exception("No successful client evaluations")

            # Aggregate results
            metrics = aggregate_results(all_metrics, len(all_metrics))

            # Extract mean values for main metrics
            main_metrics = {}
            for key, value_dict in metrics.items():
                if isinstance(value_dict, dict) and 'mean' in value_dict:
                    main_metrics[key] = value_dict['mean']
                else:
                    main_metrics[key] = value_dict

            training_time = time.time() - start_time

            # Store results
            result = {
                'experiment_idx': experiment_idx,
                'method': 'basic_emmn',
                'training_time': training_time,
                'success': True,
                **params,
                **main_metrics
            }

            self.results.append(result)
            self._save_intermediate_results(result, experiment_idx)

            # Print key metrics
            forget_acc = main_metrics.get('forget_accuracy', 'N/A')
            retain_acc = main_metrics.get('retain_accuracy', 'N/A')
            mia = main_metrics.get('mia', 'N/A')

            print(f"✓ Experiment {experiment_idx + 1} completed in {training_time:.2f}s")
            print(f"  Forget Accuracy: {forget_acc:.4f}" if forget_acc != 'N/A' else f"  Forget Accuracy: {forget_acc}")
            print(f"  Retain Accuracy: {retain_acc:.4f}" if retain_acc != 'N/A' else f"  Retain Accuracy: {retain_acc}")
            print(f"  MIA: {mia:.4f}" if mia != 'N/A' else f"  MIA: {mia}")

            return result

        except Exception as e:
            print(f"✗ Experiment {experiment_idx + 1} failed: {str(e)}")
            error_result = {
                'experiment_idx': experiment_idx,
                'method': 'basic_emmn',
                'error': str(e),
                'training_time': time.time() - start_time,
                'success': False,
                **params
            }
            self.results.append(error_result)
            self._save_intermediate_results(error_result, experiment_idx)
            return error_result

    def _evaluate_model(self, unlearned_model_path):
        """Evaluate model with better error handling"""
        try:
            metrics = get_metrics(
                original_model_path=os.path.join(
                    self.config_manager.get_models_path_original(),
                    f"model_{self.model_name.upper()}_{self.n_clients}_{self.dataset_name}.pth"
                ),
                unlearned_model_path=unlearned_model_path,
                scratch_model_path=os.path.join(
                    self.config_manager.get_models_path_original(),
                    f"scratch_model_{str(self.model_name).upper()}_{self.n_clients}_{self.dataset_name}.pth"
                ),
                forget_test_loader=self.forget_test_loader,
                retain_test_loader=self.retain_test_loader,
                device=self.device,
                forget_train_loader=self.forget_train_loader,
                retain_train_loader=self.retain_train_loader,
                test_loader=self.test_loaders,
                alpha=self.config_manager.get_alpha_unlearn_eval(),
            )

            processed_metrics = {
                "mia": metrics.get("mia", 0.0),
                "anamnesis_index": metrics.get("anamnesis_index", 0.0),
                "forget_accuracy": metrics.get("forget_acc", 0.0),
                "retain_accuracy": metrics.get("retain_acc", 0.0),
                "original_forget_acc": metrics.get("original_forget_acc", 0.0),
                "original_retain_acc": metrics.get("original_retain_acc", 0.0),
                "original_accuracy": metrics.get("original_accuracy", 0.0),
            }

            return processed_metrics

        except Exception as e:
            print(f"Error in model evaluation: {e}")
            # Return default metrics to prevent crash
            return {
                "mia": 1.0,
                "anamnesis_index": 1.0,
                "forget_accuracy": 1.0,
                "retain_accuracy": 0.0,
                "original_forget_acc": 1.0,
                "original_retain_acc": 1.0,
                "original_accuracy": 1.0,
            }

    def _save_intermediate_results(self, result, experiment_idx):
        """Save results with better formatting"""
        try:
            # Individual result
            result_file = os.path.join(self.results_dir, f'result_{experiment_idx:04d}.json')
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)

            # All results
            results_file = os.path.join(self.results_dir, 'all_results.json')
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)

            # CSV for analysis
            df = pd.DataFrame(self.results)
            csv_file = os.path.join(self.results_dir, 'results.csv')
            df.to_csv(csv_file, index=False)

        except Exception as e:
            print(f"Error saving results: {e}")

    def run_grid_search(self, max_experiments=None):
        """Run the grid search with basic EMMN"""
        print("🚀 Starting Basic EMMN Grid Search Experiment")
        print(f"Experiment ID: {self.experiment_id}")
        print(f"Dataset: {self.dataset_name}, Model: {self.model_name}")
        print(f"Forget Class: {self.forget_class}, Device: {self.device}")

        # Generate parameter combinations
        param_combinations = self.define_parameter_grid()

        if max_experiments:
            param_combinations = param_combinations[:max_experiments]
            print(f"Limited to {max_experiments} experiments")

        total_experiments = len(param_combinations)
        print(f"Running {total_experiments} experiments")

        # Run experiments
        successful_experiments = 0
        for idx, params in enumerate(param_combinations):
            result = self.run_single_experiment(params, idx, total_experiments)
            if result.get('success', False):
                successful_experiments += 1

            # Progress update
            if (idx + 1) % 10 == 0:
                success_rate = (successful_experiments / (idx + 1)) * 100
                print(f"\n📊 Progress: {idx + 1}/{total_experiments} ({success_rate:.1f}% success rate)")

        # Generate final analysis
        print("\n📈 Generating final analysis...")
        self.analyze_results()

        print(f"\n🎉 Grid search completed!")
        print(f"Total experiments: {total_experiments}")
        print(f"Successful experiments: {successful_experiments}")
        print(f"Results saved in: {self.results_dir}")

    def analyze_results(self):
        """Basic results analysis"""
        if not self.results:
            print("No results to analyze")
            return

        df = pd.DataFrame(self.results)
        successful_df = df[df.get('success', False) == True].copy()

        if successful_df.empty:
            print("No successful experiments to analyze")
            return

        # Find best experiments
        best_experiments = self._find_best_experiments(successful_df)

        # Save analysis summary
        self._save_analysis_summary(successful_df, best_experiments)

        print(f"Analysis completed. Found {len(successful_df)} successful experiments.")

    def _find_best_experiments(self, df):
        """Find the best experiments based on different criteria"""
        best_experiments = {}

        # Best forget accuracy (lowest)
        if 'forget_accuracy' in df.columns:
            best_forget_idx = df['forget_accuracy'].idxmin()
            best_experiments['best_forget'] = df.loc[best_forget_idx].to_dict()

        # Best retain accuracy (highest)
        if 'retain_accuracy' in df.columns:
            best_retain_idx = df['retain_accuracy'].idxmax()
            best_experiments['best_retain'] = df.loc[best_retain_idx].to_dict()

        # Best MIA (lowest)
        if 'mia' in df.columns:
            best_mia_idx = df['mia'].idxmin()
            best_experiments['best_mia'] = df.loc[best_mia_idx].to_dict()

        # Fastest training
        if 'training_time' in df.columns:
            fastest_idx = df['training_time'].idxmin()
            best_experiments['fastest'] = df.loc[fastest_idx].to_dict()

        return best_experiments

    def _save_analysis_summary(self, df, best_experiments):
        """Save a comprehensive analysis summary"""
        summary = {
            'experiment_overview': {
                'total_experiments': len(df),
                'successful_experiments': len(df[~df.get('error', pd.Series()).notna()]),
                'failed_experiments': len(df[df.get('error', pd.Series()).notna()]) if 'error' in df.columns else 0
            },
            'performance_statistics': {},
            'best_experiments': best_experiments,
        }

        # Performance statistics
        for metric in ['forget_accuracy', 'retain_accuracy', 'training_time', 'mia', 'anamnesis_index']:
            if metric in df.columns:
                summary['performance_statistics'][metric] = {
                    'mean': float(df[metric].mean()),
                    'std': float(df[metric].std()),
                    'min': float(df[metric].min()),
                    'max': float(df[metric].max()),
                    'median': float(df[metric].median())
                }

        # Save summary
        summary_file = os.path.join(self.results_dir, 'analysis_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Create a readable report
        self._create_readable_report(summary)

    def _create_readable_report(self, summary):
        """Create a human-readable report"""
        report_file = os.path.join(self.results_dir, 'experiment_report.txt')

        with open(report_file, 'w') as f:
            f.write(f"EMMN Unlearning Grid Search Experiment Report\n")
            f.write(f"=" * 60 + "\n\n")
            f.write(f"Experiment ID: {self.experiment_id}\n")
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Forget Class: {self.forget_class}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Overview
            f.write("EXPERIMENT OVERVIEW\n")
            f.write("-" * 20 + "\n")
            overview = summary['experiment_overview']
            f.write(f"Total experiments: {overview['total_experiments']}\n")
            f.write(f"Successful experiments: {overview['successful_experiments']}\n")
            f.write(f"Failed experiments: {overview['failed_experiments']}\n\n")

            # Best experiments
            f.write("BEST EXPERIMENTS\n")
            f.write("-" * 20 + "\n")
            for category, experiment in summary['best_experiments'].items():
                f.write(f"\nBest {category}:\n")
                f.write(f"  Learning Rate: {experiment.get('learning_rate', 'N/A')}\n")
                f.write(f"  UNSIR Epochs: {experiment.get('unsir_epochs', 'N/A')}\n")
                f.write(f"  EMMN Epochs: {experiment.get('emmn_epochs', 'N/A')}\n")
                f.write(f"  Finetune Epochs: {experiment.get('finetune_epochs', 'N/A')}\n")
                f.write(f"  Noise Batch Size: {experiment.get('noise_batch_size', 'N/A')}\n")
                f.write(f"  Forget Accuracy: {experiment.get('forget_accuracy', 'N/A'):.4f}\n")
                f.write(f"  Retain Accuracy: {experiment.get('retain_accuracy', 'N/A'):.4f}\n")
                f.write(f"  Training Time: {experiment.get('training_time', 'N/A'):.2f}s\n")

            # Performance statistics
            f.write("\nPERFORMANCE STATISTICS\n")
            f.write("-" * 20 + "\n")
            for metric, stats in summary['performance_statistics'].items():
                f.write(f"\n{metric}:\n")
                f.write(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                f.write(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n")
                f.write(f"  Median: {stats['median']:.4f}\n")


def main():
    # ====== 1. Setup logging ======
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # ====== 2. Load configuration ======
    try:
        config_manager = ConfigurationManager()
        print("Configuration loaded successfully.")
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
            print(f"Using CUDA device: {gpu_name}")
        else:
            print("Using CPU")
    except Exception as e:
        print(f"Failed to set seed / CUDA options cleanly: {e}")

    # ====== 5. Create experiment runner ======
    try:
        runner = EMMNGridSearch(config_manager)
        print("EMMNGridSearch initialized.")
    except Exception as e:
        print(f"Failed to create EMMNGridSearch: {e}")
        return

    # ====== 6. Run grid search ======
    try:
        runner.run_grid_search(max_experiments=100)
    except KeyboardInterrupt:
        print("Interrupted by user. Saving partial results...")
        try:
            # force a final save if possible
            runner._save_intermediate_results(
                {'experiment_idx': -1, 'method': 'basic_emmn', 'success': False, 'error': 'Interrupted'},
                experiment_idx=len(runner.results)
            )
        except Exception as ie:
            print(f"Failed to save partial results: {ie}")
    except Exception as e:
        print(f"Grid search failed: {e}")

    print("Done.")

if __name__ == "__main__":
    main()
