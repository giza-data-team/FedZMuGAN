# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Federated Averaging (FedAvg) [McMahan et al., 2016] strategy.

Paper: arxiv.org/abs/1602.05629
"""
import time
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
from datetime import datetime
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import (
    aggregate,
    aggregate_inplace,
    weighted_loss_avg,
)
from flwr.server.strategy import Strategy
from file_controller import FileController
import numpy as np
from config_manager import ConfigurationManager
from client_utils.weights_controller import WeightsController
from Machine_Unlearning.zMuGAN.main import ZMuGANTrainer
from Machine_Unlearning.Unlearning.train.main import UnlearningProcess
from Machine_Unlearning.Unlearning.train.emmn_unlearning import EMMNUnlearning
from client_utils.models.model_factory import ModelFactory
from config_manager import ConfigurationManager
import os
import torch
from logging import WARNING, INFO
from server_utils.helper import multi_weighted_avg


WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


# pylint: disable=line-too-long
class CustomStrategy(Strategy):
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = True,
        rounds: int = 5,
        dataset_name: str = "",
        homogeneous: bool = True,
        model_name:str=None,
    ) -> None:
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.inplace = inplace
        self.rounds = rounds
        self.best_weights = None
        self.best_loss = float('inf')
        self.last_weights = None
        self.dataset_name = dataset_name
        self.model_name = model_name
        # self.current_lr = None
        # self.aggregated_lr = None
        self.weight_controller = WeightsController()
        self.config_manager = ConfigurationManager()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_results = FileController(
            file_name=f"server_{self.model_name}_{self.min_fit_clients}_{self.dataset_name}.csv",
            path="server_results",
            client=False,
        )
        self.info_tracker = {}
        self.loss_flag = 0
        self.return_best_weights = False
        self.homogeneous = homogeneous
        self.training_status = "unlearning"
        self.unlearning_method = self.config_manager.get_unlearning_method()
        
        # Lipschitz-specific tracking variables
        self.lipschitz_round = 0
        self.best_utility_score = -float('inf')  # retain_acc + (100-forget_acc)
        self.best_lipschitz_weights = None
        self.lipschitz_no_improvement_count = 0
        self.lipschitz_max_no_improvement = 3  # Stop after 3 rounds of no improvement
        self.model_path_unlearn = self.config_manager.get_models_path_unlearn()
        self.unlearned_model_path = os.path.join(
            self.model_path_unlearn,
            f"unlearned_{self.model_name}_{self.dataset_name}_forget_class_{self.config_manager.get_forget_class()}.pth"
        ) if self.unlearning_method == "zmugan" else os.path.join(
            self.model_path_unlearn,
            f"unlearned_{self.model_name}_{self.dataset_name}_forget_class_{self.config_manager.get_forget_class()}_{self.unlearning_method}.pth"
        )  
        """Federated Averaging strategy.
    
        Implementation based on https://arxiv.org/abs/1602.05629
    
        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. In case `min_fit_clients`
            is larger than `fraction_fit * available_clients`, `min_fit_clients`
            will still be sampled. Defaults to 1.0.
        fraction_evaluate : float, optional
            Fraction of clients used during validation. In case `min_evaluate_clients`
            is larger than `fraction_evaluate * available_clients`,
            `min_evaluate_clients` will still be sampled. Defaults to 1.0.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_evaluate_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]],Optional[Tuple[float, Dict[str, Scalar]]]]]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        inplace : bool (default: True)
            Enable (True) or disable (False) in-place aggregation of model updates.
        """
        # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        config["server_round"] = server_round
        config["training_status"] = self.training_status
        config["unlearning_method"] = self.unlearning_method

        fit_ins = FitIns(parameters, config)
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        num_clients_connected = client_manager.num_available()
        while num_clients_connected < min_num_clients:
            num_clients_connected = client_manager.num_available()
            if num_clients_connected == min_num_clients:
                time.sleep(5)
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        config["server_round"] = server_round
        config["training_status"] = self.training_status
        config["unlearning_method"] = self.unlearning_method
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        
        self.info_tracker = {}
        self.info_tracker["dataset_name"] = self.dataset_name
        self.info_tracker["model_name"] = self.model_name
        self.info_tracker["round_number"] = server_round
        start_time = datetime.now()
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # metrics_aggregated = {}

        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        self.last_weights = parameters_aggregated
        log(INFO, f"return best weights{self.return_best_weights}")
        if self.return_best_weights:
            print("\n===== best weights are returned ====")
            parameters_aggregated = self.best_weights

            if self.training_status == "learning":
                self.weight_controller.save_weights(
                    model_pram=parameters_to_ndarrays(self.best_weights),
                    model_id=self.model_name,
                    dataset_name=self.dataset_name,
                )
            elif self.training_status == "scratch_training":
                self.weight_controller.save_weights(
                    model_pram=parameters_to_ndarrays(self.best_weights),
                    model_id=f"{self.model_name}_scratch",
                    dataset_name=self.dataset_name,
                )

        client_results = [res.metrics for _, res in results]

        if self.training_status == "unlearning":
            if self.unlearning_method == "zmugan":
                print("\n=== Starting zMuGAN Training and Unlearning Process ===")
                # Train zMuGAN
                config_manager = ConfigurationManager()
                n_generators = config_manager.get_n_generators_zmugan()
                for i in range(n_generators):
                    print(f"\n********** Training Generator {i+1} of {n_generators} **********")
                    trainer = ZMuGANTrainer(generator_index=i+1)
                    trainer.run(generator_index=i+1)

                # Perform unlearning
                print(f"\n********** Starting zMuGAN Unlearning **********")
                unlearning_process = UnlearningProcess()
                unlearning_process.load_pretrained_model()
                unlearning_process.perform_unlearning()
                
                print("zMuGAN unlearning process completed successfully")
                self.unlearned_model = self.weight_controller.load_model(self.unlearned_model_path, self.device)

                

            elif self.unlearning_method == "emmn":
                print("\n=== Starting EMMN Unlearning Process (Server-side) ===")
                
                # Call EMMN - it will load everything from config and model from disk
                emmn_unlearning = EMMNUnlearning(device=self.device)
                unlearned_model = emmn_unlearning.emmn_fit()
                self.unlearned_model = unlearned_model
                
                print("EMMN unlearning completed successfully")
                print("Unlearned model saved by EMMN")
                
            elif self.unlearning_method == "lipschitz":
                print(f"\n=== Lipschitz Unlearning Round {server_round} (Client-side) ===")
                # For lipschitz, clients perform unlearning locally and return their results
                # We continue with normal federated aggregation and track metrics for early stopping
                self.lipschitz_round = server_round
                self.info_tracker.update({
                    "unlearning_round": server_round,
                    "unlearning_method": self.unlearning_method,
                    "lipschitz_round": self.lipschitz_round
                })
                
                # Save the aggregated parameters for lipschitz unlearning
                # Create a model instance and load the aggregated weights
                temp_model = ModelFactory.create_model().to(self.device)
                aggregated_arrays = parameters_to_ndarrays(parameters_aggregated)
                self.weight_controller.set_weights(temp_model, aggregated_arrays)
                
                # Save as proper PyTorch state_dict
                torch.save(temp_model.state_dict(), self.unlearned_model_path)
                self.unlearned_model = temp_model
                
                end_time = datetime.now()
                time_difference = abs(end_time - start_time)
                time_difference_in_seconds = time_difference.total_seconds()
                self.info_tracker["weights_agg_time"] = time_difference_in_seconds
                
                return parameters_aggregated, self.info_tracker
                
            else:
                raise ValueError(f"Unknown unlearning method: {self.unlearning_method}")
            
            # For zmugan and emmn: Load the unlearned model and use it as parameters
            parameters_aggregated = ndarrays_to_parameters(
                [
                    param.cpu().numpy()
                    for param in self.unlearned_model.state_dict().values()
                ]
            )
            # Update info tracker
            self.info_tracker.update({
                "unlearning_round": server_round,
                "unlearning_method": self.unlearning_method
            })

            end_time = datetime.now()
            time_difference = abs(end_time - start_time)
            time_difference_in_seconds = time_difference.total_seconds()
            self.info_tracker["unlearning_time"] = time_difference_in_seconds
            
            # Verify the state change
            print(f"\n=== Completed {self.unlearning_method.upper()} unlearning ===")

            return parameters_aggregated, self.info_tracker

        if self.training_status == "start_scratch_training":
            # Initialize scratch model for the first time
            self.scratch_model = ModelFactory.create_model().to(self.device)

            # Initialize with random weights
            parameters_aggregated = ndarrays_to_parameters(
                [
                    param.cpu().numpy()
                    for param in self.scratch_model.state_dict().values()
                ]
            )

            # Update training status
            self.training_status = "scratch_training"

        metric_pairs = [
            ("train_num_examples", "train_loss"),
            ("train_num_examples", "train_acc"),
            ("val_num_examples", "val_loss"),
            ("val_num_examples", "val_acc"),
        ]

        if client_results:
            results_aggregated = multi_weighted_avg(client_results, metric_pairs)
        else:
            results_aggregated = {
                "train_loss": 0.0,
                "train_acc": 0.0,
                "val_loss": 0.0,
                "val_acc": 0.0
            }

        if server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")
        else:
            self.info_tracker["train_agg_loss"] = results_aggregated["train_loss"]
            self.info_tracker["train_agg_acc"] = results_aggregated["train_acc"]
            self.info_tracker["val_agg_loss"] = results_aggregated["val_loss"]
            self.info_tracker["val_agg_acc"] = results_aggregated["val_acc"]

        end_time = datetime.now()
        time_difference = abs(end_time - start_time)
        time_difference_in_seconds = time_difference.total_seconds()
        self.info_tracker["weights_agg_time"] = time_difference_in_seconds
        return parameters_aggregated, results_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}     
    
        
        if self.training_status == "learning" or self.training_status == "scratch_training":
            start_time = datetime.now()
        # Aggregate loss
            results_aggregated = {}
            test_agg_loss = weighted_loss_avg(
                [
                    (test_res.num_examples, test_res.metrics["test_loss"])
                    for _, test_res in results
                ]
            )

            test_agg_acc = weighted_loss_avg(
                [
                    (test_res.num_examples, test_res.metrics["test_acc"])
                    for _, test_res in results
                ]
            )

            results_aggregated["test_agg_loss"] = test_agg_loss
            results_aggregated["test_agg_acc"] = test_agg_acc

            self.info_tracker["test_agg_loss"] = test_agg_loss
            self.info_tracker["test_agg_acc"] = test_agg_acc

            print("inside aggregate evaluate")
            print(self.best_loss)
            print(self.loss_flag)
            if test_agg_loss < self.best_loss:
                self.best_loss = test_agg_loss
                self.best_weights = self.last_weights
                self.loss_flag = 0



            else:
                self.loss_flag += 1

            if server_round == 1:  # Only log this warning once
                log(WARNING, "No evaluate_metrics_aggregation_fn provided")

            self.info_tracker["homogeneous"] = self.homogeneous
            end_time = datetime.now()
            time_difference = abs(end_time - start_time)
            time_difference_in_seconds = time_difference.total_seconds()
            self.info_tracker["loss_agg_time"] = time_difference_in_seconds
            self.info_tracker["training_status"] = self.training_status

        if self.training_status == "unlearning":
            client_results = [res.metrics for _, res in results]
            metric_pairs = [
                ("forget_acc_num_examples", "forget_accuracy"),
                ("retain_acc_num_examples", "retain_accuracy"),
                ("mia_num_examples", "mia"),
                ("anamnesis_index_num_example", "anamnesis_index"),
                ("forget_acc_num_examples", "original_forget_acc"),
                ("retain_acc_num_examples", "original_retain_acc"),
                ("original_accuracy_num_examples", "original_accuracy"),
            ]

            results_aggregated = multi_weighted_avg(client_results, metric_pairs)

            self.info_tracker["avg_forget_acc"] = results_aggregated["forget_accuracy"]
            self.info_tracker["avg_retain_acc"] = results_aggregated["retain_accuracy"]
            self.info_tracker["avg_mia"] = results_aggregated["mia"]
            self.info_tracker["avg_anamnesis_index"] = results_aggregated[
                "anamnesis_index"
            ]
            self.info_tracker["avg_original_forget_acc"] = results_aggregated["original_forget_acc"]
            self.info_tracker["avg_original_retain_acc"] = results_aggregated["original_retain_acc"]
            self.info_tracker["avg_original_accuracy"] = results_aggregated["original_accuracy"]
            self.info_tracker["unlearning_method_evaluated"] = self.unlearning_method
            print(f"aggregated forget accuracy: {results_aggregated['forget_accuracy']}")
            print(f"aggregated retain accuracy: {results_aggregated['retain_accuracy']}")
            print(f"aggregated mia: {results_aggregated['mia']}")
            print(f"aggregated anamnesis index: {results_aggregated['anamnesis_index']}")
            print(f"aggregated original forget accuracy: {results_aggregated['original_forget_acc']}")  
            print(f"aggregated original retain accuracy: {results_aggregated['original_retain_acc']}")
            print(f"aggregated original accuracy: {results_aggregated['original_accuracy']}")
            
            # Handle lipschitz early stopping
            if self.unlearning_method == "lipschitz":
                # Calculate the utility score
                current_utility_score = results_aggregated["retain_accuracy"] + (100 - results_aggregated["forget_accuracy"])
                self.info_tracker["utility_score"] = current_utility_score
                
                print(f"Current utility score (retain_acc + (100-forget_acc)): {current_utility_score}")
                print(f"Best utility score so far: {self.best_utility_score}")
                
                # Check if this is the best score
                if current_utility_score > self.best_utility_score:
                    self.best_utility_score = current_utility_score
                    self.best_lipschitz_weights = self.last_weights
                    self.lipschitz_no_improvement_count = 0
                    print(f"NEW BEST utility score: {current_utility_score}")
                else:
                    self.lipschitz_no_improvement_count += 1
                    print(f"No improvement for {self.lipschitz_no_improvement_count} rounds")
                
                self.info_tracker["best_utility_score"] = self.best_utility_score
                self.info_tracker["lipschitz_no_improvement_count"] = self.lipschitz_no_improvement_count
                
                # Check early stopping condition
                if self.lipschitz_no_improvement_count >= self.lipschitz_max_no_improvement:
                    print(f"\n=== LIPSCHITZ EARLY STOPPING ===")
                    print(f"No improvement for {self.lipschitz_max_no_improvement} rounds")
                    print(f"Best utility score: {self.best_utility_score}")
                    print(f"Saving best lipschitz weights...")
                    
                    # Save the best unlearned model
                    if self.best_lipschitz_weights is not None:
                        best_model = ModelFactory.create_model().to(self.device)
                        self.weight_controller.set_weights(best_model, parameters_to_ndarrays(self.best_lipschitz_weights))
                        
                        # Save the best lipschitz model
                        torch.save(best_model.state_dict(), self.unlearned_model_path)
                        print(f"Best Lipschitz model saved to: {self.unlearned_model_path}")
                    
                    self.save_results.save(self.info_tracker)
                    raise Exception(f"Lipschitz unlearning completed at round {server_round} with early stopping")
            
            self.save_results.save(self.info_tracker)
            
            # For non-lipschitz methods, raise exception to complete unlearning
            if self.unlearning_method != "lipschitz":
                raise Exception(f"learning and unlearning completed at round {server_round}")

        self.save_results.save(self.info_tracker)

        if self.loss_flag >= 2:
            if self.return_best_weights == True:
                if self.training_status == "learning":
                    self.training_status = "start_scratch_training"
                    self.best_loss = float("inf")
                    self.loss_flag = 0
                    self.return_best_weights = False

                elif self.training_status == "scratch_training":
                    self.training_status = "unlearning"
                    self.best_loss = float("inf")
                    self.loss_flag = 0
                    self.return_best_weights = False

                    
            else:
                self.return_best_weights = True
                self.best_loss = -1
                

        return test_agg_loss, results_aggregated
