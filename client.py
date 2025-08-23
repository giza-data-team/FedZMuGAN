import os
import random
import torch
import numpy as np
from client_utils.general_utils import set_seed

set_seed()

import argparse
import signal

import torch
import torch.optim
import flwr as fl

from client_utils.load_client_data import LoadClientData
from client_utils.train_model import ModelTrainer
from client_utils.test_model import ValidateData
from client_utils.models.model_factory import ModelFactory
from client_utils.weights_controller import WeightsController
from datetime import datetime
from file_controller import FileController
from flwr.common import (
    EvaluateRes,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from config_manager import ConfigurationManager
from Machine_Unlearning.Unlearning.metrics.all_metrics import get_metrics
import logging
from Machine_Unlearning.Unlearning.train.lipschitz_unlearning import LipschitzUnlearning


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FlowerClient(fl.client.Client):
    def __init__(self, target_class, client_id, local_epochs, homogeneous, lr):
        self.client_id = client_id
        self.config_manager = ConfigurationManager()
        self.model_factory = ModelFactory()
        self.net = self.model_factory.create_model()
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        self.batch_size = self.config_manager.get_batch_size_original()
        self.train_loader, self.val_loader, self.test_loaders = LoadClientData(
            client_id=self.client_id
        ).get_normal_loaders(self.batch_size)
        self.target_class = target_class
        self.model_trainer_original = ModelTrainer(
            model=self.net,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device=self.device,
        )
        

        self.weight_controller = WeightsController()
        self.homogeneous = homogeneous
        self.model_name = self.net.__class__.__name__
        self.info_tracker = {}
        self.save_results = FileController(
            file_name=f"client_{self.model_name}_{self.client_id}_results.csv"
        )
        self.lr = lr

        self.dataset_name = self.config_manager.get_dataset_name()
        self.mu = None

        # loaders
        self.batch_size_unlearn = self.config_manager.get_batch_size_unlearn()
        self.forget_train_loader, self.retain_train_loader = LoadClientData(
            client_id=self.client_id
        ).get_forget_retain_loaders(
            forget_class=self.target_class,
            split="train",
            batch_size=self.batch_size_unlearn,
        )

        self.forget_val_loader, self.retain_val_loader = LoadClientData(
            client_id=self.client_id
        ).get_forget_retain_loaders(
            forget_class=self.target_class,
            split="val",
            batch_size=self.batch_size_unlearn,
        )

        self.forget_test_loader, self.retain_test_loader = LoadClientData(
            client_id=self.client_id
        ).get_forget_retain_loaders(
            forget_class=target_class, split="test", batch_size=self.batch_size_unlearn
        )

        self.model_trainer_scratch = ModelTrainer(
            model=self.net,
            train_loader=self.retain_train_loader,
            val_loader=self.retain_val_loader,
            device=self.device,
        )
        self.validate_data_original = ValidateData(
            validateloader=self.test_loaders,
            device=self.device
        )
        
        self.validate_data_scratch = ValidateData(
            validateloader=self.retain_test_loader,
            device=self.device
        )
       

    def fit(self, parameters):
        configs = parameters.config
        self.training_status = configs["training_status"]
        unlearning_method = configs["unlearning_method"]

        if self.training_status == "unlearning":
            print(f"\n===== Processing unlearning phase with {unlearning_method.upper()} =====")
            weights = parameters_to_ndarrays(parameters.parameters)
            if weights:
                self.weight_controller.set_weights(self.net, weights)

            # Handle different unlearning methods
            if unlearning_method == "lipschitz":
                print(f"\n===== Client {self.client_id} performing Lipschitz unlearning =====")
                start_time = datetime.now()
 
                # Get lipschitz parameters from config
                lipschitz_lr = self.config_manager.get_lr_lipschitz() 
                lipschitz_noise_std = self.config_manager.get_noise_std_lipschitz() 
                
                # Perform lipschitz unlearning
                lipschitz_unlearner = LipschitzUnlearning(
                    model=self.net,
                    device=self.device,
                    forget_dataloader=self.forget_train_loader,
                    opt_func=torch.optim.Adam,
                    learning_rate=lipschitz_lr,
                    noise_std=lipschitz_noise_std
                )
                
                # Perform unlearning
                unlearned_model = lipschitz_unlearner.lipschitz_unlearn()
                
                end_time = datetime.now()
                time_difference = abs(end_time - start_time)
                time_difference_in_seconds = time_difference.total_seconds()
                
                print(f"Client {self.client_id} completed Lipschitz unlearning in {time_difference_in_seconds:.2f} seconds")
                
                # Return the unlearned model parameters
                unlearned_parameters = ndarrays_to_parameters([
                    param.cpu().numpy() for param in unlearned_model.state_dict().values()
                ])
                
                return FitRes(
                    parameters=unlearned_parameters,
                    num_examples=len(self.forget_train_loader.dataset),
                    metrics={
                        "status": f"{unlearning_method}_unlearning_completed",
                        "unlearning_time": time_difference_in_seconds,
                        "client_id": self.client_id
                    },
                    status=Status(code=fl.common.Code.OK, message="Lipschitz unlearning completed"),
                )
            else:
                # For both zMuGAN and EMMN, unlearning is now done on server side
                # Clients just receive the aggregated parameters
                return FitRes(
                    parameters=parameters.parameters,
                    num_examples=len(self.train_loader.dataset),
                    metrics={"status": f"{unlearning_method}_unlearning"},
                    status=Status(code=fl.common.Code.OK, message="Success"),
                )

        self.info_tracker = {}
        self.info_tracker["dataset_name"] = self.dataset_name
        self.info_tracker["round_number"] = configs["server_round"]

        # Handle scratch training
        print(f"\n============= training_status:{self.training_status} ===========")

        if self.training_status == "scratch_training" or self.training_status == "learning" :
            print(f"\n ====== client {self.client_id} starts {self.training_status}")
            start_time = datetime.now()
            parameters = parameters.parameters
            parameters = parameters_to_ndarrays(parameters=parameters)
            if parameters:
                self.weight_controller.set_weights(self.net, parameters)

            status = fl.common.Status(code=fl.common.Code.OK, message="Success")
            if self.training_status == "scratch_training":
                train_loss, train_accuracy, val_loss, val_accuracy = (
                    self.model_trainer_scratch.train(self.net)
                )
            else:
                train_loss, train_accuracy, val_loss, val_accuracy = (
                    self.model_trainer_original.train(self.net)
                )

            metrics = {}
            metrics["train_loss"] = train_loss
            metrics["train_acc"] = train_accuracy
            if self.training_status == "scratch_training":
                metrics["train_num_examples"] = len(self.retain_train_loader.dataset)
            else:
                metrics["train_num_examples"] = len(self.train_loader.dataset)

            metrics["val_loss"] = val_loss
            metrics["val_acc"] = val_accuracy
            if self.training_status == "scratch_training":
                metrics["val_num_examples"] = len(self.retain_val_loader.dataset)
            else:
                metrics["val_num_examples"] = len(self.val_loader.dataset)
            status = fl.common.Status(code=fl.common.Code.OK, message="done")
            weights = self.weight_controller.get_weights(self.net)

            parameters = ndarrays_to_parameters(ndarrays=weights)
            end_time = datetime.now()
            time_difference = abs(end_time - start_time)
            time_difference_in_seconds = time_difference.total_seconds()

            self.info_tracker["training_time"] = time_difference_in_seconds
            self.info_tracker["train_loss"] = train_loss
            self.info_tracker["train_acc"] = train_accuracy
            self.info_tracker["val_loss"] = val_loss
            self.info_tracker["val_acc"] = val_accuracy

            return FitRes(
                parameters=parameters,
                num_examples=len(self.retain_train_loader.dataset) if self.training_status == "scratch_training" else len(self.train_loader.dataset),
                metrics=metrics,
                status=status,
            )

    def evaluate(self, parameters):
        configs = parameters.config
        server_round = configs["server_round"]
        self.training_status = configs["training_status"]
        unlearning_method = configs["unlearning_method"]
        metrics = {}
        # Handle scratch evaluation
        if self.training_status == "unlearning" :
            print(f"\n ====== client {self.client_id} evaluates {unlearning_method.upper()} unlearning")
            start_time = datetime.now()
            weights = parameters.parameters
            weights = parameters_to_ndarrays(parameters=weights)
            if weights:
                self.weight_controller.set_weights(self.net, weights)

            status = Status(code=fl.common.Code.OK, message="done")
            end_time = datetime.now()

            # Get the correct unlearned model path based on unlearning method
            unlearned_model_path = os.path.join(
            self.config_manager.get_models_path_unlearn(),
            f"unlearned_{self.model_name}_{self.dataset_name}_forget_class_{self.config_manager.get_forget_class()}_{unlearning_method}.pth",
            )

            # Use the same evaluation approach for all methods
            final_metrics = get_metrics(
                original_model_path=os.path.join(
                    self.config_manager.get_models_path_original(), 
                    f"{str(self.model_name).lower()}_{self.dataset_name}.pth"
                ),
                unlearned_model_path=unlearned_model_path,
                scratch_model_path=os.path.join(
                    self.config_manager.get_models_path_original(),
                    f"{str(self.model_name).lower()}_{self.dataset_name}_scratch.pth"
                ),
                forget_test_loader=self.forget_test_loader,
                retain_test_loader=self.retain_test_loader,
                device=self.device,
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

            metrics["forget_acc_num_examples"] = len(
                self.forget_test_loader.dataset
            )
            metrics["retain_acc_num_examples"] = len(
                self.retain_test_loader.dataset
            )
            metrics["mia_num_examples"] = (
                len(self.retain_train_loader.dataset)
                + len(self.forget_train_loader.dataset)
                + len(self.test_loaders.dataset)
            )
            metrics["anamnesis_index_num_example"] = len(self.forget_train_loader)
            metrics["original_accuracy_num_examples"] = len(self.test_loaders.dataset)

            self.info_tracker.update(
                {
                    "mia": final_metrics["mia"],
                    "anamnesis_index": final_metrics["anamnesis_index"],
                    "forget_accuracy": final_metrics["forget_acc"],
                    "retain_accuracy": final_metrics["retain_acc"],
                    "original_forget_accuracy": final_metrics["original_forget_acc"],
                    "original_retain_accuracy": final_metrics["original_retain_acc"],
                    "original_accuracy": final_metrics["original_accuracy"],
                }
            )

            self.save_results.save(self.info_tracker)
            time_difference = abs(end_time - start_time)
            time_difference_in_seconds = time_difference.total_seconds()
            self.info_tracker["test_time"] = time_difference_in_seconds
            self.info_tracker["homogeneous"] = self.homogeneous

            return EvaluateRes(
                loss=0,
                num_examples=len(self.retain_test_loader.dataset),
                metrics=metrics,
                status=status,
            )
        
        else:
            print(f"\n ====== client {self.client_id} evaluats {self.training_status}")
            start_time = datetime.now()
            weights = parameters.parameters
            weights = parameters_to_ndarrays(parameters=weights)
            if weights:
                self.weight_controller.set_weights(self.net, weights)
            
            if self.training_status == "scratch_training":
                loss, val_metric = self.validate_data_scratch.validate(net=self.net,validateloader=self.retain_test_loader, target_class=self.target_class)
            else:
                loss, val_metric = self.validate_data_original.validate(net=self.net,validateloader=self.test_loaders, target_class=self.target_class)
        
            accuracy = val_metric["overall_accuracy"]

            metrics["test_loss"] = loss
            metrics["test_acc"] = accuracy

            self.info_tracker["test_loss"] = loss
            self.info_tracker["test_acc"] = accuracy

            status = Status(code=fl.common.Code.OK, message="done")
            end_time = datetime.now()
            time_difference = abs(end_time - start_time)
            time_difference_in_seconds = time_difference.total_seconds()
            self.info_tracker["test_time"] = time_difference_in_seconds
            self.info_tracker["homogeneous"] = self.homogeneous
            self.save_results.save(self.info_tracker)

            return EvaluateRes(
                loss=loss,
                num_examples=len(self.retain_test_loader.dataset) if self.training_status == "scratch_training" else len(self.test_loaders.dataset),
                metrics=metrics,
                status=status,
            )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process named arguments.")
    # Create an instance of your custom strategy
    parser.add_argument("--client_id", type=str, required=True, help="client id")

    args = parser.parse_args()
    config_manager = ConfigurationManager()
    client_id = args.client_id  # No need for int conversion since we specified type=int
    target_class = config_manager.get_forget_class()
    n_epochs = config_manager.get_epochs_original()
    port = config_manager.get_port()
    homogeneous = config_manager.get_homogeneous()
    lr = config_manager.get_lr_original()

    try:
        client = FlowerClient(
            target_class=target_class,
            client_id=client_id,
            local_epochs=n_epochs,
            homogeneous=homogeneous,
            lr=lr,
        )
        fl.client.start_client(server_address=f"localhost:{port}", client=client)
    except Exception as e:
        logger.error(f"Error in client execution: {str(e)}")
        os.kill(os.getpid(), signal.SIGTERM)