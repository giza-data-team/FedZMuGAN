import os
import random
import torch
import numpy as np
from client_utils.general_utils import set_seed

set_seed()

import time
import json
import sys
from data_split import DataSplitter
import argparse
import logging

# Flower imports
import flwr as fl
from flwr.simulation import run_simulation
from flwr.common import Context

from log_exception import log_exception
from config_manager import ConfigurationManager

# Import your client and server components
from client import FlowerClient
from server_utils.custom_strategy import CustomStrategy

config_manager = ConfigurationManager()

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set the environment variable
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

train_size = config_manager.get_train_split()
eval_size = 1 - train_size

# Change to the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

parser = argparse.ArgumentParser(description="Process named arguments.")

n_clients = config_manager.get_n_clients()
logging.info("num of clients is %s", n_clients)
dataset_name = config_manager.get_dataset_name()
homogenous = config_manager.get_homogeneous()
experiment_info = {
    "dataset_name": dataset_name,
    "n_clients": n_clients,
    "type": "Run file",
}


def client_fn(context: Context) -> fl.client.Client:
    """Create a Flower client representing a single organization."""

    # Get client-specific parameters from context
    partition_id = context.node_config["partition-id"]

    # Get configuration
    target_class = config_manager.get_forget_class()
    n_epochs = config_manager.get_epochs_original()
    homogeneous = config_manager.get_homogeneous()
    lr = config_manager.get_lr_original()

    # Create and return client
    return FlowerClient(
        target_class=target_class,
        client_id=str(partition_id),
        local_epochs=n_epochs,
        homogeneous=homogeneous,
        lr=lr,
    )


def server_fn(context: Context) -> fl.server.ServerAppComponents:
    """Create server components."""

    # Get configuration
    n_rounds = config_manager.get_n_rounds()
    model_name = config_manager.get_model_name()

    # Create strategy
    strategy = CustomStrategy(
        min_fit_clients=n_clients,
        min_evaluate_clients=n_clients,
        min_available_clients=n_clients,
        rounds=n_rounds,
        dataset_name=dataset_name,
        model_name=model_name,
    )

    # Create server config
    config = fl.server.ServerConfig(num_rounds=n_rounds)

    return fl.server.ServerAppComponents(
        strategy=strategy,
        config=config,
    )


try:
    # data_splitter = DataSplitter(
    #     num_clients=n_clients,
    #     homogeneous=homogenous,
    #     min_instances_per_client=1000,
    #     dataset_name=dataset_name,
    # )
    # data_splitter.generate_and_split_data()

    logger.info("Starting Flower simulation...")

    # Create ClientApp and ServerApp
    client_app = fl.client.ClientApp(client_fn=client_fn)
    server_app = fl.server.ServerApp(server_fn=server_fn)

    # Specify the resources each client needs
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

    # If CUDA is available, allocate GPU resources
    if torch.cuda.is_available():
        backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}

    # Run the simulation
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=n_clients,
        backend_config=backend_config,
    )

    logger.info("Simulation completed successfully!")

except Exception as e:
    error_message = f"Error during run.py: {e}"
    logging.error(error_message)
    log_exception(experiment_info, error_message, n_clients=n_clients)
    raise
