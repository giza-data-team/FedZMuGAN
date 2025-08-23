import os
import random
import torch
import numpy as np
from dotenv import load_dotenv
from client_utils.general_utils import set_seed

set_seed()


import signal
import flwr as fl

from server_utils.custom_strategy import CustomStrategy
from config_manager import ConfigurationManager
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FlowerServer(fl.server.Server):
    def __init__(self, strategy: fl.server.strategy.Strategy):
        # Create a SimpleClientManager to manage the clients
        client_manager = fl.server.SimpleClientManager()

        # Initialize the Server class with the SimpleClientManager and custom strategy
        super().__init__(client_manager=client_manager, strategy=strategy)


if __name__ == "__main__":
    config_manager = ConfigurationManager()
    # Access arguments by name
    n_clients = config_manager.get_n_clients()
    n_rounds = config_manager.get_n_rounds()
    port = config_manager.get_port()
    dataset_name = config_manager.get_dataset_name()
    homogeneous = config_manager.get_homogeneous()
    model_name = config_manager.get_model_name()
    # custom_strategy = CustomStrategy(
    #     min_fit_clients=n_clients,
    #     min_evaluate_clients=n_clients,
    #     min_available_clients=n_clients,
    #     rounds=n_rounds,
    #     dataset_name=dataset_name,
    #     homogeneous=homogeneous,
    # )
    # server = FlowerServer(strategy=custom_strategy)
    # fl.server.start_server(
    #             server_address=f"localhost:{port}",
    #             server=server,
    #             config=fl.server.ServerConfig(num_rounds=n_rounds),
    # )
    try:
        custom_strategy = CustomStrategy(
            min_fit_clients=n_clients,
            min_evaluate_clients=n_clients,
            min_available_clients=n_clients,
            rounds=n_rounds,
            dataset_name=dataset_name,
            model_name=model_name,
        )
        server = FlowerServer(strategy=custom_strategy)
        fl.server.start_server(
            server_address=f"localhost:{port}",
            server=server,
            config=fl.server.ServerConfig(num_rounds=n_rounds),
        )
    except Exception as e:
        logger.error(f"{e}")
        os.kill(os.getpid(), signal.SIGTERM)
