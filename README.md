# Federated Unlearning

**Abstract:**
This project integrates the Zero-shot Machine Unlearning GAN (zMuGAN) algorithm into a federated learning (FL) environment to unlearn a designated class from the model. The system orchestrates a three-phase federated workflow—initial federated learning, scratch training for baseline calculation, and ZMUGAN-based unlearning—across a centralized server and multiple clients. 

## Introduction

This project aims to advance the state of machine unlearning in federated learning by incorporating the ZMUGAN algorithm into a realistic FL workflow. The system architecture consists of a centralized server and multiple clients, each responsible for local data management, model training, and evaluation. The federated workflow is divided into three distinct phases:

- **Phase 1: Initial Federated Learning** – Clients train on their complete datasets, including the target forget class, with the server coordinating model aggregation and early stopping. The best-performing model is saved for subsequent phases.
- **Phase 2: Scratch Training** – Clients retrain models from scratch, excluding the forget class, to establish a baseline for the Anamnesis Index (AIN) calculation.
- **Phase 3: ZMUGAN Unlearning** – The server initiates the ZMUGAN unlearning process, training multiple generators to produce synthetic data via model inversion. This synthetic data is used for a two-step unlearning process: impairing the model's memory of the forget class and repairing performance on the retain classes.

The unlearning process leverages synthetic data generation, advanced loss functions, and a two-step impair/repair protocol to ensure effective removal of the forget class while preserving utility on retained data. Evaluation metrics include forget/retain accuracy, membership inference resistance, and the anamnesis index, all aggregated in a federated workflow.

---

## Getting Started


### Installation

1. Clone the repository:
   ```bash
   git clone ------
   cd Federated_Unlearning
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

All experiment and system parameters are set in the `.env` file located in the project root. Before running any experiments, edit this file to:

- Choose the unlearning method (e.g., set the main method to `zmugan`)
- Set dataset, model, and federated learning parameters
- Adjust unlearning-specific hyperparameters

**Example `.env` settings:**
```
UNLEARNING_METHOD=zmugan
N_CLIENTS=10
N_ROUNDS=20
DATASET_NAME=your_dataset
MODEL_NAME=your_model
# ... other parameters
```

---

## Running Experiments

1. **Edit the `.env` file** to set your desired configuration and unlearning method.

2. **Run the main experiment script:**
   ```bash
   python run.py
   ```

   This will launch a federated learning simulation using the specified settings and perform unlearning as configured.

3. **Batch experiments:**  
   To run multiple experiments with different hyperparameters, use:
   ```bash
   python experiment_runner.py
   ```

4. **Results:**  
   - Server and client results are saved in the `server_results/` and `client_results/` directories.
   - Models are saved in `saved_models/`.

---

## Project Structure

- `client.py` – Federated client logic, including unlearning routines
- `server.py` – Federated server logic and orchestration
- `run.py` – Main entry point for running a federated learning/unlearning experiment
- `experiment_runner.py` – Automates running multiple experiments with different settings
- `config_manager.py` – Loads and manages configuration from `.env`
- `client_utils/` and `server_utils/` – Helper modules for client/server operations
- `data/`, `client_data/` – Datasets and data splits
- `saved_models/` – Saved model checkpoints
- `server_results/`, `client_results/`, `experiment_results/` – Output and logs

---

## Additional Unlearning Methods

This repository also includes implementations of EMMN unlearning and JIT unlearning in the federated context.
