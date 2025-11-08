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

- Choose the unlearning method (set `UNLEARNING_METHOD` to `zmugan`, `emmn`, or `lipschitz`)
- Set dataset, model, and federated learning parameters
- Adjust unlearning-specific hyperparameters

**Example `.env` settings:**
```
UNLEARNING_METHOD=zmugan
N_CLIENTS=5
N_ROUNDS=10
DATASET_NAME=svhn
MODEL_NAME=resnet18
FORGET_CLASS=0
# ... other parameters
```

Notes:
- Some booleans are parsed strictly and must be exactly `True` or `False` (case-sensitive, no quotes).
- Quotes around values are acceptable; python-dotenv will handle them.

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

4. **Optional standalone server:**
   ```bash
   python server.py
   ```
   Starts a Flower server at `localhost:PORT` (from `.env`). Separate client processes would be required for this mode.

---

## Unlearning methods (via `UNLEARNING_METHOD`)

- **zMuGAN (`UNLEARNING_METHOD=zmugan`)**
  - Server trains `N_GENERATORS` zMuGAN generators, then runs the unlearning pipeline and replaces global weights with the unlearned model.
  - Model path: `saved_models/unlearned_{MODEL_NAME}_{DATASET_NAME}_forget_class_{FORGET_CLASS}.pth`.

- **EMMN (`UNLEARNING_METHOD=emmn`)**
  - Server runs EMMN-based unlearning and loads the resulting model as global weights.
  - Model path: `saved_models/unlearned_{MODEL_NAME}_{DATASET_NAME}_forget_class_{FORGET_CLASS}_emmn.pth`.

- **Lipschitz (`UNLEARNING_METHOD=lipschitz`)**
  - Clients perform local Lipschitz unlearning; the server aggregates weights per round and tracks a utility score `retain_acc + (100 - forget_acc)` with early stopping.
  - Best model is saved when early stopping triggers.
  - Model path: `saved_models/unlearned_{MODEL_NAME}_{DATASET_NAME}_forget_class_{FORGET_CLASS}_lipschitz.pth`.

4. **Results:**  
   - Server and client results are saved in the `server_results/` and `client_results/` directories.
   - Models are saved in `saved_models/`.

---

## Code Structure

- **Top-level**
  - `run.py` – main simulation entrypoint
  - `server.py` – optional standalone Flower server
  - `client.py` – Flower client implementation
  - `config_manager.py` – loads/validates `.env`
  - Data and experiments: `data_loader.py`, `data_split.py`, `experiment_runner.py`, `emmn_grid_search.py`, `jit_experiment_runner.py`, `jit_grid_search.py`, `zmugan_experiment_runner.py`
  - Utilities: `file_controller.py`, `log_exception.py`, `requirements.txt`, `README.md`

- **Directories**
  - `Machine_Unlearning/`
    - `zMuGAN/` – zMuGAN training and generator components
    - `Unlearning/` – unlearning training/evaluation pipelines (zMuGAN, EMMN, etc.)
  - `client_utils/`
    - `models/` – model factory and architectures
    - Training/eval/data helpers: `train_model.py`, `test_model.py`, `load_client_data.py`, `early_stopping.py`, `weights_controller.py`, `general_utils.py`
  - `server_utils/`
    - `custom_strategy.py` – federated orchestration and unlearning
    - `helper.py`
  - `docs/` – additional documentation (see `docs/POC_Documentation.md`)

---

## Outputs and paths

- Models
  - Original: `MODELS_PATH_ORIGINAL` (e.g., `saved_models`)
  - Unlearned: `MODELS_PATH_UNLEARN` (e.g., `saved_models`)
- Results
  - Training/unlearning: `RESULTS_PATH_UNLEARN`
  - Evaluation: `RESULTS_PATH_UNLEARN_EVAL`
  - Server CSV logs under `server_results/`

---

## Tips and troubleshooting
- Ensure `.env` values exist for keys accessed without defaults (e.g., `NUM_GPUS`, some patience/min-delta fields) to avoid `None`/type errors.
- Use exact `True`/`False` where required by `ConfigurationManager` boolean parsing.
- If using GPUs, CUDA must be available; otherwise, simulation runs on CPU.
- Data splits are controlled via `TRAIN_SPLIT` (train) and implicit `1-TRAIN_SPLIT` (eval) in `run.py`.

---

## Additional Unlearning Methods

This repository also includes implementations of EMMN unlearning and JIT unlearning in the federated context.
