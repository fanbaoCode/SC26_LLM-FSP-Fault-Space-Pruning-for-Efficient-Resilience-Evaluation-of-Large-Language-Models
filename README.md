# README

## Prerequisites

Ensure that the following prerequisites are met before setting up the environment:

- **Operating System**: Ubuntu 20.04
- **Hardware**: Follow the previously described hardware configuration.
- **Software**: CUDA and cuDNN deep neural network acceleration library.

## Installation

### Step 1: Set Up the Computing Platform

Install **CUDA** and **cuDNN** on your **Ubuntu 20.04** system. Follow the official installation guides for [CUDA](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn).

### Step 2: Create the Conda Environment

Create a **Conda** environment and install all the required dependencies by running the following command:

```bash
conda env update --file environment.yml --prune
```

This will create the environment and install all dependencies listed in the `environment.yml` file.

## Data Preparation

Download the required LLMs (Large Language Models) and datasets needed for the artifact. Ensure that you have access to the specific models and datasets, as described in the project documentation.

## Running the Scripts

### Running Full Space Fault Injection

To perform fault injection across the entire fault space, run the script `run_all_full_space.sh`. This will execute fault injections for both 1-bit and 2-bit faults.

```bash
bash run_all_full_space.sh
```

### Running Pruned Space Fault Injection

To perform fault injection within the pruned fault space, run the script `run_all_pruning_space.sh`. This will execute fault injections for both 1-bit and 2-bit faults in the pruned fault space.

```bash
bash run_all_pruning_space.sh
```

### Modifying Parameters for Fault Injection

To change the data type for fault injection (e.g., FP16, BF16, or FP32), navigate to the `models_inference_fi` directory and modify the `torch_dtype` parameter in the fault injection code. You can set it to one of the following options:

```python
torch_dtype = FP16
torch_dtype = BF16
torch_dtype = FP32
```

Additionally, you can control the number of fault injection trials by modifying the `num_trials` parameter:

```python
num_trials = <number_of_trials>
```

Set `num_trials` to the desired number of fault injection experiments to be performed.

You can also adjust the following parameters for fault space pruning:

- **`K`**: Controls the number of decode steps. Set this parameter to the desired number of steps. The initial value is:

  ```python
  K = 2
  ```

- **`delta`**: Controls the pruning of the hidden layer dimension error space. Set this parameter to the desired threshold. The initial value is:

  ```python
  delta = 0.3
  ```

## Storing Fault Injection Results

The results of each fault injection experiment will be organized in the `results` folder in JSON format. Each experiment will have a corresponding JSON file containing the fault injection information.

## Analyzing the Results

### Full Space Resilience Analysis

After completing the fault injection experiments, run the script `analysis_full_space.sh` to analyze the LLM resilience across different dimensions as described in the paper.

```bash
bash analysis/analysis_full_space.sh
```

### Pruned Space Resilience Analysis

To analyze the resilience of the pruned LLM fault space, run the script `analysis_pruning_space.sh`.

```bash
bash analysis/analysis_pruning_space.sh
```

## Conclusion

This repository provides the tools and instructions necessary to perform fault injection and resilience analysis on large language models (LLMs) across different fault spaces and data types. Ensure that you follow the setup instructions carefully and modify the parameters as required for your specific experiments.