## Repository Structure

```
.
├── README.md
├── customized_trainer
│   └── customized_trainer.py         # Customized trainer implementation
├── data_file                          # Raw datasets for various tasks
│   └── ...                            # (Subfolders include convai2, llm_adapt, etc.)
├── dataset
│   ├── dataset_hg.py                 # Data loader for heterogeneous datasets
│   ├── dataset_hg_combined.py        # Combined dataset handling
│   └── format_inputs.py              # Functions for formatting model inputs
├── talklora.yml                           # Environment configuration for Conda
├── eval_commonsense.py                    # Evaluation script for commonsense reasoning
├── talklora                              # talklora core modules and tuners
│   ├── peft_model.py                 # Implementation of talklora and related PEFT models
│   ├── mapping.py                    # Mapping utilities for adapting models
│   ├── import_utils.py               # Helper functions for model import and setup
│   └── tuners                        # Various PEFT tuners (e.g., lora, prefix, p_tuning)
│       └── ...                       # Tuners implementations (lora.py, prefix_tuning.py, etc.)
├── models
│   └── get_models.py                 # Functions to load pre-trained models and talklora variants
|
└── train_talklora.py                   # Main entrance script for training and evaluation
```

## Installation

1. **Set Up the Environment:**

   We recommend using Conda with the provided `talklora.yml` file:

   ```bash
   conda env create -f talklora.yml
   conda activate talklora
   ```
2. **Install Dependencies:**

   Ensure that required packages such as `torch`, `transformers`, `numpy`, `tqdm`, `python-dotenv`, and `jsonlines` are installed. The Conda environment should handle these dependencies.

## Quick Start

### Training Example

```
bash train.sh
```

This command fine-tunes the Meta-Llama-3-8B model on the `common_170k` dataset using talklora. Key options include:

- **--peft_type=talklora:** Selects the talklora adaptation method.
- **--model=../llama3:** Specifies the pre-trained model.  The last file is named llama3 or llama2
- **--r_ab=32:** Sets the talklora-specific hyperparameter.
- **--enable_grad_ckpt:** Enables gradient checkpointing.
- **--epoch, --lr, --batch, --seed, --warmup, --eval_strategy, --eval_steps:** Configure training hyperparameters.
- **--output_folder:** Directory to store training outputs.
- **--target_modules:** Specifies target modules for adaptation.

### Evaluation

```
bash run_eval_decoding.sh /path/to/your/checkpoint
```

### Final

Execute the eval_commonsense.py file and assign " your output_folder" to the variable "eval_folder_path" in the file

## Code Details

- **Data Handling:**

  - The `dataset` folder includes modules for loading and preprocessing data from various tasks.
  - Data files are organized under `data_file` by task (e.g., `convai2`, `boolq`, etc.).
- **Model & Tuners:**

  - The core implementations of talklora and other PEFT methods are located in the `talklora` directory.
  - The file `models/get_models.py` contains functions to load pre-trained models and integrate talklora-based adaptations.
- **Training & Evaluation:**

  - The main script, `train_talklora.py`, orchestrates training, evaluation, and checkpointing.
  - Customized training routines and gradient checkpointing are implemented in `customized_trainer/customized_trainer.py`.

## Data Link

- Download the complete commonsense datasets from [here](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main/dataset) and download the commonsense 170k finetuning dataset from [here](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/commonsense_170k.json)
- Then put the downloaded file into corresponding folders (e.g., `data_file/llm_adapt`)
- We also provide a [Google drive download link](https://drive.google.com/file/d/1S_tsqJ8zC_L6fJ4bIQKRf0PUDwNV46HE/view?usp=sharing) for the ease of data downloading.
- Please read and accept the DATA_LICENSE before you download.

## Citation

```bibtex
@inproceedings{
huang2025hira,
title={Hi{RA}: Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models},
author={Qiushi Huang and Tom Ko and Zhan Zhuang and Lilian Tang and Yu Zhang},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=TwJrTz9cRS}
}
```
