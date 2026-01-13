
<img width="2148" height="869" alt="diagram-export-1-13-2026-4_41_59-AM" src="https://github.com/user-attachments/assets/10d5072f-f5e3-491f-a20a-a80a82059a69" />

# Tashkees-AI: Arabic Medical Question Classification

This repository contains the codebase for **Tashkees-AI**, a system for fine-grained Arabic medical question classification (82 categories). This work corresponds to the Tashkees-AI paper.

## Directory Structure

```
Tashkees-AI/
├── src/
│   ├── config.py               # Configuration constants
│   ├── data_loader.py          # Data loading and preprocessing logic
│   ├── preprocessing.py        # Text cleaning functions (regex)
│   ├── models/
│   │   ├── train_encoder.py    # Training logic for MARBERT/CAMeLBERT
│   │   └── fine_tune_llm.py    # Training logic for Jais (LoRA)
│   └── utils.py                # Helper functions
├── scripts/
│   ├── train.py                # Main training script for baseline models
│   └── tune_hyperparams.py     # Optuna hyperparameter optimization script
├── notebooks/                  # Original notebooks (archived)
├── configs/                    # Configuration files (optional)
└── requirements.txt            # Python dependencies
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Train Baseline (MARBERTv2 / CAMeLBERT)

To train the primary encoder model (Flat Classification):

```bash
python scripts/train.py --data_path "path/to/shared_task_train.csv" --model_name "UBC-NLP/MARBERTv2"
```

Arguments:
- `--model_name`: "UBC-NLP/MARBERTv2" or "CAMeL-Lab/bert-base-arabic-camelbert-mix"
- `--batch_size`: Default 16
- `--epochs`: Default 15 (with early stopping)

### 2. Hyperparameter Tuning

To run Optuna search:

```bash
python scripts/tune_hyperparams.py
```
(Ensure paths in `tune_hyperparams.py` point to your data).

### 3. Fine-tune Jais-13B (LLM)

```bash
python src/models/fine_tune_llm.py "path/to/train.csv" "output_dir"
```

## Preprocessing

The `src/preprocessing.py` module handles Arabic text cleaning, including:
- Removing greetings ("السلام عليكم", etc.)
- Removing structural artifacts ("السؤال", "الجواب")
- Normalizing whitespace

## Citation

Please refer to the `acl_latex.tex` file for citation details.

