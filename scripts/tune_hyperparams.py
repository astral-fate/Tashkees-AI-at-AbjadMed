import optuna
import os
import sys
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import *
from src.data_loader import load_and_prepare_data, create_tokenize_fn
from src.models.train_encoder import WeightedTrainer, compute_metrics

def objective(trial, train_ds, val_ds, num_labels, model_name, class_weights):
    
    # Define Hyperparameters to Tune
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True)
    batch_size = trial.suggest_categorical("per_device_train_batch_size", [16, 32])
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.3)
    num_train_epochs = trial.suggest_int("num_train_epochs", 3, 5)

    # Model Init Function used by Optuna
    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "optuna_temp"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = WeightedTrainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights
    )

    trainer.train()
    eval_result = trainer.evaluate()
    return eval_result['eval_f1_macro']

def main():
    # Hardcoded for example, should be arg
    DATA_PATH = os.path.join(DATA_DIR, 'shared_task_train.csv')
    MODEL_NAME = DEFAULT_MODEL_NAME
    
    if not os.path.exists(DATA_PATH):
        print(f"Data not found at {DATA_PATH}. Please ensure data exists.")
        return

    train_ds, val_ds, num_labels = load_and_prepare_data(DATA_PATH)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenize_fn = create_tokenize_fn(tokenizer)
    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)
    
    cols = ['input_ids', 'attention_mask', 'label']
    train_ds.set_format("torch", columns=cols)
    val_ds.set_format("torch", columns=cols)
    
    # Pre-calculate weights
    labels = train_ds['label']
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(weights, dtype=torch.float)
    if torch.cuda.is_available():
        class_weights = class_weights.to("cuda")

    print("🚀 Starting Hyperparameter Search...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, train_ds, val_ds, num_labels, MODEL_NAME, class_weights), n_trials=10)
    
    print("\n🏆 Best Trial:")
    print(study.best_trial.params)

if __name__ == "__main__":
    main()
