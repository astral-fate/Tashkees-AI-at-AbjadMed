import torch
import numpy as np
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback,
    AutoTokenizer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
             loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
             loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
             loss_fct = nn.CrossEntropyLoss()
             loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
             
        return (loss, outputs) if return_outputs else loss

class LogMetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print("\n" + "="*40)
            print(f"📢 Epoch {state.epoch:.1f} Evaluation Results:")
            if 'eval_accuracy' in metrics:
                print(f"   ✅ Accuracy:  {metrics['eval_accuracy']:.2%}")
            if 'eval_f1_macro' in metrics:
                print(f"   ✅ Macro F1:  {metrics['eval_f1_macro']:.2%}")
            if 'eval_loss' in metrics:
                print(f"   📉 Validation Loss: {metrics['eval_loss']:.4f}")
            print("="*40 + "\n")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')
    return {
        "accuracy": acc,
        "f1_macro": f1_macro
    }

def train_model(
    model_name,
    train_dataset,
    val_dataset,
    num_labels,
    output_dir,
    batch_size=16,
    epochs=4,
    learning_rate=2e-5,
    use_weighted_loss=True,
    tokenizer=None
):
    # Setup tokenizer if not provided
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Calculate class weights
    class_weights = None
    if use_weighted_loss:
        labels = train_dataset['label']
        # Convert to numpy if it's a tensor
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
            
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(labels),
            y=labels
        )
        class_weights = torch.tensor(weights, dtype=torch.float)
        if torch.cuda.is_available():
            class_weights = class_weights.to("cuda")
    
    # Model init
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[LogMetricsCallback, EarlyStoppingCallback(early_stopping_patience=3)],
        class_weights=class_weights
    )

    trainer.train()
    
    return trainer
