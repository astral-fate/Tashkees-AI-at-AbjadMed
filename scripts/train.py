import argparse
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import *
from src.data_loader import load_and_prepare_data, create_tokenize_fn
from src.models.train_encoder import train_model
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Train Tashkees-AI Baseline Model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Model name (e.g., UBC-NLP/MARBERTv2)")
    parser.add_argument("--output_dir", type=str, default=MODELS_DIR, help="Output directory for model")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=2e-5)
    
    args = parser.parse_args()
    
    print(f"Initializing training for {args.model_name}...")
    
    # Load Data
    train_ds, val_ds, num_labels = load_and_prepare_data(args.data_path)
    print(f"Number of labels: {num_labels}")
    
    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenize_fn = create_tokenize_fn(tokenizer)
    
    print("Tokenizing datasets...")
    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)
    
    # Set format
    cols = ['input_ids', 'attention_mask', 'label']
    train_ds.set_format("torch", columns=cols)
    val_ds.set_format("torch", columns=cols)
    
    # Train
    output_path = os.path.join(args.output_dir, args.model_name.replace("/", "_"))
    trainer = train_model(
        model_name=args.model_name,
        train_dataset=train_ds,
        val_dataset=val_ds,
        num_labels=num_labels,
        output_dir=output_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        tokenizer=tokenizer
    )
    
    print("Training complete.")
    
    # Save Final
    final_save_path = os.path.join(output_path, "final_model")
    trainer.save_model(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"Model saved to {final_save_path}")

if __name__ == "__main__":
    main()
