import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from .preprocessing import clean_text

def load_and_prepare_data(train_path, val_split=0.1, seed=42):
    """
    Loads data from CSV, cleans it, and splits into train/validation datasets.
    Returns Hugging Face Datasets.
    """
    print(f"Loading data from {train_path}...")
    df = pd.read_csv(train_path)
    
    # Basic cleanup
    df = df.dropna(subset=['text', 'label'])
    
    # Ensure label is int if possible, or handle mapping if needed
    # For now assuming 'label' column exists and is numeric as per notebooks
    if 'label' in df.columns:
        df['label'] = df['label'].astype(int)

    # Clean text
    print("Cleaning text...")
    df['text'] = df['text'].astype(str).apply(clean_text)
    
    # Split
    print(f"Splitting data with test_size={val_split}...")
    train_df, val_df = train_test_split(
        df,
        test_size=val_split,
        random_state=seed,
        stratify=df['label'] if 'label' in df.columns else None
    )
    
    # Convert to HF Datasets
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    
    return train_ds, val_ds, df['label'].nunique() if 'label' in df.columns else 0

def create_tokenize_fn(tokenizer, max_length=128):
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)
    return tokenize_function
