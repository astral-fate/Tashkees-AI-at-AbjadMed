import torch
import os
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from sklearn.model_selection import train_test_split

def create_prompt(text, category=None, for_training=True):
    """Create Arabic prompt for classification"""
    prompt = f"""أنت نظام ذكاء اصطناعي طبي. صنف السؤال الطبي إلى واحدة من 82 فئة.

السؤال: {text.strip()}

الفئة:"""

    if for_training and category is not None:
        prompt += f" {category}"

    return prompt

def train_jais_lora(
    train_path,
    output_dir,
    base_model_name="core42/jais-13b",
    epochs=3,
    max_length=512
):
    print(f"Loading data from {train_path}")
    df = pd.read_csv(train_path)
    df = df.dropna(subset=['text', 'label', 'category'])
    
    # Label mapping (assuming column 'category' exists with names)
    label_to_category = dict(zip(df['label'], df['category']))
    
    train_split, val_split = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])
    
    # Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
        bnb_8bit_use_double_quant=True,
    )
    
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # LoRA Config
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    
    # Preprocess
    def preprocess_function(examples):
        texts = []
        for text, label in zip(examples['text'], examples['label']):
            category = label_to_category[label]
            prompt = create_prompt(text, category, for_training=True)
            texts.append(prompt)
        
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    train_ds = Dataset.from_pandas(train_split)
    val_ds = Dataset.from_pandas(val_split)
    
    train_ds = train_ds.map(preprocess_function, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(preprocess_function, batched=True, remove_columns=val_ds.column_names)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    print("Starting Training...")
    trainer.train()
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 2:
        train_jais_lora(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python fine_tune_llm.py <train_csv> <output_dir>")
