import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

# Configuration
class Config:
    model_name = "t5-tiny-random"  # Very small model
    train_file = "data/processed/small_training_data.jsonl"
    val_file = "data/processed/small_test_data.jsonl"
    output_dir = "models/geez_t5_small"
    max_length = 64
    batch_size = 4
    num_epochs = 3
    learning_rate = 1e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"

# Load data
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# Main training function
def train():
    # Setup
    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small").to(config.device)
    
    # Load data
    train_data = load_data(config.train_file)
    val_data = load_data(config.val_file)
    
    # Training loop
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        
        # Train
        model.train()
        total_loss = 0
        for batch in tqdm(train_data, desc="Training"):
            inputs = tokenizer(
                batch['input'],
                max_length=config.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(config.device)
            
            labels = tokenizer(
                batch['target'],
                max_length=config.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(config.device)
            
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        
        print(f"Train loss: {total_loss/len(train_data):.4f}")
        
        # Save model
        model.save_pretrained(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)

if __name__ == "__main__":
    train()