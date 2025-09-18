import os
import json
import torch
from torch.optim import AdamW
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm

# Configuration
class Config:
    model_name = "google/mt5-small"  # Using mT5 which has better multilingual support
    train_file = "data/processed/small_training_data.jsonl"
    val_file = "data/processed/small_test_data.jsonl"
    output_dir = "models/geez_t5_small"
    max_length = 64
    batch_size = 4
    num_epochs = 5
    learning_rate = 3e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"

# Load data
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def preprocess_text(text):
    # Add space around Ge'ez word separators for better tokenization
    for sep in ['፡', '።', '፣', '፤', '፥', '፦', '፧', '፨']:
        text = text.replace(sep, f' {sep} ')
    return ' '.join(text.split())  # Normalize whitespace

# Main training function
def train():
    # Setup
    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)
    
    print(f"Using device: {config.device}")
    print("Loading tokenizer and model...")
    
    # Load tokenizer and model with special tokens
    tokenizer = T5Tokenizer.from_pretrained(
        config.model_name,
        extra_ids=0,
        additional_special_tokens=[],
        legacy=False
    )
    
    # Add Ge'ez tokens to the tokenizer
    ge_ez_chars = set()
    for word in ["እግዚአብሔር", "አምላክ", "እስራኤል", "ሰላም", "ወልድ", "ንጉሥ", "ምድር", "ሰማይ", 
                "ዓለም", "ነቢይ", "መሰንበት", "ቤተ", "ክርስትያን", "ነገሥታት", "ጸሎት", "መጽሐፍ",
                "ቅዱስ", "ንጹህ", "ቤተክርስቲያን", "መንግሥት", "ንጉሠ", "ነገደ", "ሕዝብ", "ነገር"]:
        ge_ez_chars.update(word)
    
    tokenizer.add_tokens(list(ge_ez_chars))
    
    # Initialize model with resized token embeddings
    model = T5ForConditionalGeneration.from_pretrained(config.model_name)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(config.device)
    
    print("Loading data...")
    # Load and preprocess data
    train_data = load_data(config.train_file)
    val_data = load_data(config.val_file)
    
    # Preprocess text data
    for item in train_data + val_data:
        item['input'] = preprocess_text(item['input'])
        item['target'] = preprocess_text(item['target'])
    
    print(f"Training on {len(train_data)} examples, validating on {len(val_data)} examples")
    
    # Training loop
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        
        # Train
        model.train()
        total_loss = 0
        
        for item in tqdm(train_data, desc="Training"):
            # Tokenize input and target
            input_encoding = tokenizer(
                item['input'],
                max_length=config.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(config.device)
            
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    item['target'],
                    max_length=config.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).input_ids.to(config.device)
            
            # Forward pass
            outputs = model(
                input_ids=input_encoding.input_ids,
                attention_mask=input_encoding.attention_mask,
                labels=labels
            )
            
            # Backward pass
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        
        # Print training stats
        avg_loss = total_loss / len(train_data)
        print(f"Train loss: {avg_loss:.4f}")
        
        # Save model
        model.save_pretrained(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)
        print(f"Model saved to {config.output_dir}")
        
        # Quick validation
        model.eval()
        with torch.no_grad():
            # Test on first few examples
            test_item = val_data[0]
            test_input = test_item['input']
            
            # Tokenize input
            input_encoding = tokenizer(
                test_input,
                max_length=config.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(config.device)
            
            # Generate output
            generated_ids = model.generate(
                input_ids=input_encoding.input_ids,
                attention_mask=input_encoding.attention_mask,
                max_length=config.max_length,
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True,
                temperature=0.9,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )
            
            output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print("\nSample prediction:")
            print(f"Input:    {test_input}")
            print(f"Expected: {test_item['target']}")
            print(f"Output:   {output}")
            
            # Show tokenization details
            print("\nToken Debug:")
            print(f"Input tokens: {tokenizer.convert_ids_to_tokens(input_encoding.input_ids[0])}")
            print(f"Expected tokens: {tokenizer.tokenize(test_item['target'])}")
            print(f"Generated tokens: {tokenizer.convert_ids_to_tokens(generated_ids[0])}")

if __name__ == "__main__":
    train()
