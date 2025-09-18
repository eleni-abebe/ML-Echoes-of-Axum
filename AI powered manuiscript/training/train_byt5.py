#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Ge'ez Text Restoration Model Training

Trains a T5 model for Ge'ez text restoration with improved data handling and training configuration.
"""

import os
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
    set_seed
)
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GeEzDataset(Dataset):
    """Dataset for Ge'ez text restoration."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._load_examples(data_path)
    
    def _load_examples(self, data_path: str) -> List[Dict]:
        """Load examples from a JSONL file."""
        examples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    example = json.loads(line.strip())
                    if 'input' in example and 'target' in example:
                        examples.append(example)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line: {line[:100]}...")
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict:
        example = self.examples[idx]
        return {
            'input_text': example['input'],
            'target_text': example['target']
        }

def collate_fn(batch, tokenizer, max_length: int = 128):
    """Collate function for DataLoader."""
    inputs = [f"restore: {item['input_text']}" for item in batch]
    targets = [item['target_text'] for item in batch]
    
    # Tokenize inputs
    input_encoding = tokenizer(
        inputs,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        target_encoding = tokenizer(
            targets,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
    
    # Replace padding token id with -100 for loss calculation
    labels = target_encoding['input_ids'].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    return {
        'input_ids': input_encoding['input_ids'],
        'attention_mask': input_encoding['attention_mask'],
        'labels': labels
    }

def train_model(
    model,
    train_dataset,
    val_dataset,
    tokenizer,
    output_dir: str,
    batch_size: int = 8,
    num_epochs: int = 10,
    learning_rate: float = 3e-5,
    warmup_steps: int = 100,
    max_grad_norm: float = 1.0,
    device: str = None,
    seed: int = 42
):
    """Train the model with improved training loop."""
    # Set random seed for reproducibility
    set_seed(seed)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set up device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    
    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        # Training phase
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_train_loss += loss.item()
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate average training loss
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_val_loss += outputs.loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Log metrics
        logger.info(f"Epoch {epoch + 1}/{num_epochs}:")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        logger.info(f"  Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(output_path / 'best_model')
            tokenizer.save_pretrained(output_path / 'best_model')
            logger.info(f"  New best model saved with val_loss: {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"  No improvement in val_loss for {patience_counter} epochs")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break
    
    # Save final model
    model.save_pretrained(output_path / 'final_model')
    tokenizer.save_pretrained(output_path / 'final_model')
    logger.info(f"Training completed. Final model saved to {output_path / 'final_model'}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a T5 model for Ge\'ez text restoration')
    parser.add_argument('--train_file', type=str, required=True,
                       help='Path to training data (JSONL file)')
    parser.add_argument('--val_file', type=str, required=True,
                       help='Path to validation data (JSONL file)')
    parser.add_argument('--output_dir', type=str, default='models/geez_t5_enhanced',
                       help='Directory to save the trained model')
    parser.add_argument('--model_name', type=str, default='t5-small',
                       help='Base model name or path')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                       help='Learning rate')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sequence length')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Initialize tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    
    # Add special tokens if needed
    special_tokens = {'additional_special_tokens': ['<sep>', '<mask>']}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    # Load datasets
    train_dataset = GeEzDataset(args.train_file, tokenizer, args.max_length)
    val_dataset = GeEzDataset(args.val_file, tokenizer, args.max_length)
    
    logger.info(f"Loaded {len(train_dataset)} training examples and {len(val_dataset)} validation examples")
    
    # Train the model
    train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate
    )

if __name__ == "__main__":
    main()
