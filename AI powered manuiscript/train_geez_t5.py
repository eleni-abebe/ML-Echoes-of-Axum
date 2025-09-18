#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed Ge'ez T5 Model Training

This script trains a T5 model for Ge'ez text restoration with proper configuration.
"""

import os
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
    set_seed
)
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
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
                example = json.loads(line)
                examples.append({
                    'input': example['input'],
                    'target': example['target']
                })
        return examples
   
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.examples[idx]

def collate_fn(batch, tokenizer, max_length: int = 128):
    """Collate function for DataLoader."""
    inputs = [f"restore Ge'ez: {item['input']}" for item in batch]
    targets = [item['target'] for item in batch]
    
    # Tokenize inputs
    input_encodings = tokenizer(
        inputs,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
        add_special_tokens=True
    )
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(
            targets,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True
        )
    
    return {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['input_ids'],
        'decoder_attention_mask': target_encodings['attention_mask']
    }

def train_epoch(model, train_loader, optimizer, scheduler, device, max_grad_norm=1.0):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        decoder_attention_mask = batch['decoder_attention_mask'].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )
        
        # Backward pass
        loss = outputs.loss
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # Update parameters
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Update progress
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(train_loader)

def evaluate(model, val_loader, device):
    """Evaluate the model on the validation set."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            decoder_attention_mask = batch['decoder_attention_mask'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                decoder_attention_mask=decoder_attention_mask
            )
            
            total_loss += outputs.loss.item()
    
    return total_loss / len(val_loader)

def main():
    # Set random seed for reproducibility
    set_seed(42)
    
    # Configuration
    config = {
        'model_name': 't5-small',  # Start with a smaller model for faster training
        'train_data_path': 'data/processed/enhanced_training_data.jsonl',
        'val_data_path': 'data/processed/valid.jsonl',
        'output_dir': 'models/geez_t5_restored',
        'max_length': 128,
        'batch_size': 8,
        'learning_rate': 3e-4,
        'num_epochs': 10,
        'warmup_steps': 100,
        'max_grad_norm': 1.0,
        'save_steps': 500,
        'logging_steps': 100,
    }
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Initialize tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(
        config['model_name'],
        use_fast=False,
        add_prefix_space=True
    )
    
    # Add special tokens if needed
    special_tokens = {
        'additional_special_tokens': ['<geez>', '</geez>']
    }
    tokenizer.add_special_tokens(special_tokens)
    
    # Initialize model
    model = T5ForConditionalGeneration.from_pretrained(config['model_name'])
    model.resize_token_embeddings(len(tokenizer))
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load datasets
    train_dataset = GeEzDataset(config['train_data_path'], tokenizer, config['max_length'])
    val_dataset = GeEzDataset(config['val_data_path'], tokenizer, config['max_length'])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, tokenizer, config['max_length'])
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        collate_fn=lambda x: collate_fn(x, tokenizer, config['max_length'])
    )
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    
    # Calculate total training steps
    total_steps = len(train_loader) * config['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=total_steps
    )
    
    
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            config['max_grad_norm']
        )
        
        val_loss = evaluate(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(config['output_dir'])
            tokenizer.save_pretrained(config['output_dir'])
            print(f"Saved best model to {config['output_dir']}")
    
    print("Training complete!")

if __name__ == "__main__":
    main()
