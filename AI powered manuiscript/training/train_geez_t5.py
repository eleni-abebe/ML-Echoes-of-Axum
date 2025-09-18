#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ge'ez T5 Model Training

Trains a T5 model for Ge'ez text restoration using the enhanced dataset.
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
    get_linear_schedule_with_warmup
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
    inputs = [f"restore: {item['input']}" for item in batch]
    targets = [item['target'] for item in batch]
    
    # Tokenize inputs
    input_encodings = tokenizer(
        inputs,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(
            targets,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
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
        
        # Backward pass and optimize
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
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    
    # Configuration
    config = {
        'model_name': 't5-small',  # Start with a smaller model for faster training
        'train_file': str(project_root / 'data' / 'processed' / 'enhanced_training_data.jsonl'),
        'val_file': str(project_root / 'data' / 'processed' / 'enhanced_test_data.jsonl'),
        'output_dir': str(project_root / 'models' / 'geez_t5_enhanced'),
        'batch_size': 16,
        'num_epochs': 20,
        'learning_rate': 3e-4,
        'warmup_steps': 100,
        'max_grad_norm': 1.0,
        'max_length': 128,
        'seed': 42
    }
    
    # Set random seed for reproducibility
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("Configuration:")
    logger.info(f"- Model: {config['model_name']}")
    logger.info(f"- Training file: {config['train_file']}")
    logger.info(f"- Validation file: {config['val_file']}")
    logger.info(f"- Output directory: {config['output_dir']}")
    
    # Verify data files exist
    for file_type in ['train_file', 'val_file']:
        file_path = Path(config[file_type])
        if not file_path.exists():
            logger.error(f"{file_type} not found at: {file_path}")
            logger.info("Please run data/enhance_training_data.py first to generate the training data.")
            return

    # Load tokenizer and model
    logger.info("Loading tokenizer and model...")
    tokenizer = T5Tokenizer.from_pretrained(config['model_name'])
    model = T5ForConditionalGeneration.from_pretrained(config['model_name']).to(device)
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = GeEzDataset(config['train_file'], tokenizer, config['max_length'])
    val_dataset = GeEzDataset(config['val_file'], tokenizer, config['max_length'])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, config['max_length'])
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        collate_fn=lambda b: collate_fn(b, tokenizer, config['max_length'])
    )
    
    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    total_steps = len(train_loader) * config['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=total_steps
    )
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        logger.info(f"Epoch {epoch + 1}/{config['num_epochs']}")
        
        # Train for one epoch
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            config['max_grad_norm']
        )
       
        # Evaluate on validation set
        val_loss = evaluate(model, val_loader, device)
        
        logger.info(f"Epoch {epoch + 1} - Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info(f"New best model saved to {output_dir}")
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()
