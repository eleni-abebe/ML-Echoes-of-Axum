#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Ge'ez Text Restoration Model

Evaluates the performance of the trained model on the test set.
"""

import json
import logging
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple
from transformers import T5ForConditionalGeneration, T5Tokenizer
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeEzDataset:
    """Dataset for Ge'ez text restoration evaluation."""
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

def load_model_and_tokenizer(model_dir: str, device: str) -> Tuple[T5ForConditionalGeneration, T5Tokenizer]:
    """Load the trained model and tokenizer."""
    logger.info(f"Loading model from {model_dir}")
    model = T5ForConditionalGeneration.from_pretrained(model_dir).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    return model, tokenizer

def evaluate_model(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    test_data: List[Dict],
    device: str,
    max_length: int = 128,
    num_samples: int = 10
) -> None:
    """Evaluate the model and print sample predictions."""
    model.eval()
    
    logger.info("\nSample Predictions:")
    logger.info("-" * 80)
    
    for i in range(min(num_samples, len(test_data))):
        example = test_data[i]
        input_text = f"restore: {example['input']}"
        
        # Tokenize input
        input_ids = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True
        ).input_ids.to(device)
        
        # Generate prediction with better parameters
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_length=max_length,
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                decoder_start_token_id=tokenizer.pad_token_id
            )
        
        # Decode prediction
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info(f"Input:    {example['input']}")
        logger.info(f"Expected: {example['target']}")
        logger.info(f"Predicted: {prediction}")
        logger.info("-" * 80)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate Ge'ez T5 model")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--test_file', type=str, required=True, help='Path to the test JSONL file')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to evaluate')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('evaluation.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_path}")
    
    # Load tokenizer with special tokens handling
    tokenizer = T5Tokenizer.from_pretrained(
        args.model_path,
        legacy=False
    )
    
    # Load model
    model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set model config for generation
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.pad_token_id
    
    # Load test data
    test_dataset = GeEzDataset(args.test_file, tokenizer)
    
    # Evaluate model
    evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_data=test_dataset.examples,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_samples=args.num_samples
    )

if __name__ == "__main__":
    main()
