#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Test Split

Creates a test split from existing training data for model evaluation.
"""

import os
import json
import random
from pathlib import Path

# Configuration
DATA_DIR = Path("data/processed")
TRAIN_FILE = DATA_DIR / "training_data.jsonl"
TEST_FILE = DATA_DIR / "test.jsonl"
TEST_SPLIT = 0.2  # 20% for testing
RANDOM_SEED = 42

def load_examples(file_path):
    """Load examples from a JSONL file."""
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                ex = json.loads(line)
                if 'input' in ex and 'target' in ex:
                    examples.append(ex)
            except json.JSONDecodeError:
                continue
    return examples

def create_test_split():
    """Create test split from training data."""
    # Create output directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Load training data
    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")
    
    examples = load_examples(TRAIN_FILE)
    if not examples:
        raise ValueError(f"No valid examples found in {TRAIN_FILE}")
    
    # Shuffle examples
    random.seed(RANDOM_SEED)
    random.shuffle(examples)
    
    # Split into train and test
    split_idx = int(len(examples) * (1 - TEST_SPLIT))
    test_examples = examples[split_idx:]
    
    # Save test split
    with open(TEST_FILE, 'w', encoding='utf-8') as f:
        for ex in test_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    print(f"Created test split with {len(test_examples)} examples at {TEST_FILE}")
    print(f"Example test entry: {json.dumps(test_examples[0], ensure_ascii=False)}")

if __name__ == "__main__":
    create_test_split()
