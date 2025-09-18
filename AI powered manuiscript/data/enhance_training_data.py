#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhance Ge'ez Training Data

Generates diverse training examples for Ge'ez text restoration.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
import re

# Common Ge'ez words and phrases
GE_EZ_WORDS = [
    "እግዚአብሔር", "አምላክ", "እስራኤል", "ሰላም", "ወልድ", "ንጉሥ", "ምድር", "ሰማይ", 
    "ዓለም", "ነቢይ", "መሰንበት", "ቤተ", "ክርስትያን", "ነገሥታት", "ጸሎት", "መጽሐፍ",
    "ቅዱስ", "ንጹህ", "ቤተክርስቲያን", "መንግሥት", "ንጉሠ", "ነገደ", "ሕዝብ", "ነገር"
]

# Common Ge'ez word separators and punctuation
SEPARATORS = ["፡", "።", "፣", "፤", "፥", "፦", "፧", "፨"]

def generate_sentence(min_words=3, max_words=8) -> str:
    """Generate a random Ge'ez sentence with proper separators."""
    num_words = random.randint(min_words, max_words)
    words = random.sample(GE_EZ_WORDS, num_words)
    
    # Add separators between words
    sentence = words[0]
    for word in words[1:]:
        sep = random.choice(SEPARATORS)
        sentence += f"{sep}{word}"
    
    return sentence

def create_corrupt_example(text: str) -> Tuple[str, str]:
    """Create a corrupted version of the text with common errors."""
    if not text or not text.strip():
        return "", ""
    
    # Split into tokens (words and separators)
    tokens = re.split(r'([፡-፧፨])', text)
    tokens = [t for t in tokens if t.strip()]
    
    if len(tokens) < 2:
        return text, text
    
    # Choose a corruption type
    corruption_type = random.choice([1, 2, 3, 4, 5])
    
    if corruption_type == 1:  # Remove a separator
        sep_indices = [i for i, t in enumerate(tokens) if t in SEPARATORS]
        if sep_indices:
            idx = random.choice(sep_indices)
            corrupted = ''.join(tokens[:idx] + tokens[idx+1:])
            return corrupted, text
    
    elif corruption_type == 2:  # Add extra separator
        if len(tokens) > 2:
            idx = random.randint(1, len(tokens)-1)
            sep = random.choice(SEPARATORS)
            corrupted = ''.join(tokens[:idx] + [sep] + tokens[idx:])
            return corrupted, text
    
    elif corruption_type == 3:  # Replace a word
        word_indices = [i for i, t in enumerate(tokens) if t not in SEPARATORS]
        if word_indices:
            idx = random.choice(word_indices)
            new_word = random.choice([w for w in GE_EZ_WORDS if w != tokens[idx]])
            corrupted = tokens.copy()
            corrupted[idx] = new_word
            return ''.join(corrupted), text
    
    elif corruption_type == 4:  # Remove a word and its following separator
        if len(tokens) >= 3:
            word_indices = [i for i, t in enumerate(tokens) if t not in SEPARATORS]
            if word_indices:
                idx = random.choice(word_indices)
                if idx + 1 < len(tokens) and tokens[idx+1] in SEPARATORS:
                    corrupted = tokens[:idx] + tokens[idx+2:]
                    return ''.join(corrupted), text
    
    # If no corruption was applied, return the original
    return text, text

def generate_examples(count: int = 1000) -> List[Dict[str, str]]:
    """Generate training examples."""
    examples = []
    
    # Add some base examples
    base_examples = [
        ("እግዚአብሔር፡አምላክ፡እስራኤል፡ንጹሕ፡ነው", "እግዚአብሔር፡አምላክ፡እስራኤል፡ንጹሕ፡ነው"),
        ("ወኮነ፡በዓመተ፡አርብዓ፡ወሰማንያ", "ወኮነ፡በዓመተ፡አርብዓ፡ወሰማንያ"),
        ("በሰማይ፡ወበምድር፡ወበገጸ፡ምድር", "በሰማይ፡ወበምድር፡ወበገጸ፡ምድር")
    ]
    
    # Add base examples first
    for corrupt, clean in base_examples:
        examples.append({"input": corrupt, "target": clean})
    
    # Generate random examples
    attempts = 0
    max_attempts = count * 2
    
    while len(examples) < count and attempts < max_attempts:
        attempts += 1
        try:
            clean = generate_sentence()
            if not clean.strip():
                continue
                
            corrupt, clean = create_corrupt_example(clean)
            if corrupt != clean and corrupt.strip() and clean.strip():
                examples.append({"input": corrupt, "target": clean})
        except Exception as e:
            print(f"Error generating example: {e}")
            continue
    
    return examples

def save_examples(examples: List[Dict[str, str]], output_file: str):
    """Save examples to a JSONL file."""
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

def main():
    # Configuration
    output_dir = Path("data/processed")
    train_file = output_dir / "small_training_data.jsonl"
    test_file = output_dir / "small_test_data.jsonl"
    
    # Generate a smaller dataset (100 training, 20 test examples)
    print("Generating small training dataset...")
    examples = generate_examples(count=120)  # Total examples
    
    # Split into train and test (100/20)
    train_examples = examples[:100]
    test_examples = examples[100:]
    
    # Save to files
    save_examples(train_examples, train_file)
    save_examples(test_examples, test_file)
    
    print("\nSmall dataset generation complete!")
    print(f"Training examples: {len(train_examples)}")
    print(f"Test examples: {len(test_examples)}")
    print(f"Saved to:\n- {train_file}\n- {test_file}")

if __name__ == "__main__":
    main()
