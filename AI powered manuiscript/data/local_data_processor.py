#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Ge'ez Text Processing Pipeline

Processes local Ge'ez text files with advanced augmentation and better data handling.
"""

import os
import re
import json
import random
import logging
import unicodedata
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_processing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GeEzDataEnhancer:
    """Enhanced Ge'ez text processing with advanced augmentation."""
    
    def __init__(self):
        # Define Ge'ez Unicode ranges
        self.geez_unicode_ranges = [
            (0x1200, 0x137F),  # Ethiopic
            (0x1380, 0x139F),  # Ethiopic Supplement
            (0x2D80, 0x2DDF),  # Ethiopic Extended
            (0xAB00, 0xAB2F)   # Ethiopic Extended-A
        ]
        
        # Common Ge'ez words and particles for augmentation
        self.common_words = [
            'ወ', 'እና', 'ከ', 'በ', 'ወደ', 'እንደ', 'የ', 'አለ', 'ነው', 'ነበር',
            'ይህ', 'ያ', 'እነዚህ', 'እነዚያ', 'እኔ', 'አንተ', 'እሱ', 'እርሷ', 'እኛ', 'እናንተ',
            'እነሱ', 'ይህ', 'ያ', 'እነዚህ', 'እነዚያ', 'የት', 'እንዴት', 'ለምን', 'መቼ', 'ስንት'
        ]
        
        # Special characters and punctuation
        self.punctuation = '።፤፥፦፡.,;:!?()[]{}<>"\''
    
    def is_geez_text(self, text: str, threshold: float = 0.6) -> bool:
        """Check if text contains sufficient Ge'ez characters."""
        if not text or not text.strip():
            return False
            
        # Remove punctuation and whitespace for better ratio
        clean_text = ''.join(c for c in text if c not in self.punctuation).strip()
        if not clean_text:
            return False
            
        total_chars = len(clean_text)
        if total_chars < 10:  # Minimum length requirement
            return False
            
        geez_count = sum(1 for c in clean_text if self._is_geez_char(c))
        return (geez_count / total_chars) >= threshold
    
    def _is_geez_char(self, char: str) -> bool:
        """Check if character is in Ge'ez Unicode ranges."""
        cp = ord(char)
        return any(start <= cp <= end for start, end in self.geez_unicode_ranges)
    
    def normalize_text(self, text: str) -> str:
        """Normalize Ge'ez text with advanced cleaning."""
        if not text:
            return ""
            
        # Normalize Unicode
        text = unicodedata.normalize('NFC', text)
        
        # Standardize whitespace and newlines
        text = ' '.join(text.split())
        
        # Standardize Ge'ez word space (U+1361)
        text = re.sub(r'[\s\u1361]+', '፡', text)
        
        # Remove control characters except basic whitespace
        text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C' or c in '\n\r\t')
        
        return text.strip()
    
    def augment_text(self, text: str, num_variations: int = 3) -> List[str]:
        """Generate augmented versions of the text."""
        if not text or len(text.split()) < 3:
            return []
            
        variations = set()
        words = text.split()
        
        # Try different augmentation methods
        methods = [
            self._random_deletion,
            self._random_swap,
            self._random_insertion,
            self._random_synonym_replacement,
            self._random_case_mixing,
            self._random_punctuation_addition
        ]
        
        while len(variations) < num_variations and len(methods) > 0:
            method = random.choice(methods)
            try:
                variation = method(words.copy())
                if variation and variation != text and self.is_geez_text(variation):
                    variations.add(variation)
            except Exception as e:
                logger.warning(f"Error in augmentation: {e}")
                methods.remove(method)
                
        return list(variations)
    
    def _random_deletion(self, words: List[str]) -> str:
        """Randomly delete words from the text."""
        if len(words) <= 3:
            return ' '.join(words)
            
        num_deletions = random.randint(1, min(3, len(words) // 3))
        for _ in range(num_deletions):
            idx = random.randint(0, len(words)-1)
            words.pop(idx)
            
        return ' '.join(words)
    
    def _random_swap(self, words: List[str]) -> str:
        """Randomly swap adjacent words."""
        if len(words) < 2:
            return ' '.join(words)
            
        num_swaps = random.randint(1, min(3, len(words) // 2))
        for _ in range(num_swaps):
            idx = random.randint(0, len(words)-2)
            words[idx], words[idx+1] = words[idx+1], words[idx]
            
        return ' '.join(words)
    
    def _random_insertion(self, words: List[str]) -> str:
        """Randomly insert common words."""
        if not words:
            return ""
            
        num_insertions = random.randint(1, min(3, len(words) // 2 + 1))
        for _ in range(num_insertions):
            word = random.choice(self.common_words)
            idx = random.randint(0, len(words))
            words.insert(idx, word)
            
        return ' '.join(words)
    
    def _random_synonym_replacement(self, words: List[str]) -> str:
        """Replace words with synonyms (placeholder - can be enhanced)."""
        if not words:
            return ""
            
        # Simple replacement with common words for now
        for i in range(len(words)):
            if random.random() < 0.3 and words[i] in self.common_words:
                replacement = random.choice(self.common_words)
                if replacement != words[i]:
                    words[i] = replacement
                    
        return ' '.join(words)
    
    def _random_case_mixing(self, words: List[str]) -> str:
        """Randomly change the case of some words."""
        for i in range(len(words)):
            if random.random() < 0.2:  # 20% chance to modify a word
                if random.random() < 0.5:
                    words[i] = words[i].upper()
                else:
                    words[i] = words[i].lower()
                    
        return ' '.join(words)
    
    def _random_punctuation_addition(self, words: List[str]) -> str:
        """Randomly add or remove punctuation."""
        if not words:
            return ""
            
        # Add punctuation at the end
        if random.random() < 0.3:
            words[-1] += random.choice(['።', '።', '፤', '፥', '!', '?'])
            
        # Randomly add commas
        if len(words) > 3 and random.random() < 0.4:
            idx = random.randint(1, len(words)-2)
            words[idx] += '፣'
            
        return ' '.join(words)

def process_directory(input_dir: str, output_file: str, num_augmentations: int = 3):
    """Process all text files in a directory with augmentation."""
    processor = GeEzDataEnhancer()
    input_path = Path(input_dir)
    output_path = Path(output_file)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get all text files
    text_files = list(input_path.glob('**/*.txt'))
    if not text_files:
        logger.error(f"No text files found in {input_dir}")
        return
        
    logger.info(f"Found {len(text_files)} text files to process")
    
    # Process files
    all_paragraphs = []
    for file_path in tqdm(text_files, desc="Processing files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Split into paragraphs and process
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            for para in paragraphs:
                normalized = processor.normalize_text(para)
                if processor.is_geez_text(normalized):
                    all_paragraphs.append(normalized)
                    
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
    
    logger.info(f"Processed {len(all_paragraphs)} valid paragraphs")
    
    # Create training examples (sentence pairs)
    examples = []
    for para in all_paragraphs:
        # Simple sentence splitting - can be enhanced
        sentences = re.split(r'[።!?]+', para)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Create consecutive sentence pairs
        for i in range(len(sentences)-1):
            if len(sentences[i]) > 10 and len(sentences[i+1]) > 10:  # Minimum length
                examples.append({
                    'input': sentences[i],
                    'target': sentences[i+1]
                })
    
    # Augment the data
    augmented_examples = []
    for example in tqdm(examples, desc="Augmenting data"):
        augmented_examples.append(example)
        
        # Augment input
        input_variations = processor.augment_text(example['input'], num_augmentations)
        for var in input_variations:
            augmented_examples.append({'input': var, 'target': example['target']})
            
        # Augment target
        target_variations = processor.augment_text(example['target'], num_augmentations)
        for var in target_variations:
            augmented_examples.append({'input': example['input'], 'target': var})
    
    logger.info(f"Created {len(augmented_examples)} training examples (original + augmented)")
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in augmented_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved training data to {output_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process Ge\'ez text data with augmentation')
    parser.add_argument('--input_dir', type=str, default='data/raw_texts',
                       help='Directory containing input text files')
    parser.add_argument('--output_file', type=str, default='data/processed/training_data.jsonl',
                       help='Output file for processed data')
    parser.add_argument('--augment', type=int, default=3,
                       help='Number of augmentations per example')
    
    args = parser.parse_args()
    process_directory(args.input_dir, args.output_file, args.augment)

if __name__ == "__main__":
    main()
