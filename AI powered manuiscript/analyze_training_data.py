#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive analysis of training data for Ge'ez text restoration.
Examines data quality, format, and tokenization to ensure proper model training.
"""

import os
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training_data_analysis.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class TrainingDataAnalyzer:
    def __init__(self, data_paths):
        """Initialize the analyzer with paths to training data."""
        if isinstance(data_paths, (str, Path)):
            data_paths = [data_paths]
        self.data_paths = [Path(p) for p in data_paths]
        self.examples = []
        self.stats = {
            'total_examples': 0,
            'total_chars': 0,
            'total_tokens': 0,
            'char_distribution': Counter(),
            'token_lengths': [],
            'example_lengths': [],
            'geez_char_count': 0,
            'non_geez_chars': set(),
            'empty_examples': 0,
            'malformed_examples': 0,
            'missing_fields': 0,
            'file_stats': {}
        }
        
        # Ge'ez Unicode ranges
        self.geez_unicode_ranges = [
            (0x1200, 0x137F),  # Ethiopic
            (0x1380, 0x139F),  # Ethiopic Supplement
            (0x2D80, 0x2DDF),  # Ethiopic Extended
            (0xAB00, 0xAB2F)   # Ethiopic Extended-A
        ]
        
    def is_geez_char(self, char):
        """Check if a character is in the Ge'ez Unicode ranges."""
        cp = ord(char)
        return any(start <= cp <= end for start, end in self.geez_unicode_ranges)
    
    def load_data(self):
        """Load and validate training examples from all data paths."""
        logger.info("Loading training data...")
        
        for data_path in self.data_paths:
            if not data_path.exists():
                logger.warning(f"Data file not found: {data_path}")
                continue
                
            file_stats = {
                'total_examples': 0,
                'empty_examples': 0,
                'malformed_examples': 0,
                'char_count': 0,
                'geez_chars': set(),
                'non_geez_chars': set()
            }
            
            try:
                with open(data_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(tqdm(f, desc=f"Reading {data_path.name}")):
                        line = line.strip()
                        if not line:
                            self.stats['empty_examples'] += 1
                            file_stats['empty_examples'] += 1
                            continue
                            
                        try:
                            example = json.loads(line)
                            if not isinstance(example, dict) or 'input' not in example or 'target' not in example:
                                self.stats['malformed_examples'] += 1
                                file_stats['malformed_examples'] += 1
                                continue
                                
                            self.examples.append(example)
                            self.stats['total_examples'] += 1
                            file_stats['total_examples'] += 1
                            
                            # Analyze text content
                            for field in ['input', 'target']:
                                if field in example:
                                    file_stats['char_count'] += len(example[field])
                                    for char in example[field]:
                                        if self.is_geez_char(char):
                                            file_stats['geez_chars'].add(char)
                                        elif char.strip() and not char.isspace():
                                            file_stats['non_geez_chars'].add(char)
                            
                        except json.JSONDecodeError:
                            self.stats['malformed_examples'] += 1
                            file_stats['malformed_examples'] += 1
                            logger.warning(f"Invalid JSON on line {i+1} in {data_path}")
                            
                # Update file stats
                file_stats['geez_chars'] = len(file_stats['geez_chars'])
                file_stats['non_geez_chars'] = len(file_stats['non_geez_chars'])
                self.stats['file_stats'][str(data_path)] = file_stats
                            
            except Exception as e:
                logger.error(f"Error reading {data_path}: {str(e)}")
                continue
                
        logger.info(f"Loaded {len(self.examples)} valid examples from {len(self.data_paths)} files")
    
    def analyze_text(self, text):
        """Analyze text content and update statistics."""
        if not text:
            return
            
        # Update character distribution
        for char in text:
            self.stats['char_distribution'][char] += 1
            self.stats['total_chars'] += 1
            
            # Check for non-Geez characters
            if not self.is_geez_char(char) and char.strip() and not char.isspace():
                self.stats['non_geez_chars'].add(char)
    
    def analyze_examples(self):
        """Analyze all loaded examples."""
        logger.info("Analyzing examples...")
        
        for example in tqdm(self.examples, desc="Analyzing examples"):
            # Analyze input and target text
            self.analyze_text(example.get('input', ''))
            self.analyze_text(example.get('target', ''))
            
            # Update token and example lengths
            input_len = len(example.get('input', '').split())
            target_len = len(example.get('target', '').split())
            self.stats['token_lengths'].append((input_len, target_len))
            self.stats['example_lengths'].append(input_len + target_len)
    
    def calculate_statistics(self):
        """Calculate and log comprehensive statistics."""
        if not self.examples:
            logger.warning("No examples to analyze")
            return
            
        # Calculate average lengths
        total_input_tokens = sum(inp for inp, _ in self.stats['token_lengths'])
        total_target_tokens = sum(tgt for _, tgt in self.stats['token_lengths'])
        avg_input_len = total_input_tokens / len(self.examples)
        avg_target_len = total_target_tokens / len(self.examples)
        avg_example_len = sum(self.stats['example_lengths']) / len(self.examples) if self.stats['example_lengths'] else 0
        
        # Count Ge'ez characters
        geez_chars = [
            c for c in self.stats['char_distribution'] 
            if self.is_geez_char(c)
        ]
        self.stats['geez_char_count'] = len(geez_chars)
        
        # Print summary
        print("\n" + "="*80)
        print("TRAINING DATA ANALYSIS REPORT")
        print("="*80)
        
        print(f"\n📊 General Statistics")
        print(f"  Total examples: {self.stats['total_examples']:,}")
        print(f"  Total characters: {self.stats['total_chars']:,}")
        print(f"  Total tokens (input + target): {total_input_tokens + total_target_tokens:,}")
        print(f"  Average input length: {avg_input_len:.1f} tokens")
        print(f"  Average target length: {avg_target_len:.1f} tokens")
        print(f"  Average example length: {avg_example_len:.1f} tokens")
        print(f"  Unique characters: {len(self.stats['char_distribution']):,}")
        print(f"  Ge'ez characters: {self.stats['geez_char_count']}")
        
        # Print file statistics
        print("\n📂 File Statistics")
        for file_path, stats in self.stats['file_stats'].items():
            print(f"\n  File: {Path(file_path).name}")
            print(f"    Examples: {stats['total_examples']:,}")
            print(f"    Empty examples: {stats['empty_examples']:,}")
            print(f"    Malformed examples: {stats['malformed_examples']:,}")
            print(f"    Total characters: {stats['char_count']:,}")
            print(f"    Unique Ge'ez chars: {stats['geez_chars']}")
            print(f"    Non-Geez chars: {stats['non_geez_chars']}")
        
        if self.stats['non_geez_chars']:
            print(f"\n⚠️  Found {len(self.stats['non_geez_chars'])} non-Geez characters")
            print("  Sample:", " ".join(list(self.stats['non_geez_chars'])[:20]))
        
        if self.stats['empty_examples']:
            print(f"\n⚠️  Found {self.stats['empty_examples']} empty examples")
            
        if self.stats['malformed_examples']:
            print(f"\n⚠️  Found {self.stats['malformed_examples']} malformed examples")
        
        # Print most common characters
        print("\n🔠 Most Common Characters:")
        for char, count in self.stats['char_distribution'].most_common(20):
            char_name = unicodedata.name(char, 'UNKNOWN')
            print(f"  U+{ord(char):04X} {char} ({char_name}): {count:,}")
        
        # Print character coverage
        print("\n🔡 Character Coverage:")
        print(f"  Total unique characters: {len(self.stats['char_distribution']):,}")
        print(f"  Ge'ez characters: {self.stats['geez_char_count']}")
        print(f"  Non-Geez characters: {len(self.stats['non_geez_chars'])}")
        
        # Print example samples
        print("\n📝 Example Samples:")
        for i, example in enumerate(self.examples[:3]):
            print(f"\n  Example {i+1}:")
            print(f"  Input:  {example.get('input', '')[:200]}...")
            print(f"  Target: {example.get('target', '')[:200]}...")
    
    def run_analysis(self):
        """Run the complete analysis pipeline."""
        self.load_data()
        self.analyze_examples()
        self.calculate_statistics()

def main():
    # Define data paths
    data_dir = Path("data/processed/training_data")
    data_files = [
        data_dir / "train.jsonl",
        data_dir / "valid.jsonl"
    ]
    
    # Run analysis
    analyzer = TrainingDataAnalyzer(data_files)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
