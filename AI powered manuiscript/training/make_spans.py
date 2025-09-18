#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create training data with corrupted spans for text restoration.

This script takes a JSONL corpus and generates training examples by
introducing artificial gaps in the text that the model will learn to fill.
"""

import os
import re
import json
import random
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Special tokens
MASK_TOKEN = "<extra_id_0>"
EOS_TOKEN = "</s>"


def mask_random_span(text: str, mask_prob: float = 0.15, max_span_length: int = 3) -> Tuple[str, str, int]:
    """
    Mask a random span in the text.
    
    Args:
        text: Input text
        mask_prob: Probability of masking a token
        max_span_length: Maximum length of a masked span
        
    Returns:
        Tuple of (masked_text, masked_span, num_masked_tokens)
    """
    # Convert to list of characters for processing
    chars = list(text)
    n = len(chars)
    
    # Don't mask if text is too short
    if n < 5:
        return text, "", 0
    
    # Find all possible span start positions
    possible_starts = []
    for i in range(n - 1):
        # Don't mask whitespace or punctuation at start
        if not chars[i].isspace() and not chars[i] in ".,;:!?።፤፣፡":
            possible_starts.append(i)
    
    if not possible_starts:
        return text, "", 0
    
    # Decide whether to mask based on probability
    if random.random() > mask_prob and len(possible_starts) > 1:
        return text, "", 0
    
    # Choose a random start position
    start = random.choice(possible_starts)
    
    # Determine span length (1 to max_span_length)
    max_possible_length = min(max_span_length, n - start)
    span_length = random.randint(1, max_possible_length)
    
    # Extract the span to be masked
    masked_span = ''.join(chars[start:start+span_length])
    
    # Create the masked text
    masked_text = (
        ''.join(chars[:start]) + 
        MASK_TOKEN + 
        ''.join(chars[start+span_length:])
    )
    
    return masked_text, masked_span, span_length


def process_example(
    example: Dict, 
    mask_prob: float = 0.15, 
    max_span_length: int = 3,
    num_spans: int = 3
) -> Iterator[Dict]:
    """
    Process a single example to create training data with masked spans.
    
    Args:
        example: Input example with 'text' field
        mask_prob: Probability of masking a token
        max_span_length: Maximum length of a masked span
        num_spans: Number of spans to mask per example
        
    Yields:
        Dict with 'input' (masked text) and 'target' (original text with masked spans)
    """
    text = example.get('text', '').strip()
    if not text:
        return
    
    # Create multiple masked versions of the text
    for _ in range(num_spans):
        masked_text, masked_span, span_length = mask_random_span(
            text, mask_prob, max_span_length
        )
        
        if not masked_span:  # No span was masked
            continue
            
        yield {
            'id': f"{example.get('id', '')}_{_}",
            'input': masked_text,
            'target': text,  # Original text is the target
            'masked_span': masked_span,
            'span_length': span_length,
            'source': example.get('source', ''),
            'metadata': {
                **example.get('metadata', {}),
                'mask_prob': mask_prob,
                'max_span_length': max_span_length,
            }
        }


def process_corpus(
    input_file: Path,
    output_dir: Path,
    mask_prob: float = 0.15,
    max_span_length: int = 3,
    num_spans: int = 3,
    train_ratio: float = 0.9,
    seed: int = 42
) -> None:
    """
    Process a corpus file to create training data with masked spans.
    
    Args:
        input_file: Path to input JSONL file
        output_dir: Directory to save output files
        mask_prob: Probability of masking a token
        max_span_length: Maximum length of a masked span
        num_spans: Number of spans to mask per example
        train_ratio: Ratio of training examples
        seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output files
    train_file = output_dir / 'train.jsonl'
    valid_file = output_dir / 'valid.jsonl'
    
    # Count total examples for progress reporting
    total_examples = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        total_examples = sum(1 for _ in f)
    
    if total_examples == 0:
        logger.error(f"No examples found in {input_file}")
        return
    
    logger.info(f"Processing {total_examples} examples from {input_file}")
    
    # Process the corpus
    train_count = 0
    valid_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(train_file, 'w', encoding='utf-8') as f_train, \
         open(valid_file, 'w', encoding='utf-8') as f_valid:
        
        for i, line in enumerate(f_in):
            try:
                example = json.loads(line)
                
                # Decide if this example goes to train or validation
                is_train = random.random() < train_ratio
                
                # Generate masked examples
                for masked_example in process_example(
                    example, mask_prob, max_span_length, num_spans
                ):
                    output_line = json.dumps(masked_example, ensure_ascii=False)
                    
                    if is_train:
                        f_train.write(output_line + '\n')
                        train_count += 1
                    else:
                        f_valid.write(output_line + '\n')
                        valid_count += 1
                
                # Log progress
                if (i + 1) % 100 == 0 or (i + 1) == total_examples:
                    logger.info(f"Processed {i+1}/{total_examples} examples")
                    
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON on line {i+1}: {e}")
            except Exception as e:
                logger.error(f"Error processing example {i+1}: {e}", exc_info=True)
    
    logger.info(f"Generated {train_count} training examples and {valid_count} validation examples")
    logger.info(f"Training data saved to: {train_file}")
    logger.info(f"Validation data saved to: {valid_file}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create training data with masked spans.')
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input JSONL file with text examples'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Directory to save output files'
    )
    parser.add_argument(
        '--mask-probability',
        type=float,
        default=0.15,
        help='Probability of masking a token (default: 0.15)'
    )
    parser.add_argument(
        '--max-span-length',
        type=int,
        default=3,
        help='Maximum length of a masked span (default: 3)'
    )
    parser.add_argument(
        '--num-spans',
        type=int,
        default=3,
        help='Number of spans to mask per example (default: 3)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.9,
        help='Ratio of training examples (default: 0.9)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Validate input file
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    # Process the corpus
    process_corpus(
        input_file=args.input,
        output_dir=args.output_dir,
        mask_prob=args.mask_probability,
        max_span_length=args.max_span_length,
        num_spans=args.num_spans,
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
