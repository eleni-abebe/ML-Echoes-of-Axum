#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert cleaned text files into a JSONL corpus for training.

This script processes all text files in a directory and converts them into
a single JSONL file where each line is a JSON object representing a text chunk.
"""

import os
import re
import json
import logging
import argparse
from pathlib import Path
from typing import Iterator, Dict, List, Optional

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using Ge'ez punctuation."""
    # First, normalize all whitespace to single spaces
    text = ' '.join(text.split())
    
    # Common Ge'ez sentence terminators and separators
    sentence_enders = [
        '።',  # Full stop (period)
        '። ',  # With space
        '።\n',  # With newline
        '።\r\n',  # With Windows newline
        '፤',   # Semicolon
        '፤ ',  # With space
        '፥',   # Colon
        '፥ ',  # With space
        '፦',   # Preface colon
        '፦ ',  # With space
        '፧',   # Question mark
        '፧ ',  # With space
        '፨',   # Paragraph separator
        '፨ ',  # With space
        '፠',   # Section mark
        '፠ ',  # With space
    ]
    
    # Replace all sentence enders with a special token
    for ender in sentence_enders:
        text = text.replace(ender, '።' + '\n')
    
    # Split on the special token
    sentences = [s.strip() for s in text.split('።') if s.strip()]
    
    # Add the sentence terminator back to each sentence
    sentences = [s + '።' for s in sentences if s]
    
    # Log the first few sentences for debugging
    if sentences:
        logger.debug(f"First few sentences from text: {sentences[:3]}")
    else:
        logger.warning("No sentences found after splitting")
    
    return sentences


def chunk_text(text: str, chunk_size: int = 5) -> Iterator[str]:
    """Split text into chunks of approximately chunk_size sentences."""
    sentences = split_into_sentences(text)
    
    for i in range(0, len(sentences), chunk_size):
        chunk = ' '.join(sentences[i:i+chunk_size])
        yield chunk


def process_file(file_path: Path, chunk_size: int) -> Iterator[Dict]:
    """Process a single text file and yield document chunks."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        if not text:
            logger.warning(f"Empty file: {file_path}")
            return
            
        logger.debug(f"Raw text from {file_path}: {text[:200]}...")  # Log first 200 chars
        
        # Generate chunks from the text
        chunks = list(chunk_text(text, chunk_size))
        logger.info(f"Generated {len(chunks)} chunks from {file_path}")
        
        for chunk_id, chunk in enumerate(chunks):
            if not chunk.strip():
                logger.debug(f"Empty chunk {chunk_id} in {file_path}")
                continue
                
            chunk_data = {
                'id': f"{file_path.stem}_{chunk_id:04d}",
                'text': chunk,
                'source': file_path.name,
                'chunk_id': chunk_id,
                'language': 'gez',
                'metadata': {
                    'source_file': str(file_path.name),
                    'chunk_size': chunk_size,
                }
            }
            logger.debug(f"Yielding chunk {chunk_id}: {chunk[:100]}...")  # First 100 chars
            yield chunk_data
            
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}", exc_info=True)


def process_directory(
    input_dir: Path,
    output_file: Path,
    chunk_size: int = 5,
    min_length: int = 20,
    max_length: int = 1000,
) -> int:
    """Process all text files in a directory and save as JSONL."""
    # Get all .txt files in the input directory
    text_files = list(input_dir.glob('*.txt'))
    if not text_files:
        logger.warning(f"No .txt files found in {input_dir}")
        return 0
    
    logger.info(f"Processing {len(text_files)} text files from {input_dir}")
    
    total_chunks = 0
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for file_path in text_files:
            logger.info(f"Processing {file_path.name}...")
            file_chunks = 0
            
            for chunk in process_file(file_path, chunk_size):
                # Filter by length
                chunk_length = len(chunk['text'])
                if min_length <= chunk_length <= max_length:
                    out_f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                    total_chunks += 1
                    file_chunks += 1
            
            logger.info(f"  - Processed {file_chunks} chunks from {file_path.name}")
    
    logger.info(f"Processed {total_chunks} chunks from {len(text_files)} files")
    logger.info(f"Output saved to {output_file}")
    
    if total_chunks == 0:
        logger.warning("No chunks were processed. Check input files and parameters.")
    
    return total_chunks


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Build a JSONL corpus from text files.')
    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='Directory containing text files to process'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=5,
        help='Number of sentences per chunk (default: 5)'
    )
    parser.add_argument(
        '--min-length',
        type=int,
        default=20,
        help='Minimum character length for a chunk (default: 20)'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=1000,
        help='Maximum character length for a chunk (default: 1000)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    return args


def main() -> int:
    """Main function."""
    args = parse_args()
    
    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Process the directory
    total_chunks = process_directory(
        input_dir=args.input_dir,
        output_file=args.output,
        chunk_size=args.chunk_size,
        min_length=args.min_length,
        max_length=args.max_length
    )
    
    if total_chunks == 0:
        logger.warning("No chunks were processed. Check your input files and parameters.")
        return 1
        
    return 0


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    exit(main())
