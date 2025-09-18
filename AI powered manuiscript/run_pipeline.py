#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Echoes of Axum - Training Pipeline

This script automates the entire process of preparing data and training models
for the Echoes of Axum project.
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

# Project directories
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
CLEAN_DIR = DATA_DIR / 'clean'
PROCESSED_DIR = DATA_DIR / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
LOGS_DIR = PROJECT_ROOT / 'logs'

# Create necessary directories
for directory in [RAW_DIR, CLEAN_DIR, PROCESSED_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Configuration
CONFIG = {
    'data': {
        'sample_data_urls': [
            'https://raw.githubusercontent.com/geezorg/ethiopic-corpus/master/texts/geez/ezra.txt',
            'https://raw.githubusercontent.com/geezorg/ethiopic-corpus/master/texts/geez/genesis.txt',
        ],
        'chunk_size': 5,  # sentences per chunk
        'min_chunk_length': 20,  # minimum characters per chunk
    },
    'training': {
        'model_name': 'google/byt5-small',
        'batch_size': 8,
        'num_epochs': 3,
        'learning_rate': 3e-4,
        'max_seq_length': 512,
        'warmup_steps': 1000,
        'save_steps': 1000,
        'logging_steps': 100,
    }
}


def run_command(command: List[str], cwd: Optional[Path] = None) -> bool:
    """Run a shell command and return True if successful."""
    try:
        logger.info(f"Running: {' '.join(command)}")
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            check=True,
            text=True,
            capture_output=True
        )
        if result.stdout:
            logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with error: {e}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr}")
        return False


def download_sample_data() -> bool:
    """Download sample Ge'ez text data."""
    logger.info("Downloading sample Ge'ez text data...")
    
    for url in CONFIG['data']['sample_data_urls']:
        filename = url.split('/')[-1]
        output_path = RAW_DIR / filename
        
        if not output_path.exists():
            command = ['curl', '-L', '-o', str(output_path), url]
            if not run_command(command):
                return False
    
    return True


def normalize_text() -> bool:
    """Normalize the raw text files."""
    logger.info("Normalizing text files...")
    
    # Ensure the clean directory exists
    CLEAN_DIR.mkdir(exist_ok=True)
    
    for input_file in RAW_DIR.glob('*.txt'):
        output_file = CLEAN_DIR / input_file.name
        command = [
            "python", "utils/normalize.py",
            str(input_file),
            str(output_file),
            "--unicode", "NFC"
        ]
        
        if True:
            command.append("--no-whitespace")
        else:
            command.extend(["--no-whitespace", "false"])
            
        if True:
            command.append("--keep-control")
        
        logger.info(f"Running: {' '.join(command)}")
        
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with error: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
    
    return True


def build_corpus() -> bool:
    """Build the JSONL corpus from cleaned text files."""
    logger.info("Building JSONL corpus...")
    
    output_file = PROCESSED_DIR / 'corpus.jsonl'
    command = [
        'python', 'data/make_corpus_jsonl.py',
        '--input-dir', str(CLEAN_DIR),
        '--output', str(output_file),
        '--min-length', '20',
        '--max-length', '1000',
        '--chunk-size', str(CONFIG['data']['chunk_size'])
    ]
    
    return run_command(command)


def prepare_training_data() -> bool:
    """Prepare training data with span corruption."""
    logger.info("Preparing training data with span corruption...")
    
    input_file = PROCESSED_DIR / 'corpus.jsonl'
    output_dir = PROCESSED_DIR / 'training_data'
    output_dir.mkdir(exist_ok=True)
    
    command = [
        'python', 'training/make_spans.py',
        '--input', str(input_file),
        '--output-dir', str(output_dir),
        '--mask-probability', '0.15',
        '--max-span-length', '3',
        '--num-spans', '3',
        '--seed', '42'
    ]
    
    return run_command(command)


def train_model() -> bool:
    """Train the ByT5 model for text restoration."""
    logger.info("Training ByT5 model for text restoration...")
    
    train_file = PROCESSED_DIR / 'training_data' / 'train.jsonl'
    eval_file = PROCESSED_DIR / 'training_data' / 'valid.jsonl'
    output_dir = MODELS_DIR / 'byt5_geez_span'
    
    command = [
        'python', 'training/train_byt5.py',
        '--model_name_or_path', CONFIG['training']['model_name'],
        '--train_file', str(train_file),
        '--validation_file', str(eval_file),
        '--output_dir', str(output_dir),
        '--num_train_epochs', str(CONFIG['training']['num_epochs']),
        '--per_device_train_batch_size', str(CONFIG['training']['batch_size']),
        '--per_device_eval_batch_size', str(CONFIG['training']['batch_size']),
        '--learning_rate', str(CONFIG['training']['learning_rate']),
        '--max_seq_length', str(CONFIG['training']['max_seq_length']),
        '--warmup_steps', str(CONFIG['training']['warmup_steps']),
        '--save_steps', str(CONFIG['training']['save_steps']),
        '--logging_steps', str(CONFIG['training']['logging_steps']),
        '--evaluation_strategy', 'steps',
        '--eval_steps', '1000',
        '--save_total_limit', '3',
        '--load_best_model_at_end',
        '--metric_for_best_model', 'loss',
        '--greater_is_better', 'False',
        '--report_to', 'tensorboard',
        '--logging_dir', str(LOGS_DIR / 'tensorboard'),
        '--overwrite_output_dir',
        '--do_train',
        '--do_eval',
    ]
    
    return run_command(command)


def main():
    """Run the entire pipeline."""
    logger.info("Starting Echoes of Axum training pipeline...")
    
    try:
        # Step 1: Download sample data
        logger.info("=== STEP 1: Downloading sample data ===")
        if not download_sample_data():
            raise RuntimeError("Failed to download sample data")
        
        # Step 2: Normalize text
        logger.info("\n=== STEP 2: Normalizing text ===")
        if not normalize_text():
            raise RuntimeError("Failed to normalize text")
        
        # Step 3: Build corpus
        logger.info("\n=== STEP 3: Building corpus ===")
        if not build_corpus():
            raise RuntimeError("Failed to build corpus")
        
        # Step 4: Prepare training data
        logger.info("\n=== STEP 4: Preparing training data ===")
        if not prepare_training_data():
            raise RuntimeError("Failed to prepare training data")
        
        # Step 5: Train model
        logger.info("\n=== STEP 5: Training model ===")
        if not train_model():
            raise RuntimeError("Model training failed")
        
        logger.info("\n=== Pipeline completed successfully! ===")
        logger.info(f"Model saved to: {MODELS_DIR / 'byt5_geez_span'}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
