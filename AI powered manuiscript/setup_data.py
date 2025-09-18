#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for Ge'ez manuscript data.
"""

import os
import sys
from pathlib import Path
import shutil
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories if they don't exist."""
    base_dir = Path(__file__).parent
    dirs = [
        'data/raw',
        'data/clean',
        'data/processed',
        'models',
        'logs'
    ]
    
    for dir_path in dirs:
        full_path = base_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {full_path}")

def download_sample_data():
    """Download sample Ge'ez text data if it doesn't exist."""
    from data.download_sample_data import main as download_data
    
    logger.info("Checking for sample data...")
    data_dir = Path("data/raw")
    if not any(data_dir.glob("*.txt")):
        logger.info("Downloading sample data...")
        download_data()
    else:
        logger.info("Sample data already exists.")

def process_data():
    """Process the raw data into the required format."""
    from data.local_data_processor import process_directory
    
    raw_dir = Path("data/raw")
    clean_dir = Path("data/clean")
    
    if not any(clean_dir.glob("*.txt")) and any(raw_dir.glob("*.txt")):
        logger.info("Processing raw data...")
        process_directory(str(raw_dir), str(clean_dir))
    else:
        logger.info("Data already processed or no raw data found.")

def main():
    """Main function to set up the data directory structure and download data."""
    try:
        logger.info("Setting up directories...")
        setup_directories()
        
        logger.info("Checking data...")
        download_sample_data()
        process_data()
        
        logger.info("Setup completed successfully!")
    except Exception as e:
        logger.error(f"Error during setup: {str(e)}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
