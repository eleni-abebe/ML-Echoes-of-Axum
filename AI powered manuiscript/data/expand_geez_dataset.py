#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ge'ez Text Data Collection and Augmentation Pipeline

This script:
1. Scrapes Ge'ez text from various online sources
2. Augments existing data with variations
3. Preprocesses and normalizes the text
4. Saves to training-ready format
"""

import os
import re
import json
import random
import requests
import logging
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import unicodedata
from typing import List, Dict, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GeEzDataCollector:
    """Collects Ge'ez text data from various online sources."""
    
    def __init__(self, output_dir: str = 'data/raw'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.geez_unicode_ranges = [
            (0x1200, 0x137F),  # Ethiopic
            (0x1380, 0x139F),  # Ethiopic Supplement
            (0x2D80, 0x2DDF),  # Ethiopic Extended
            (0xAB00, 0xAB2F)   # Ethiopic Extended-A
        ]
    
    def is_geez_text(self, text: str, threshold: float = 0.7) -> bool:
        """Check if text contains sufficient Ge'ez characters."""
        if not text.strip():
            return False
            
        total_chars = len(text)
        if total_chars == 0:
            return False
            
        geez_count = sum(1 for c in text if self._is_geez_char(c))
        return (geez_count / total_chars) >= threshold
    
    def _is_geez_char(self, char: str) -> bool:
        """Check if character is in Ge'ez Unicode ranges."""
        cp = ord(char)
        return any(start <= cp <= end for start, end in self.geez_unicode_ranges)
    
    def normalize_text(self, text: str) -> str:
        """Normalize Ge'ez text."""
        # Normalize Unicode
        text = unicodedata.normalize('NFC', text)
        
        # Standardize whitespace and newlines
        text = ' '.join(text.split())
        
        # Standardize Ge'ez word space (U+1361)
        text = re.sub(r'[\s\u1361]+', '፡', text)
        
        # Remove control characters
        text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C' or c in '\n\r\t')
        
        return text.strip()
    
    def scrape_geez_website(self, base_url: str, max_pages: int = 10) -> List[str]:
        """Scrape Ge'ez text from a website."""
        logger.info(f"Scraping {base_url}")
        
        try:
            response = self.session.get(base_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract and process text
            paragraphs = []
            for p in soup.find_all(['p', 'div', 'article']):
                text = self.normalize_text(p.get_text())
                if self.is_geez_text(text):
                    paragraphs.append(text)
            
            return paragraphs
            
        except Exception as e:
            logger.error(f"Error scraping {base_url}: {str(e)}")
            return []
    
    def collect_from_sources(self, sources: List[Dict], max_workers: int = 5) -> List[str]:
        """Collect text from multiple sources in parallel."""
        all_paragraphs = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(self.scrape_geez_website, source['url']): source
                for source in sources
            }
            
            for future in tqdm(as_completed(future_to_url), total=len(future_to_url), desc="Scraping sources"):
                source = future_to_url[future]
                try:
                    paragraphs = future.result()
                    all_paragraphs.extend(paragraphs)
                    logger.info(f"Collected {len(paragraphs)} paragraphs from {source['url']}")
                except Exception as e:
                    logger.error(f"Error processing {source['url']}: {str(e)}")
        
        return all_paragraphs


class GeEzDataAugmenter:
    """Augments existing Ge'ez text data with variations."""
    
    def __init__(self):
        # Common Ge'ez words and variations
        self.common_words = {
            'ወ': ['እና', 'እንዲሁም', 'እንዲህ'],
            'በ': ['በውስጥ', 'በላይ', 'በታች'],
            'እንደ': ['እንደምን', 'እንደዚህ', 'እንደዛ'],
            'ነው': ['ናቸው', 'ነበሩ', 'ነበር']
        }
        
    def synonym_replacement(self, text: str, n: int = 2) -> str:
        """Replace words with synonyms."""
        words = text.split()
        new_words = words.copy()
        
        for _ in range(min(n, len(words))):
            word = random.choice(words)
            if word in self.common_words:
                new_word = random.choice(self.common_words[word])
                new_words = [new_word if w == word else w for w in new_words]
        
        return ' '.join(new_words)
    
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """Randomly delete words with probability p."""
        words = text.split()
        if len(words) == 1:
            return text
            
        new_words = [word for word in words if random.random() > p]
        return ' '.join(new_words) if new_words else random.choice(words)
    
    def random_swap(self, text: str, n: int = 3) -> str:
        """Randomly swap two words n times."""
        words = text.split()
        if len(words) < 2:
            return text
            
        for _ in range(min(n, len(words) // 2)):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
            
        return ' '.join(words)
    
    def augment_text(self, text: str, num_augmentations: int = 3) -> List[str]:
        """Generate augmented versions of the text."""
        augmentations = []
        
        for _ in range(num_augmentations):
            # Choose an augmentation method
            method = random.choice([
                lambda x: self.synonym_replacement(x),
                lambda x: self.random_deletion(x),
                lambda x: self.random_swap(x)
            ])
            
            augmented = method(text)
            if augmented != text:  # Only add if different
                augmentations.append(augmented)
        
        return augmentations


def create_training_examples(texts: List[str], min_length: int = 20, max_length: int = 200) -> List[Dict]:
    """Create training examples from raw text."""
    examples = []
    
    for text in texts:
        # Split into sentences (simplified for Ge'ez)
        sentences = re.split(r'[።!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Create examples from sentences
        for i in range(len(sentences) - 1):
            if min_length <= len(sentences[i]) <= max_length and min_length <= len(sentences[i+1]) <= max_length:
                examples.append({
                    'input': sentences[i],
                    'target': sentences[i+1]
                })
    
    return examples


def main():
    # Initialize components
    collector = GeEzDataCollector('data/raw')
    augmenter = GeEzDataAugmenter()
    
    # Define sources for Ge'ez text
    sources = [
        {'url': 'https://www.ethiopic.com/collation/Ethiopian.html'},
        {'url': 'http://www.madote.com/'},
        {'url': 'https://www.ethiopiaobserver.com/category/culture/'},
        {'url': 'http://www.ethiopianreporter.com/'},
        {'url': 'https://www.zehabesha.com/'}
    ]
    
    # Step 1: Collect data from online sources
    logger.info("Starting data collection...")
    raw_texts = collector.collect_from_sources(sources)
    
    # Step 2: Augment the data
    logger.info("Augmenting data...")
    augmented_texts = []
    for text in tqdm(raw_texts, desc="Augmenting texts"):
        augmented_texts.append(text)
        augmented_texts.extend(augmenter.augment_text(text))
    
    # Step 3: Create training examples
    logger.info("Creating training examples...")
    examples = create_training_examples(augmented_texts)
    
    # Step 4: Save to file
    output_file = collector.output_dir.parent / 'processed' / 'augmented_training_data.jsonl'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved {len(examples)} training examples to {output_file}")


if __name__ == "__main__":
    main()
