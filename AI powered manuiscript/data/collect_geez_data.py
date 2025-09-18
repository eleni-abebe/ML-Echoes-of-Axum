#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ge'ez Text Data Collection Script

Collects Ge'ez text data from various online sources to build a comprehensive dataset.
"""

import os
import re
import json
import time
import random
import logging
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Optional, Set

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

# Set a user agent to avoid being blocked
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

class GeEzDataCollector:
    """Collects Ge'ez text data from various online sources."""
    
    def __init__(self, output_dir: str = 'data/raw_texts'):
        """Initialize the data collector with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visited_urls: Set[str] = set()
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
    
    def save_text(self, text: str, source: str, title: str = None) -> str:
        """Save collected text to a file."""
        # Clean the title to create a valid filename
        if not title:
            title = str(int(time.time()))
        
        # Clean and format the filename
        clean_title = re.sub(r'[^\w\s-]', '', title).strip().lower()
        clean_title = re.sub(r'[-\s]+', '_', clean_title)
        
        # Ensure the filename is not too long
        if len(clean_title) > 50:
            clean_title = clean_title[:47] + '...'
        
        # Add source and timestamp
        filename = f"{source}_{clean_title}_{int(time.time())}.txt"
        filepath = self.output_dir / filename
        
        # Save the text
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        
        logger.info(f"Saved text to {filepath}")
        return str(filepath)
    
    def get_page_content(self, url: str, timeout: int = 10) -> Optional[str]:
        """Fetch the content of a web page with error handling."""
        try:
            response = self.session.get(url, timeout=timeout, headers=HEADERS)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.warning(f"Error fetching {url}: {str(e)}")
            return None
    
    def extract_text_from_html(self, html: str, base_url: str = '') -> str:
        """Extract clean text from HTML content."""
        if not html:
            return ""
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text from paragraphs and other text containers
        text_parts = []
        for element in soup.find_all(['p', 'div', 'article', 'section', 'main']):
            # Skip elements with little text (likely navigation, etc.)
            if len(element.get_text(strip=True)) < 50:
                continue
                
            # Get clean text
            element_text = element.get_text('\n', strip=True)
            text_parts.append(element_text)
        
        # If no text found in containers, fall back to body text
        if not text_parts:
            text = soup.get_text('\n', strip=True)
        else:
            text = '\n\n'.join(text_parts)
        
        # Clean up the text
        text = re.sub(r'\n{3,}', '\n\n', text)  # Remove excessive newlines
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        
        return text
    
    def collect_from_ethiopic_heritage_digital_library(self) -> List[str]:
        """Collect Ge'ez texts from the Ethiopic Heritage Digital Library."""
        base_url = "https://betamasaheft.eu/"
        logger.info(f"Collecting from Ethiopic Heritage Digital Library: {base_url}")
        
        saved_files = []
        
        # Example implementation - would need to be adapted to the actual site structure
        try:
            # Get the main page
            html = self.get_page_content(base_url)
            if not html:
                return saved_files
            
            # Extract text from the main page
            text = self.extract_text_from_html(html, base_url)
            if text:
                filepath = self.save_text(text, "ethiopic_heritage", "homepage")
                saved_files.append(filepath)
            
            # Here you would add code to navigate to specific texts and download them
            # This is a placeholder that would need to be implemented based on the site's structure
            
        except Exception as e:
            logger.error(f"Error collecting from Ethiopic Heritage Digital Library: {str(e)}")
        
        return saved_files
    
    def collect_from_wikisource(self) -> List[str]:
        """Collect Ge'ez texts from Wikisource."""
        base_url = "https://wikisource.org/wiki/"
        search_url = "https://wikisource.org/w/api.php"
        
        logger.info("Collecting from Wikisource...")
        
        saved_files = []
        
        try:
            # Search for Ge'ez texts
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': 'intitle:Ge\'ez OR intitle:Ge'ez OR intitle:ግዕዝ',
                'srlimit': 50,
                'srprop': ''
            }
            
            response = self.session.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Process search results
            if 'query' in data and 'search' in data['query']:
                for result in data['query']['search']:
                    page_title = result['title']
                    page_url = urljoin(base_url, page_title.replace(' ', '_'))
                    
                    # Skip if we've already visited this URL
                    if page_url in self.visited_urls:
                        continue
                    
                    # Get the page content
                    page_html = self.get_page_content(page_url)
                    if not page_html:
                        continue
                    
                    # Extract text
                    text = self.extract_text_from_html(page_html, page_url)
                    if text and len(text) > 100:  # Only save if we have substantial text
                        filepath = self.save_text(text, "wikisource", page_title)
                        saved_files.append(filepath)
                    
                    # Mark as visited
                    self.visited_urls.add(page_url)
                    
                    # Be polite with a delay
                    time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error collecting from Wikisource: {str(e)}")
        
        return saved_files
    
    def collect_from_github_gists(self) -> List[str]:
        """Collect Ge'ez texts from GitHub Gists."""
        base_url = "https://gist.github.com"
        search_url = "https://api.github.com/search/code"
        
        logger.info("Searching for Ge'ez texts in GitHub Gists...")
        
        saved_files = []
        
        try:
            # Search for Ge'ez texts in public gists
            params = {
                'q': 'language:Amharic OR language:Ge\'ez OR language:Ethiopic',
                'per_page': 30
            }
            
            # GitHub API requires a user agent
            headers = HEADERS.copy()
            if 'GitHub' not in headers['User-Agent']:
                headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            
            response = self.session.get(search_url, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            
            results = response.json()
            
            if 'items' in results:
                for item in results['items']:
                    try:
                        gist_url = item['html_url']
                        
                        # Skip if we've already visited this URL
                        if gist_url in self.visited_urls:
                            continue
                        
                        # Get the raw content URL
                        raw_url = item['html_url'].replace('github.com', 'raw.githubusercontent.com').replace('/blob', '')
                        
                        # Get the content
                        content = self.get_page_content(raw_url)
                        if not content:
                            continue
                        
                        # Save the content
                        title = item['name'] or f"gist_{item['id']}"
                        filepath = self.save_text(content, "github_gist", title)
                        saved_files.append(filepath)
                        
                        # Mark as visited
                        self.visited_urls.add(gist_url)
                        
                        # Be polite with a delay
                        time.sleep(2)
                        
                    except Exception as e:
                        logger.warning(f"Error processing GitHub gist: {str(e)}")
                        continue
        
        except Exception as e:
            logger.error(f"Error collecting from GitHub Gists: {str(e)}")
        
        return saved_files
    
    def collect_from_local_sources(self, local_dirs: List[str]) -> List[str]:
        """Collect Ge'ez texts from local directories."""
        logger.info("Collecting from local sources...")
        
        saved_files = []
        
        for dir_path in local_dirs:
            dir_path = Path(dir_path)
            if not dir_path.exists() or not dir_path.is_dir():
                logger.warning(f"Directory not found: {dir_path}")
                continue
                
            # Look for text files
            for ext in ['*.txt', '*.md', '*.json', '*.jsonl']:
                for file_path in dir_path.rglob(ext):
                    try:
                        # Skip files that are too small (likely not meaningful content)
                        if file_path.stat().st_size < 100:  # Less than 100 bytes
                            continue
                            
                        # Read the file
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        if not content.strip():
                            continue
                            
                        # Save to our output directory
                        rel_path = file_path.relative_to(dir_path)
                        output_path = self.output_dir / f"local_{rel_path.name}"
                        
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                            
                        saved_files.append(str(output_path))
                        logger.info(f"Copied local file: {output_path}")
                        
                    except Exception as e:
                        logger.warning(f"Error processing {file_path}: {str(e)}")
        
        return saved_files

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect Ge'ez text data from various sources")
    parser.add_argument('--output-dir', type=str, default='data/raw_texts',
                      help='Directory to save collected text files')
    parser.add_argument('--sources', type=str, nargs='+', 
                      default=['wikisource', 'github', 'ethiopic'],
                      help='Sources to collect from (wikisource, github, ethiopic, local)')
    parser.add_argument('--local-dirs', type=str, nargs='+',
                      default=[],
                      help='Local directories to scan for Ge\'ez text files')
    
    args = parser.parse_args()
    
    # Initialize the collector
    collector = GeEzDataCollector(output_dir=args.output_dir)
    
    # Track all saved files
    all_saved_files = []
    
    # Collect from specified sources
    try:
        if 'ethiopic' in args.sources:
            files = collector.collect_from_ethiopic_heritage_digital_library()
            all_saved_files.extend(files)
            
        if 'wikisource' in args.sources:
            files = collector.collect_from_wikisource()
            all_saved_files.extend(files)
            
        if 'github' in args.sources:
            files = collector.collect_from_github_gists()
            all_saved_files.extend(files)
            
        if 'local' in args.sources and args.local_dirs:
            files = collector.collect_from_local_sources(args.local_dirs)
            all_saved_files.extend(files)
            
    except KeyboardInterrupt:
        logger.info("Data collection interrupted by user")
    except Exception as e:
        logger.error(f"Error during data collection: {str(e)}")
    
    # Print summary
    logger.info(f"\nData collection complete. Saved {len(all_saved_files)} files to {args.output_dir}")
    
    # Save list of collected files
    if all_saved_files:
        list_file = Path(args.output_dir) / "collected_files.txt"
        with open(list_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(all_saved_files))
        logger.info(f"List of collected files saved to {list_file}")

if __name__ == "__main__":
    main()
