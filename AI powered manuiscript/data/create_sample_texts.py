#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Sample Ge'ez Text Files

Generates sample Ge'ez text files for testing the data processing pipeline.
"""

import os
import random
from pathlib import Path

# Sample Ge'ez text paragraphs (from public domain sources)
SAMPLE_TEXTS = [
    """
    በሰማይ፡ ወበምድር፡ ወበገጸ፡ ምድር፡ ወበነገሩ፡ ኵሎ፡ ዘእንበለ፡ ይኩን።
    ወምድር፡ ትኩን፡ ትህቅ፡ ወባዳ፡ ወጽልመት፡ እስከ፡ ላዕሌሃ።
    ወመንፈሰ፡ አምላክ፡ ይነፍስ፡ እምበርሴት፡ ማይ፡ ወአስተብህሎ፡ ኵሎ።
    """,
    
    """
    ሰላም፡ ለአንተ፡ ወልደ፡ ሰብአ፡ ዘንተ፡ ወልድከ፡ ወውልድከ፡ ወውልደ፡ ወልድከ።
    ወይቤላ፡ ለእሙ፡ ሰላም፡ ለከ፡ እሙ፡ ወለከ፡ ሰላም።
    """,
    
    """
    ወኮነ፡ በዓመተ፡ አርብዓ፡ ወሰማንያ፡ ለአርብዓ፡ ዓመት፡ ለንጉሥነ፡ ዳዊት፡ ወለልደቱ፡ ሰሎሞን።
    ወኮነ፡ በዓመተ፡ ስምንት፡ ወሰማንያ፡ ለንጉሥነ፡ ዳዊት፡ ወለልደቱ፡ ሰሎሞን።
    """,
    
    """
    እግዚአብሔር፡ ለእግዚአብሔር፡ ነገረ፡ ንጉሥ፡ ለንጉሥ፡ ነገረ።
    እግዚአብሔር፡ ከመ፡ እመንበረ፡ ምድር፡ ወከመ፡ እምንትን፡ አንተ፡ ነበርከ።
    """,
    
    """
    እግዚአብሔር፡ አምላክ፡ እስራኤል፡ ንጹሕ፡ ነው።
    እግዚአብሔር፡ አምላክ፡ እስራኤል፡ ንጹሕ፡ ነው።
    """
]

def create_sample_files(output_dir: str, num_files: int = 10):
    """Create sample Ge'ez text files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i in range(1, num_files + 1):
        # Select random text samples
        num_paragraphs = random.randint(3, 8)
        selected_texts = random.choices(SAMPLE_TEXTS, k=num_paragraphs)
        
        # Create file content
        content = '\n\n'.join(selected_texts)
        
        # Write to file
        file_path = output_path / f'geez_sample_{i:03d}.txt'
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Created {file_path}")

def main():
    # Create sample files in the raw_texts directory
    create_sample_files('data/raw_texts', num_files=20)
    print("\nSample files created successfully!")
    print("You can now run: python data/local_data_processor.py")

if __name__ == "__main__":
    main()
