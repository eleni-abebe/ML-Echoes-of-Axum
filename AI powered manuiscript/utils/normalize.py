#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text normalization utilities for Ge'ez script.
Handles Unicode normalization, whitespace cleaning, and other text preprocessing.
"""

import re
import unicodedata
from pathlib import Path
from typing import Optional, Union

# Define Ge'ez Unicode block ranges
GEEZ_RANGE = (0x1200, 0x137F)  # Base Ge'ez block
GEEZ_EXTENDED_RANGE = (0xAB00, 0xAB2F)  # Extended Ge'ez block (Ethiopic Extended)
GEEZ_EXTENDED_A_RANGE = (0x1E7E0, 0x1E7FF)  # Extended Ge'ez block (Ethiopic Extended-A)

def is_geez_char(char: str) -> bool:
    """Check if a character is in the Ge'ez Unicode blocks."""
    cp = ord(char[0])  # Only check first character if string is passed
    return (GEEZ_RANGE[0] <= cp <= GEEZ_RANGE[1]) or \
           (GEEZ_EXTENDED_RANGE[0] <= cp <= GEEZ_EXTENDED_RANGE[1]) or \
           (GEEZ_EXTENDED_A_RANGE[0] <= cp <= GEEZ_EXTENDED_A_RANGE[1])

def normalize_text(text: str, 
                  normalize_unicode: str = "NFC", 
                  remove_control_chars: bool = True,
                  normalize_whitespace: bool = True) -> str:
    """
    Normalize Ge'ez text by applying various normalization steps.
    
    Args:
        text: Input text to normalize
        normalize_unicode: Unicode normalization form ('NFC', 'NFD', 'NFKC', 'NFKD')
        remove_control_chars: Whether to remove control characters
        normalize_whitespace: Whether to normalize whitespace
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Apply Unicode normalization
    text = unicodedata.normalize(normalize_unicode, text)
    
    if remove_control_chars:
        # Remove control characters except for common whitespace
        text = ''.join(
            char for char in text 
            if char.isprintable() or char.isspace()
        )
        
        # Remove other common unwanted characters
        text = text.replace('\u200b', '')  # Zero-width space
        text = text.replace('\ufeff', '')  # Byte order mark
        text = text.replace('\u00ad', '')  # Soft hyphen
    
    if normalize_whitespace:
        # Replace various whitespace with a single space
        text = re.sub(r'[\s\u200b\u200c\u200d\u2060\ufeff]+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
    
    return text

def clean_file(input_path: Union[str, Path], 
              output_path: Optional[Union[str, Path]] = None,
              **kwargs) -> str:
    """
    Clean and normalize a text file containing Ge'ez script.
    
    Args:
        input_path: Path to input file
        output_path: Path to output file (if None, will use input_path with _clean suffix)
        **kwargs: Additional arguments to pass to normalize_text()
        
    Returns:
        Path to the cleaned file
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Read input file with error handling for encoding
    try:
        text = input_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        # Try with error handling if default encoding fails
        text = input_path.read_text(encoding='utf-8', errors='replace')
    
    # Normalize the text
    normalized_text = normalize_text(text, **kwargs)
    
    # Determine output path if not provided
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_clean{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write output file
    output_path.write_text(normalized_text, encoding='utf-8')
    
    return str(output_path)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Normalize Ge\'ez text files')
    parser.add_argument('input_file', help='Input text file')
    parser.add_argument('output_file', nargs='?', default=None, 
                       help='Output text file (default: <input_file>_clean.<ext>)')
    parser.add_argument('--unicode', default='NFC', 
                       choices=['NFC', 'NFD', 'NFKC', 'NFKD'],
                       help='Unicode normalization form (default: NFC)')
    parser.add_argument('--keep-control', action='store_false', dest='remove_control',
                      help='Keep control characters')
    parser.add_argument('--no-whitespace', action='store_false', dest='normalize_whitespace',
                      help='Do not normalize whitespace')
    
    args = parser.parse_args()
    
    output_file = clean_file(
        args.input_file,
        args.output_file,
        normalize_unicode=args.unicode,
        remove_control_chars=args.remove_control,
        normalize_whitespace=args.normalize_whitespace
    )
    
    print(f"Cleaned file saved to: {output_file}")


if __name__ == "__main__":
    main()
