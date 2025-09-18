#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Ge'ez text restoration functionality.
"""

import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent))

from app.restore import GeEzTextRestorer

def test_restoration():
    # Initialize the restorer with the correct model path
    model_path = "models/geez_t5"
    print(f"Loading model from: {model_path}")
    
    try:
        restorer = GeEzTextRestorer(
            model_path=model_path,
            num_beams=5,
            max_length=512,
            num_return_sequences=3
        )
        
        # Test with a sample Ge'ez text with missing spans
        test_text = "ይህ የሙከራ ጽሑፍ ነው። [MISSING] ይህ የማሟያ ጽሑፍ ነው።"
        
        print(f"\nTesting with input text:\n{test_text}\n")
        
        # Get restorations
        results = restorer.restore_text(test_text)
        
        print("\nRestoration results:")
        for i, (text, confidence) in enumerate(results, 1):
            print(f"\nOption {i} (Confidence: {confidence:.2f}):")
            print(f"{text}")
            
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_restoration()
