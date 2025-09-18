#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Ge'ez Text Restoration Model

Loads the trained model and runs inference on sample inputs.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
import json
from pathlib import Path

def load_model(model_dir):
    """Load the trained model and tokenizer."""
    print(f"Loading model from {model_dir}")
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    
    return model, tokenizer, device

def generate_text(model, tokenizer, input_text, device, max_length=50):
    """Generate text using the model."""
    # Encode the input with the task prefix
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    
    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
    
    # Decode and return the output
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def run_tests(model_dir, test_cases=None):
    """Run test cases on the model."""
    # Default test cases if none provided
    if test_cases is None:
        test_cases = [
            "መጽሐፈ፡ ኩፋሌ፡ ዘአዳም፡ ወሄኖክ፡ ወኖህ፡ ወአብርሃም፡ ወኢሳይያስ፡ ወኢዮብ፡ ወሙሴ።",
            "በሰማይ፡ ወበምድር፡ ወበገጸ፡ ምድር፡ ወበነገሩ፡ ኵሎ፡ ዘእንበለ፡ ይኩን።",
            "ወምድር፡ ትኩን፡ ትህቅ፡ ወባዳ፡ ወጽልመት፡ እስከ፡ ላዕሌሃ።",
            "[MASK] የግእዝ ጽሑፍ ነው።"  # Test with a masked token
        ]
    
    # Load the model
    model, tokenizer, device = load_model(model_dir)
    
    print("\n" + "="*80)
    print("GEEZ TEXT RESTORATION MODEL TESTING")
    print("="*80)
    
    # Test each case
    for i, test_input in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Input:    {test_input}")
        
        # Generate output
        output = generate_text(model, tokenizer, test_input, device)
        print(f"Output:   {output}")
        print("-"*80)

def main():
    parser = argparse.ArgumentParser(description='Test the Ge\'ez text restoration model')
    parser.add_argument('--model_dir', type=str, default='models/geez_t5',
                       help='Directory containing the trained model')
    parser.add_argument('--test_file', type=str, default=None,
                       help='Optional JSON file containing test cases')
    args = parser.parse_args()
    
    # Load test cases from file if provided
    test_cases = None
    if args.test_file and Path(args.test_file).exists():
        with open(args.test_file, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
    
    # Run tests
    run_tests(args.model_dir, test_cases)

def test_script():
    # Path to the model
    model_path = "models/geez_t5"
    
    # Sample input text with missing spans
    test_text = "restore Ge'ez: ይህ የሙከራ ጽሑፍ ነው። [MISSING] ይህ የማሟያ ጽሑፍ ነው።"
    
    try:
        # Load the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        
        # Test restoration
        print(f"\nInput text: {test_text}")
        
        # Encode the input
        input_ids = tokenizer(test_text, return_tensors="pt").input_ids.to(device)
        
        # Generate output
        outputs = model.generate(
            input_ids,
            max_length=512,
            num_beams=5,
            num_return_sequences=3,
            early_stopping=True
        )
        
        # Decode and print results
        print("\nRestoration results:")
        for i, output in enumerate(outputs, 1):
            decoded = tokenizer.decode(output, skip_special_tokens=True)
            print(f"\nOption {i}:")
            print(decoded)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    test_script()
