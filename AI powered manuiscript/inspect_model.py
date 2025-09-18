#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Inspection Script

This script inspects the model's configuration and tests its generation capabilities.
"""

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
import json

def inspect_model(model_path: str = "models/geez_t5"):
    """Inspect the model configuration and test generation."""
    print(f"\n{'='*50}")
    print(f"Inspecting model at: {model_path}")
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"{'='*50}\n")
    
    # Load config
    try:
        config = AutoConfig.from_pretrained(model_path)
        print("\nModel Configuration:")
        print("-" * 30)
        print(json.dumps(config.to_dict(), indent=2, default=str))
    except Exception as e:
        print(f"Error loading config: {e}")
    
    # Load tokenizer and model
    try:
        print("\nLoading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.eval()
        
        # Test generation with different prompts and settings
        test_prompts = [
            "",  # Empty prompt
            " ",  # Space
            "ይህ",  # Single word
            "በስመ አብ"  # Common phrase
        ]
        
        for prompt in test_prompts:
            print(f"\n{'='*50}")
            print(f"Testing with prompt: '{prompt}'")
            print(f"{'='*50}")
            
            # Test different generation methods
            for method in ["greedy", "beam", "sampling"]:
                print(f"\n--- {method.upper()} SEARCH ---")
                
                inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
                
                try:
                    if method == "greedy":
                        outputs = model.generate(
                            inputs.input_ids,
                            max_length=50,
                            num_beams=1,
                            do_sample=False,
                            early_stopping=True
                        )
                    elif method == "beam":
                        outputs = model.generate(
                            inputs.input_ids,
                            max_length=50,
                            num_beams=5,
                            early_stopping=True
                        )
                    else:  # sampling
                        outputs = model.generate(
                            inputs.input_ids,
                            max_length=50,
                            do_sample=True,
                            top_k=50,
                            top_p=0.95,
                            temperature=0.7
                        )
                    
                    # Decode and print output
                    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    print(f"Output: {output_text}")
                    
                except Exception as e:
                    print(f"Error during {method} generation: {e}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Inspect a Ge'ez language model")
    parser.add_argument("--model", default="models/geez_t5", 
                       help="Path to the model directory")
    
    args = parser.parse_args()
    inspect_model(args.model)

if __name__ == "__main__":
    main()
