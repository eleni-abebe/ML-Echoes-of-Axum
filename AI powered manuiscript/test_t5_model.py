#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test T5 Model

A simple script to test the T5 model with proper input formatting.
"""

import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

def main():
    # Initialize model and tokenizer
    model_path = "models/geez_t5"
    
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    # Set model to evaluation mode
    model.eval()
    
    # Test prompts with T5 task prefix
    test_prompts = [
        "restore Ge'ez: በስመ አብ",
        "translate to Ge'ez: In the name of the Father",
        "complete: በስመ አብ"
    ]
    
    for prompt in test_prompts:
        print(f"\n{'='*50}")
        print(f"Input: {prompt}")
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate output
        with torch.no_grad():
            # Try with different generation methods
            for method in ["greedy", "beam", "sampling"]:
                print(f"\n--- {method.upper()} ---")
                
                try:
                    if method == "greedy":
                        outputs = model.generate(
                            inputs.input_ids,
                            max_length=50,
                            num_beams=1,
                            do_sample=False
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

if __name__ == "__main__":
    main()
