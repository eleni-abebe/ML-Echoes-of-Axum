#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Restored Ge'ez T5 Model

This script tests the trained T5 model for Ge'ez text restoration.
"""

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

def load_model_and_tokenizer(model_path):
    """Load the trained model and tokenizer."""
    print(f"Loading model from: {model_path}")
    
    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(
        model_path,
        use_fast=False,
        add_prefix_space=True
    )
    
    # Load model
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=50):
    """Generate text using the trained model."""
    # Prepare input
    input_text = f"restore Ge'ez: {prompt}"
    
    # Tokenize input
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    
    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True
        )
    
    # Decode output
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text

def main():
    # Set console output encoding
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    # Model path
    model_path = "models/geez_t5_restored"
    
    # Test prompts
    test_prompts = [
        "በስመ አብ ወወልድ ወመንፈስ ቅዱስ",
        "ይህ የሙከራ ጽሑፍ ነው",
        "መጽሐፈ ኩፋሌ"
    ]
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(model_path)
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Test each prompt
        for prompt in test_prompts:
            print("\n" + "="*50)
            print(f"Input: {prompt}")
            
            # Generate and print output
            output = generate_text(model, tokenizer, prompt)
            print(f"Output: {output}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
