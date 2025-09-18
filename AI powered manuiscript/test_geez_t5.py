#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Ge'ez T5 Model with Proper Configuration
"""

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

def load_model_and_tokenizer(model_path):
    """Load model and tokenizer with proper configuration."""
    print(f"Loading model from: {model_path}")
    
    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(
        model_path,
        use_fast=False,
        add_prefix_space=True
    )
    
    # Print tokenizer info
    print("\nTokenizer info:")
    print(f"- Vocab size: {len(tokenizer)}")
    print(f"- Special tokens: {tokenizer.special_tokens_map}")
    
    # Load model
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    # Print model info
    print("\nModel info:")
    print(f"- Model type: {model.__class__.__name__}")
    print(f"- Model config: {model.config}")
    
    return model, tokenizer

def test_generation(model, tokenizer, prompt, max_length=50):
    """Test text generation with the model."""
    # Test different input formats
    input_formats = [
        f"restore Ge'ez: {prompt}",  # Original format
        f"translate to Ge'ez: {prompt}",  # Translation format
        f"complete: {prompt}",  # Completion format
        prompt  # Raw prompt
    ]
    
    results = []
    
    for input_text in input_formats:
        print(f"\nTesting input format: {input_text}")
        
        # Tokenize
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        print(f"Tokenized input: {inputs.input_ids.tolist()}")
        
        # Generate output with different methods
        generation_params = [
            {"name": "greedy", "params": {"num_beams": 1, "do_sample": False}},
            {"name": "beam", "params": {"num_beams": 5, "do_sample": False}},
            {"name": "sampling", "params": {"do_sample": True, "temperature": 0.7, "top_k": 50, "top_p": 0.95}}
        ]
        
        for method in generation_params:
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_length=max_length,
                        early_stopping=True,
                        **method["params"]
                    )
                
                output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"{method['name']}: {output_text}")
                results.append((input_text, method['name'], output_text))
                
            except Exception as e:
                print(f"Error with {method['name']}: {str(e)}")
    
    return results

def main():
    # Set console output encoding
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    # Model path
    model_path = "models/geez_t5"
    
    # Test prompts
    test_prompts = [
        "በስመ አብ ወወልድ ወመንፈስ ቅዱስ",
        "ይህ የሙከራ ጽሑፍ ነው",
        "መጽሐፈ ኩፋሌ"
    ]
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(model_path)
        
        # Test each prompt
        for prompt in test_prompts:
            print("\n" + "="*50)
            print(f"Input: {prompt}")
            
            # Generate and print output
            output = test_generation(model, tokenizer, prompt)
            print(f"Output: {output}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
