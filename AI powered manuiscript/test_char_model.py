#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for character-level Ge'ez text prediction.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_path):
    """Load the model and tokenizer."""
    print(f"Loading model from {model_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    return model, tokenizer, device

def predict_next_chars(model, tokenizer, device, prompt, num_predictions=3):
    """Generate next character predictions."""
    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate predictions
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=len(prompt) + 1,
            num_return_sequences=num_predictions,
            do_sample=True,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and return predictions
    predictions = []
    for output in outputs:
        pred_text = tokenizer.decode(output, skip_special_tokens=True)
        # Get only the predicted character (last character in the output)
        if len(pred_text) > len(prompt):
            pred_char = pred_text[-1]
            predictions.append(pred_char)
    
    return predictions

def main():
    # Path to the model
    model_path = "models/geez_t5"
    
    try:
        # Load the model
        model, tokenizer, device = load_model(model_path)
        
        # Test with different prompts
        test_prompts = [
            "ይህ የሙከራ ጽሑፍ ነው",
            "በስመ አብ ወወልድ",
            "መጽሐፈ ኩፋሌ"
        ]
        
        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            predictions = predict_next_chars(model, tokenizer, device, prompt)
            print(f"Next character predictions: {', '.join(predictions)}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
