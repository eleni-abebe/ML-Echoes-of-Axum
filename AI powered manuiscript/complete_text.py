#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ge'ez Text Completion Script

This script uses a character-level model to complete Ge'ez text.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional

class GeEzTextCompleter:
    def __init__(self, model_path: str = "models/geez_t5"):
        """Initialize the text completer with a pre-trained model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def complete_text(self, prompt: str, max_length: int = 100, temperature: float = 0.7) -> str:
        """
        Complete the given text prompt using the model.
        
        Args:
            prompt: The starting text to complete
            max_length: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            The completed text
        """
        # Encode the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate completion
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=len(prompt) + max_length,
                temperature=temperature,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and return the completed text
        completed_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return completed_text

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete Ge'ez text using a pre-trained model")
    parser.add_argument("prompt", nargs="?", help="Starting text to complete")
    parser.add_argument("--model", default="models/geez_t5", help="Path to the model directory")
    parser.add_argument("--length", type=int, default=100, help="Maximum number of new characters to generate")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature (0.1 to 1.0)")
    
    args = parser.parse_args()
    
    # If no prompt provided, use a default one
    if not args.prompt:
        args.prompt = "በስመ አብ"
    
    try:
        # Initialize the completer
        completer = GeEzTextCompleter(args.model)
        
        # Generate completion
        print(f"\nPrompt: {args.prompt}")
        print("\nGenerating completion...\n")
        
        completed_text = completer.complete_text(
            args.prompt,
            max_length=args.length,
            temperature=args.temp
        )
        
        print("Completed text:")
        print(completed_text)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
