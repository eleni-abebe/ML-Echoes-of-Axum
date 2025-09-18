#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Analysis Script

This script helps analyze the behavior of the Ge'ez text generation model.
It tests various inputs, parameters, and model configurations to understand its capabilities.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from typing import List, Tuple

class GeEzTextAnalyzer:
    def __init__(self, model_path: str = "models/geez_t5"):
        """Initialize the analyzer with a pre-trained model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n{'='*50}")
        print(f"Initializing model from: {model_path}")
        print(f"Using device: {self.device}")
        print(f"{'='*50}\n")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            add_prefix_space=True
        )
        
        try:
            # First try loading as a T5 model
            self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(self.device)
        except:
            # Fall back to AutoModelForSeq2SeqLM if T5 loading fails
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        
        self.model.eval()
        
        # Set model-specific generation parameters
        self.generation_config = {
            'max_length': 100,
            'num_beams': 5,
            'no_repeat_ngram_size': 2,
            'early_stopping': True,
            'num_return_sequences': 1,
            'temperature': 0.7,
            'top_k': 50,
            'top_p': 0.95,
            'do_sample': True,
        }
        
        # Test prompts in Ge'ez
        self.test_prompts = [
            "ይህ የሙከራ ጽሑፍ ነው",  # "This is a test document"
            "በስመ አብ ወወልድ ወመንፈስ ቅዱስ አሐዱ አምላክ።",  # Common opening phrase
            "መጽሐፈ ኩፋሌ ይባርክ እግዚአብሔር አምላክነት።"  # Title and blessing
        ]
    
    def generate_text(self, prompt: str, **generation_params) -> List[str]:
        """Generate text using the model."""
        # Use default generation parameters, but allow overriding
        params = {**self.generation_config, **generation_params}
        
        # Tokenize the input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **params
            )
        
        # Decode the output
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(text)
            
        return generated_texts
    
    def run_analysis(self):
        """Run a comprehensive analysis of the model."""
        print("\n" + "="*50)
        print("RUNNING BASIC GENERATION TESTS")
        print("="*50)
        
        for prompt in self.test_prompts:
            print(f"\nInput: {prompt}")
            try:
                # Try with different generation strategies
                print("\n1. Standard generation:")
                outputs = self.generate_text(prompt)
                for i, output in enumerate(outputs, 1):
                    print(f"   {i}. {output}")
                
                print("\n2. More deterministic (lower temperature):")
                outputs = self.generate_text(prompt, temperature=0.3, do_sample=True)
                for i, output in enumerate(outputs, 1):
                    print(f"   {i}. {output}")
                
                print("\n3. More creative (higher temperature):")
                outputs = self.generate_text(prompt, temperature=1.2, do_sample=True)
                for i, output in enumerate(outputs, 1):
                    print(f"   {i}. {output}")
                
            except Exception as e:
                print(f"Error during generation: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print("\nAnalysis complete!")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze a Ge'ez language model")
    parser.add_argument("--model", default="models/geez_t5", 
                       help="Path to the model directory")
    
    args = parser.parse_args()
    
    try:
        analyzer = GeEzTextAnalyzer(args.model)
        analyzer.run_analysis()
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
