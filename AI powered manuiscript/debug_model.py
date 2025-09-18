#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug Ge'ez Model Predictions

Analyzes model predictions to understand performance issues.
"""

import os
import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Configuration
MODEL_PATH = "./models/geez_t5"  # Try both models: geez_t5 and t5_geez_span
TEST_FILE = "./data/processed/test.jsonl"
OUTPUT_FILE = "./debug_predictions.txt"

def load_model_and_tokenizer(model_path):
    """Load model and tokenizer."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = T5Tokenizer.from_pretrained(model_path, local_files_only=True)
    
    print(f"Loading model from: {model_path}")
    model = T5ForConditionalGeneration.from_pretrained(
        model_path, 
        local_files_only=True
    ).to(device)
    model.eval()
    
    return model, tokenizer, device

def load_test_data(test_file, max_examples=10):
    """Load test examples."""
    examples = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                ex = json.loads(line)
                if 'input' in ex and 'target' in ex:
                    examples.append(ex)
                    if len(examples) >= max_examples:
                        break
            except json.JSONDecodeError:
                continue
    return examples

def analyze_predictions(model, tokenizer, device, examples):
    """Generate and analyze model predictions."""
    results = []
    
    for ex in tqdm(examples, desc="Analyzing predictions"):
        # Prepare input
        input_text = f"restore: {ex['input']}"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        
        # Generate prediction
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=128,
                num_beams=5,
                early_stopping=True
            )
        
        # Decode prediction
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Store results
        results.append({
            'input': ex['input'],
            'target': ex['target'],
            'prediction': prediction,
            'input_length': len(ex['input']),
            'target_length': len(ex['target']),
            'prediction_length': len(prediction),
            'exact_match': prediction == ex['target']
        })
    
    return results

def print_analysis(results):
    """Print detailed analysis of predictions."""
    print("\n" + "="*80)
    print("MODEL PREDICTION ANALYSIS")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        print(f"\nExample {i}:")
        print(f"Input    : {result['input']}")
        print(f"Target   : {result['target']}")
        print(f"Predicted: {result['prediction']}")
        print(f"Lengths  : Input={result['input_length']}, "
              f"Target={result['target_length']}, "
              f"Prediction={result['prediction_length']}")
        print(f"Exact Match: {result['exact_match']}")
    
    # Summary statistics
    exact_matches = sum(1 for r in results if r['exact_match'])
    avg_input_len = sum(r['input_length'] for r in results) / len(results)
    avg_target_len = sum(r['target_length'] for r in results) / len(results)
    avg_pred_len = sum(r['prediction_length'] for r in results) / len(results)
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Examples analyzed: {len(results)}")
    print(f"Exact matches: {exact_matches} ({exact_matches/len(results):.1%})")
    print(f"Avg input length: {avg_input_len:.1f}")
    print(f"Avg target length: {avg_target_len:.1f}")
    print(f"Avg prediction length: {avg_pred_len:.1f}")
    
    # Save results to file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {OUTPUT_FILE}")

def main():
    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(MODEL_PATH)
    
    # Load test data
    examples = load_test_data(TEST_FILE, max_examples=10)
    if not examples:
        print(f"No test examples found in {TEST_FILE}")
        return
    
    # Analyze predictions
    results = analyze_predictions(model, tokenizer, device, examples)
    
    # Print and save analysis
    print_analysis(results)

if __name__ == "__main__":
    main()
