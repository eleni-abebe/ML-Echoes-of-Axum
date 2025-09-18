import os
import json
import torch
import logging
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from tqdm import tqdm

# Fix Windows console encoding
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('evaluation.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class GeEzTestDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    self.examples.append((data['input'], data['target']))
            logger.info(f"Loaded {len(self.examples)} examples from {file_path}")
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        input_text, target_text = self.examples[idx]
        
        try:
            input_encoding = self.tokenizer(
                "restore Ge'ez: " + input_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': input_encoding['input_ids'].squeeze(),
                'attention_mask': input_encoding['attention_mask'].squeeze(),
                'input_text': input_text,
                'target_text': target_text
            }
        except Exception as e:
            logger.error(f"Error processing example {idx}: {str(e)}")
            raise

def evaluate():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Ge'ez Text Restoration Model")
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model')
    parser.add_argument('--test_file', type=str, required=True,
                      help='Path to the test data file')
    parser.add_argument('--max_length', type=int, default=128,
                      help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size for evaluation')
    parser.add_argument('--output_file', type=str, default='evaluation_results.json',
                      help='File to save evaluation results')
    
    args = parser.parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    try:
        # Load tokenizer and model
        logger.info(f"Loading tokenizer and model from {args.model_path}")
        tokenizer = MT5Tokenizer.from_pretrained(
            args.model_path,
            legacy=False,
            use_fast=True
        )
        
        # Load model with proper configuration
        model = MT5ForConditionalGeneration.from_pretrained(args.model_path).to(device)
        model.eval()
        
        # Test model with a simple prompt
        test_prompt = "restore Ge'ez: ንጉሥ፥ቤተ፥ክርስትያን።ነቢይ"
        input_ids = tokenizer(test_prompt, return_tensors="pt").input_ids.to(device)
        outputs = model.generate(input_ids, max_length=50)
        test_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info("\n=== Model Test ===")
        logger.info(f"Input: {test_prompt}")
        logger.info(f"Output: {test_output}")
        logger.info("==================\n")
        
        # Load test data
        test_dataset = GeEzTestDataset(args.test_file, tokenizer, args.max_length)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        results = {
            'samples': [],
            'metrics': {
                'exact_match': 0,
                'total': 0
            }
        }
        
        logger.info(f"Evaluating on {len(test_dataset)} samples...")
        logger.info("=" * 80)
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=args.max_length,
                    num_beams=4,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
                
                pred_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                for i in range(len(batch['input_text'])):
                    input_text = batch['input_text'][i]
                    target_text = batch['target_text'][i]
                    pred_text = pred_texts[i]
                    
                    is_correct = pred_text.strip() == target_text.strip()
                    results['metrics']['exact_match'] += int(is_correct)
                    results['metrics']['total'] += 1
                    
                    results['samples'].append({
                        'input': input_text,
                        'prediction': pred_text,
                        'reference': target_text,
                        'correct': is_correct
                    })
        
        # Calculate metrics
        results['metrics']['accuracy'] = results['metrics']['exact_match'] / results['metrics']['total'] if results['metrics']['total'] > 0 else 0
        
        # Save results
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Print summary
        logger.info("\n=== Evaluation Summary ===")
        logger.info(f"Exact Match Accuracy: {results['metrics']['accuracy']:.2%}")
        logger.info(f"Correct: {results['metrics']['exact_match']}/{results['metrics']['total']}")
        
        # Print examples
        logger.info("\n=== Example Predictions ===")
        for i, sample in enumerate(results['samples'][:5]):
            logger.info(f"\nExample {i+1}:")
            logger.info(f"Input:     {sample['input']}")
            logger.info(f"Predicted: {sample['prediction']}")
            logger.info(f"Reference: {sample['reference']}")
            logger.info(f"Correct:   {sample['correct']}")
        
        logger.info("\nEvaluation complete. Results saved to:", args.output_file)
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    evaluate()