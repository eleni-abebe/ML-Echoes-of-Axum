#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive analysis of tokenizer's handling of Ge'ez text.
Generates a detailed report on character coverage, vocabulary analysis,
and provides recommendations for improvement.
"""

import os
import sys
import json
import re
from collections import defaultdict
from pathlib import Path
from transformers import T5Tokenizer, AutoTokenizer

def get_geez_character_set():
    """Return a comprehensive set of Ge'ez characters and common combinations."""
    # Basic Ge'ez syllables (Fidel)
    base_consonants = [
        'ሀ', 'ለ', 'ሐ', 'መ', 'ሠ', 'ረ', 'ሰ', 'ሸ', 'ቀ', 'በ', 'ተ', 'ቸ', 'ኀ', 'ነ', 'ኘ', 'አ',
        'ከ', 'ኸ', 'ወ', 'ዐ', 'ዘ', 'ዠ', 'የ', 'ደ', 'ጀ', 'ገ', 'ጠ', 'ጨ', 'ጰ', 'ጸ', 'ፀ', 'ፈ', 'ፐ'
    ]
    
    # Common diacritics
    diacritics = ['', '፡', '።', '፣', '፤', '፥', '፦', '፧', '፨']
    
    # Generate common combinations
    geez_chars = set()
    for cons in base_consonants:
        # Add base character
        geez_chars.add(cons)
        
        # Add with common diacritics
        for dia in diacritics:
            geez_chars.add(cons + dia)
    
    # Add common Ge'ez punctuation and numbers
    geez_chars.update(['፩', '፪', '፫', '፬', '፭', '፮', '፯', '፰', '፱', '፲', '፳', '፴', '፵', '፶', '፷', '፸', '፹', '፺', '፻'])
    
    return sorted(geez_chars)

class TokenizerAnalyzer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = None
        self.vocab = None
        self.special_tokens = {}
        self.unknown_tokens = set()
        self.geez_chars = get_geez_character_set()
        self.common_geez_words = [
            "ግእዝ", "መጽሐፍ", "ኪዳን", "አማርኛ", "ትግርኛ", 
            "ኢትዮጵያ", "ክርስትና", "ኦርቶዶክስ", "ቤተክርስቲያን",
            "ሰላም", "እንቋዕ", "ቃለ", "መለኮት", "ሥላሴ",
            "ቅድስት", "ማርያም", "ገድል", "ወንጌል", "ቅዱስ"
        ]
    
    def load_tokenizer(self):
        """Load the tokenizer and its vocabulary."""
        print(f"\n{'='*80}")
        print(f"Loading tokenizer from: {self.model_path}")
        print(f"{'='*80}")
        
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_path)
            
            # Add special tokens if they don't exist
            special_tokens = {"additional_special_tokens": ["[MASK]"]}
            num_added = self.tokenizer.add_special_tokens(special_tokens)
            
            if num_added > 0:
                print(f"Added {num_added} special tokens to the tokenizer")
            
            self.vocab = self.tokenizer.get_vocab()
            self.special_tokens = self.tokenizer.special_tokens_map
            
            print("✓ Tokenizer loaded successfully")
            return True
            
        except Exception as e:
            print(f"✗ Error loading tokenizer: {str(e)}")
            return False
    
    def analyze_character_coverage(self):
        """Analyze coverage of Ge'ez characters in the vocabulary."""
        print("\n" + "="*80)
        print("Ge'ez Character Coverage Analysis")
        print("="*80)
        
        covered = []
        not_covered = []
        
        for char in self.geez_chars:
            tokens = self.tokenizer.tokenize(char)
            if not tokens or tokens[0] == self.tokenizer.unk_token:
                not_covered.append(char)
            else:
                covered.append(char)
        
        # Print coverage statistics
        total = len(self.geez_chars)
        coverage = (len(covered) / total) * 100 if total > 0 else 0
        
        print(f"\n📊 Coverage: {len(covered)}/{total} Ge'ez characters ({coverage:.2f}%)")
        
        # Print sample of covered characters
        print("\n✅ Covered characters (sample):")
        print(" ".join(covered[:20]) + (" ..." if len(covered) > 20 else ""))
        
        # Print sample of missing characters if any
        if not_covered:
            print("\n❌ Missing characters (sample):")
            print(" ".join(not_covered[:20]) + (" ..." if len(not_covered) > 20 else ""))
        
        return coverage
    
    def analyze_word_handling(self):
        """Analyze how well the tokenizer handles Ge'ez words."""
        print("\n" + "="*80)
        print("Ge'ez Word Handling Analysis")
        print("="*80)
        
        results = []
        
        for word in self.common_geez_words:
            tokens = self.tokenizer.tokenize(word)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            
            # Check for unknown tokens
            has_unknown = any(t == self.tokenizer.unk_token for t in tokens)
            
            results.append({
                'word': word,
                'tokens': tokens,
                'token_ids': token_ids,
                'has_unknown': has_unknown,
                'num_tokens': len(tokens)
            })
        
        # Print results
        for result in results:
            status = "❌" if result['has_unknown'] else "✅"
            print(f"\n{status} Word: {result['word']}")
            print(f"   Tokens: {result['tokens']}")
            print(f"   Token IDs: {result['token_ids']}")
            print(f"   Token count: {result['num_tokens']}")
        
        return results
    
    def analyze_vocabulary(self):
        """Analyze the tokenizer's vocabulary."""
        print("\n" + "="*80)
        print("Vocabulary Analysis")
        print("="*80)
        
        vocab_size = len(self.vocab)
        print(f"\n📚 Vocabulary size: {vocab_size:,} tokens")
        
        # Count tokens by type
        ascii_tokens = []
        unicode_tokens = []
        special_tokens = set(self.tokenizer.all_special_tokens)
        
        for token in self.vocab.keys():
            if token in special_tokens:
                continue
            if all(ord(c) < 128 for c in token):
                ascii_tokens.append(token)
            else:
                unicode_tokens.append(token)
        
        print(f"\n🔤 ASCII tokens: {len(ascii_tokens):,}")
        print(f"🌍 Unicode tokens: {len(unicode_tokens):,}")
        print(f"⭐ Special tokens: {len(special_tokens):,}")
        
        # Print sample of unicode tokens
        print("\n🔠 Sample Unicode tokens:")
        print(" ".join(unicode_tokens[:30]) + (" ..." if len(unicode_tokens) > 30 else ""))
        
        return {
            'total_vocab_size': vocab_size,
            'ascii_tokens': len(ascii_tokens),
            'unicode_tokens': len(unicode_tokens),
            'special_tokens': len(special_tokens)
        }
    
    def generate_recommendations(self, coverage, word_results, vocab_stats):
        """Generate recommendations based on the analysis."""
        print("\n" + "="*80)
        print("Recommendations")
        print("="*80)
        
        issues = []
        
        # Check character coverage
        if coverage < 90:  # Less than 90% coverage
            issues.append(f"- Low Ge'ez character coverage: {coverage:.2f}%")
        
        # Check for unknown tokens in common words
        unknown_words = [r['word'] for r in word_results if r['has_unknown']]
        if unknown_words:
            issues.append(f"- Found unknown tokens in {len(unknown_words)} common Ge'ez words")
        
        # Check vocabulary size and composition
        if vocab_stats['unicode_tokens'] < 1000:  # Arbitrary threshold
            issues.append(f"- Low number of Unicode tokens: {vocab_stats['unicode_tokens']:,}")
        
        # Print recommendations
        if not issues:
            print("\n✅ The tokenizer appears to have good support for Ge'ez text!")
            print("   No major issues detected.")
        else:
            print("\n🔧 The following issues were identified:")
            for issue in issues:
                print(f"   {issue}")
            
            print("\n💡 Recommended actions:")
            if coverage < 90:
                print("1. Retrain the tokenizer with Ge'ez text to improve character coverage")
            if unknown_words:
                print("2. Add common Ge'ez words to the tokenizer's vocabulary")
            if vocab_stats['unicode_tokens'] < 1000:
                print("3. Consider using a tokenizer with better Unicode support")
            
            print("\n📝 Next steps:")
            print("   - Collect more Ge'ez text data for training")
            print("   - Consider using a character-level or subword tokenizer")
            print("   - Fine-tune the tokenizer on your specific Ge'ez corpus")
    
    def run_analysis(self):
        """Run the complete analysis."""
        if not self.load_tokenizer():
            return
        
        try:
            # Run analyses
            print("\n" + "="*80)
            print("🚀 Starting Tokenizer Analysis")
            print("="*80)
            
            # Basic info
            print("\n📋 Basic Tokenizer Information")
            print("-" * 40)
            print(f"Tokenizer class: {self.tokenizer.__class__.__name__}")
            print(f"Model max length: {self.tokenizer.model_max_length}")
            print(f"Special tokens: {self.special_tokens}")
            
            # Run analyses
            coverage = self.analyze_character_coverage()
            word_results = self.analyze_word_handling()
            vocab_stats = self.analyze_vocabulary()
            
            # Generate recommendations
            self.generate_recommendations(coverage, word_results, vocab_stats)
            
            print("\n✅ Analysis complete!")
            
        except Exception as e:
            print(f"\n❌ Error during analysis: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    # Path to the fine-tuned model
    model_path = "models/t5_geez_span"
    
    # Initialize and run analyzer
    analyzer = TokenizerAnalyzer(model_path)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
