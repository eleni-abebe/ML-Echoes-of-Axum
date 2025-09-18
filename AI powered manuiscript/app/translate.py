#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Translation module for Ge'ez, Amharic, and English using mT5.

This module provides functionality to translate between Ge'ez (Ethiopic),
Amharic, and English using a fine-tuned mT5 model.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
from transformers import AutoTokenizer, MT5ForConditionalGeneration
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Language codes and names
LANGUAGE_CODES = {
    'geez': 'gez',
    'amh': 'amh',
    'eng': 'eng',
    'ethiopic': 'gez',
    'amharic': 'amh',
    'english': 'eng',
}

# Language names for display
LANGUAGE_NAMES = {
    'gez': 'Ge\'ez',
    'amh': 'Amharic',
    'eng': 'English',
}

# Supported translation directions
SUPPORTED_DIRECTIONS = [
    ('gez', 'amh'),  # Ge'ez to Amharic
    ('amh', 'gez'),  # Amharic to Ge'ez
    ('gez', 'eng'),  # Ge'ez to English
    ('eng', 'gez'),  # English to Ge'ez
    ('amh', 'eng'),  # Amharic to English
    ('eng', 'amh'),  # English to Amharic
]


class GeEzTranslator:
    """Translator for Ge'ez, Amharic, and English using mT5."""
    
    def __init__(self, 
                model_path: Optional[str] = None,
                device: Optional[str] = None,
                num_beams: int = 4,
                max_length: int = 256,
                **generation_kwargs):
        """
        Initialize the translator.
        
        Args:
            model_path: Path to the fine-tuned mT5 model
            device: Device to run the model on ('cuda', 'mps', 'cpu')
            num_beams: Number of beams for beam search
            max_length: Maximum length of generated sequences
            **generation_kwargs: Additional keyword arguments for generation
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.num_beams = num_beams
        self.max_length = max_length
        self.generation_kwargs = {
            'num_beams': num_beams,
            'max_length': max_length,
            'early_stopping': True,
            'no_repeat_ngram_size': 3,
            'length_penalty': 1.0,
            **generation_kwargs
        }
        
        # Default to a base model if none provided
        if model_path is None:
            model_path = "google/mt5-base"
            logger.warning(f"No model path provided, using default: {model_path}")
        
        logger.info(f"Loading translation model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = MT5ForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Add language codes to tokenizer if not present
        self._add_special_tokens()
        
        logger.info(f"Translation model loaded on {self.device}")
    
    def _add_special_tokens(self):
        """Add language tokens to the tokenizer if they don't exist."""
        special_tokens = []
        
        # Add language tokens
        for lang_code in ['gez', 'amh', 'eng']:
            lang_token = f'<{lang_code}>'
            if lang_token not in self.tokenizer.additional_special_tokens:
                special_tokens.append(lang_token)
        
        # Add special tokens if any are missing
        if special_tokens:
            self.tokenizer.add_special_tokens({
                'additional_special_tokens': special_tokens
            })
            # Resize model token embeddings if needed
            if hasattr(self.model, 'resize_token_embeddings'):
                self.model.resize_token_embeddings(len(self.tokenizer))
    
    def _prepare_input(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Prepare input text with language tokens.
        
        Args:
            text: Input text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Formatted input string with language tokens
        """
        # Normalize language codes
        source_lang = self._normalize_lang_code(source_lang)
        target_lang = self._normalize_lang_code(target_lang)
        
        # Format: <source_lang> text <target_lang>
        return f"<{source_lang}> {text} <{target_lang}>"
    
    def _normalize_lang_code(self, lang_code: str) -> str:
        """
        Normalize language code to standard format.
        
        Args:
            lang_code: Input language code or name
            
        Returns:
            Normalized 3-letter language code
        """
        lang_code = lang_code.lower().strip()
        return LANGUAGE_CODES.get(lang_code, lang_code[:3])
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect the most likely language of the input text.
        
        This is a simple implementation that looks for Ge'ez script characters.
        For a production system, you might want to use a dedicated language
        identification library like langdetect or fasttext.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (detected_language_code, confidence)
        """
        # Check for Ge'ez script characters (U+1200 to U+137F)
        ethiopic_chars = re.findall(r'[\u1200-\u137F]', text)
        ethiopic_ratio = len(ethiopic_chars) / max(1, len(text.strip()))
        
        # Check for Amharic script (similar to Ge'ez but with additional characters)
        # This is a very rough approximation
        if ethiopic_ratio > 0.5:
            # If there are many Ethiopic characters, it's likely Ge'ez or Amharic
            # This is a simplification - in reality, you'd need a better classifier
            return ('gez', 0.7)  # Default to Ge'ez with medium confidence
        
        # Check for English (very basic check)
        if re.search(r'[a-zA-Z]', text):
            return ('eng', 0.8)
        
        # Default to unknown
        return ('unk', 0.0)
    
    def translate(self, 
                 text: str, 
                 source_lang: Optional[str] = None,
                 target_lang: str = 'eng',
                 **generation_kwargs) -> Dict[str, Any]:
        """
        Translate text from source language to target language.
        
        Args:
            text: Input text to translate
            source_lang: Source language code (auto-detected if None)
            target_lang: Target language code (default: 'eng' for English)
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing translation results
        """
        # Normalize language codes
        target_lang = self._normalize_lang_code(target_lang)
        
        # Auto-detect source language if not provided
        if source_lang is None:
            source_lang, confidence = self.detect_language(text)
            logger.info(f"Detected language: {source_lang} (confidence: {confidence:.2f})")
        else:
            source_lang = self._normalize_lang_code(source_lang)
            confidence = 1.0
        
        # Check if translation direction is supported
        if (source_lang, target_lang) not in SUPPORTED_DIRECTIONS:
            raise ValueError(
                f"Unsupported translation direction: {source_lang} -> {target_lang}. "
                f"Supported directions: {SUPPORTED_DIRECTIONS}"
            )
        
        # Prepare input
        prepared_input = self._prepare_input(text, source_lang, target_lang)
        
        # Tokenize
        inputs = self.tokenizer(
            prepared_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        
        # Generate translation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **{**self.generation_kwargs, **generation_kwargs}
            )
        
        # Decode the output
        translated_text = self.tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # Calculate confidence (this is a placeholder - in practice, you might want to
        # use the model's output probabilities for a better confidence estimate)
        confidence = min(1.0, max(0.0, confidence * 0.9))  # Adjust based on detection confidence
        
        return {
            'source_text': text,
            'translated_text': translated_text,
            'source_lang': source_lang,
            'target_lang': target_lang,
            'confidence': confidence,
            'model': self.model.config._name_or_path,
        }
    
    def batch_translate(self,
                       texts: List[str],
                       source_lang: Optional[Union[str, List[str]]] = None,
                       target_lang: Union[str, List[str]] = 'eng',
                       batch_size: int = 8,
                       **generation_kwargs) -> List[Dict[str, Any]]:
        """
        Translate a batch of texts.
        
        Args:
            texts: List of input texts to translate
            source_lang: Source language code(s) (auto-detected if None)
            target_lang: Target language code(s) (default: 'eng' for English)
            batch_size: Batch size for translation
            **generation_kwargs: Additional generation parameters
            
        Returns:
            List of translation results
        """
        results = []
        
        # Handle single language for all texts
        if isinstance(source_lang, str):
            source_lang = [source_lang] * len(texts)
        if isinstance(target_lang, str):
            target_lang = [target_lang] * len(texts)
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Translating batch"):
            batch_texts = texts[i:i+batch_size]
            batch_src = source_lang[i:i+batch_size] if source_lang else [None] * len(batch_texts)
            batch_tgt = target_lang[i:i+batch_size] if target_lang else ['eng'] * len(batch_texts)
            
            for text, src, tgt in zip(batch_texts, batch_src, batch_tgt):
                try:
                    result = self.translate(
                        text=text,
                        source_lang=src,
                        target_lang=tgt,
                        **generation_kwargs
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error translating text: {e}")
                    results.append({
                        'source_text': text,
                        'translated_text': '',
                        'source_lang': src or 'unk',
                        'target_lang': tgt,
                        'error': str(e),
                        'confidence': 0.0,
                    })
        
        return results


def load_translator(model_path: Optional[str] = None, **kwargs) -> GeEzTranslator:
    """
    Load a translator from a saved model.
    
    Args:
        model_path: Path to the fine-tuned model
        **kwargs: Additional arguments for GeEzTranslator
        
    Returns:
        GeEzTranslator instance
    """
    return GeEzTranslator(model_path=model_path, **kwargs)


def main():
    """Command-line interface for the translator."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Translate between Ge'ez, Amharic, and English")
    parser.add_argument('text', nargs='?', help='Text to translate (if not provided, enter interactive mode)')
    parser.add_argument('--source', '-s', default=None, help='Source language code (auto-detected if not provided)')
    parser.add_argument('--target', '-t', default='eng', help='Target language code (default: eng)')
    parser.add_argument('--model', default='models/mt5_geez_translator',
                      help='Path to the fine-tuned model (default: models/mt5_geez_translator)')
    parser.add_argument('--device', default=None,
                      help='Device to run the model on (cuda, mps, cpu)')
    parser.add_argument('--batch-size', type=int, default=8,
                      help='Batch size for translation (default: 8)')
    parser.add_argument('--max-length', type=int, default=256,
                      help='Maximum length of generated sequences (default: 256)')
    parser.add_argument('--num-beams', type=int, default=4,
                      help='Number of beams for beam search (default: 4)')
    
    args = parser.parse_args()
    
    # Load the translator
    try:
        translator = load_translator(
            model_path=args.model,
            device=args.device,
            num_beams=args.num_beams,
            max_length=args.max_length,
        )
    except Exception as e:
        logger.error(f"Error loading translator: {e}")
        return 1
    
    # Interactive mode
    if not args.text:
        print("Entering interactive mode. Type 'exit' or 'quit' to exit.")
        print("Available languages:")
        for code, name in LANGUAGE_NAMES.items():
            print(f"  {code}: {name}")
        
        while True:
            try:
                # Get source language
                source = input("\nSource language (or 'auto' for auto-detection): ").strip().lower()
                if source in ['exit', 'quit']:
                    break
                if source == 'auto':
                    source = None
                
                # Get target language
                target = input("Target language (default: eng): ").strip().lower() or 'eng'
                
                # Get text to translate
                text = input("\nEnter text to translate: ").strip()
                if not text:
                    continue
                
                # Translate
                result = translator.translate(
                    text=text,
                    source_lang=source,
                    target_lang=target,
                )
                
                # Print results
                print("\nTranslation results:")
                print(f"Source ({LANGUAGE_NAMES.get(result['source_lang'], result['source_lang'])}): {result['source_text']}")
                print(f"Target ({LANGUAGE_NAMES.get(result['target_lang'], result['target_lang'])}): {result['translated_text']}")
                print(f"Confidence: {result['confidence']:.2f}")
            
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
    else:
        # Single text mode
        try:
            result = translator.translate(
                text=args.text,
                source_lang=args.source,
                target_lang=args.target,
            )
            
            # Print results
            print(f"Source ({LANGUAGE_NAMES.get(result['source_lang'], result['source_lang'])}): {result['source_text']}")
            print(f"Target ({LANGUAGE_NAMES.get(result['target_lang'], result['target_lang'])}): {result['translated_text']}")
            print(f"Confidence: {result['confidence']:.2f}")
            
        except Exception as e:
            logger.error(f"Error: {e}")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
