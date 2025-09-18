#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text span restoration for Ge'ez manuscripts using T5.

This module provides functionality to restore missing or damaged text spans
in Ge'ez manuscripts using a fine-tuned T5 model.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class GeEzTextRestorer:
    """Restore missing or damaged text spans in Ge'ez manuscripts."""
    
    def __init__(self, 
                model_path: Optional[str] = None,
                device: Optional[str] = None,
                num_beams: int = 5,
                max_length: int = 512,
                num_return_sequences: int = 3):
        """
        Initialize the text restorer.
        
        Args:
            model_path: Path to the fine-tuned T5 model
            device: Device to run the model on ('cuda', 'mps', 'cpu')
            num_beams: Number of beams for beam search
            max_length: Maximum length of generated sequences
            num_return_sequences: Number of sequences to return per input
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.num_beams = num_beams
        self.max_length = max_length
        self.num_return_sequences = num_return_sequences
        
        # Load model and tokenizer
        logger.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.model.eval()
        logger.info(f"Model loaded on {self.device}")
    
    def preprocess_input(self, text: str) -> str:
        """
        Preprocess input text before feeding it to the model.
        
        Args:
            text: Input text with placeholders for missing spans
            
        Returns:
            Preprocessed text
        """
        # Normalize various placeholder formats to <MISSING-i>
        text = self._normalize_missing_markers(text)
        # Convert standardized markers to T5 sentinel tokens
        text = re.sub(r'<MISSING-(\d+)>', lambda m: f"<extra_id_{m.group(1)}>", text)
        return text.strip()
    
    def postprocess_output(self, text: str) -> str:
        """
        Postprocess model output to replace sentinel tokens with readable placeholders.
        
        Args:
            text: Model output text
            
        Returns:
            Postprocessed text with readable placeholders
        """
        # Replace sentinel tokens with readable placeholders
        text = re.sub(r'<extra_id_(\d+)>', lambda m: f"<MISSING-{m.group(1)}>", text)
        return text
    
    def extract_spans(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract missing spans from the input text.
        
        Args:
            text: Input text with placeholders
            
        Returns:
            List of dictionaries containing span information
        """
        # Find all placeholders in the format <MISSING-N>
        placeholders = list(re.finditer(r'<MISSING-(\d+)>', text))
        spans = []
        
        for i, match in enumerate(placeholders):
            span_id = int(match.group(1))
            start = match.start()
            end = match.end()
            
            # Calculate context window (characters before and after the placeholder)
            context_window = 50  # characters
            context_start = max(0, start - context_window)
            context_end = min(len(text), end + context_window)
            context = text[context_start:context_end]
            
            spans.append({
                'id': span_id,
                'start': start,
                'end': end,
                'placeholder': match.group(0),
                'context': context,
                'context_start': context_start,
                'context_end': context_end,
            })
        
        return spans
    
    def generate_restorations(self, 
                            text: str,
                            temperature: float = 0.7,
                            top_k: int = 50,
                            top_p: float = 0.95) -> List[Dict[str, Any]]:
        """
        Generate restorations for missing spans in the input text.
        
        Args:
            text: Input text with placeholders for missing spans
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            
        Returns:
            List of restoration candidates with scores and other metadata
        """
        # Preprocess input and extract spans
        processed_text = self.preprocess_input(text)
        spans = self.extract_spans(text)
        
        if not spans:
            logger.warning("No missing spans found in the input text")
            return []
        
        # Prepare input for the model
        inputs = self.tokenizer(
            processed_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        
        # Generate restorations
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=self.num_beams,
                num_return_sequences=self.num_return_sequences,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                output_scores=True,
                return_dict_in_generate=True,
            )
        
        # Decode the generated sequences
        generated_sequences = self.tokenizer.batch_decode(
            outputs.sequences, 
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True
        )
        
        # Process the outputs
        results = []
        for i in range(self.num_return_sequences):
            seq = generated_sequences[i]
            
            # Postprocess the output
            restored_text = self.postprocess_output(seq)
            
            # Calculate confidence score (average token probability)
            if hasattr(outputs, 'scores'):
                # Get the logits for the generated tokens
                logits = torch.stack([scores[i] for scores in outputs.scores], dim=1)
                # Convert logits to probabilities
                probs = torch.softmax(logits, dim=-1)
                # Get the probability of the generated tokens
                token_probs = torch.gather(probs, 2, outputs.sequences[i, 1:].unsqueeze(-1)).squeeze(-1)
                # Calculate average log probability (avoiding log(0))
                avg_log_prob = (token_probs + 1e-10).log().mean().item()
                # Convert to confidence score (0-1)
                confidence = min(1.0, max(0.0, 1.0 + avg_log_prob / 20.0))  # Scale to reasonable range
            else:
                confidence = 0.5  # Default confidence if scores not available
            
            results.append({
                'text': restored_text,
                'confidence': confidence,
                'sequence_id': i,
            })
        
        return results

    def _normalize_missing_markers(self, text: str) -> str:
        """
        Convert common placeholder formats in user input to canonical <MISSING-i>.
        Supported tokens include: <MISSING>, [MISSING], MISSING, <MASK>, [MASK], MASK, and sequences of 3+ underscores.
        Already-numbered markers like <MISSING-2> or [MASK-3] preserve their index.
        """
        pattern = re.compile(
            r"(\<\s*(?:MISSING|MASK)\s*(?:-\s*(\d+))?\s*\>|"  # <MISSING> or <MASK> (optional -N)
            r"\[\s*(?:MISSING|MASK)\s*(?:-\s*(\d+))?\s*\]|"     # [MISSING] or [MASK] (optional -N)
            r"\b(?:MISSING|MASK)\b|"                               # bare word
            r"_{3,})"                                                # ___ (3 or more underscores)
        )
        counter = 0
        parts = []
        last = 0

        for m in pattern.finditer(text):
            start, end = m.span()
            parts.append(text[last:start])
            # Choose index: preserve explicit number if provided, else assign sequential
            num = None
            for g in m.groups()[1:]:
                if g is not None:
                    num = int(g)
                    break
            if num is None:
                num = counter
                counter += 1
            parts.append(f"<MISSING-{num}>")
            last = end

        parts.append(text[last:])
        return ''.join(parts)
    
    def restore_text(self, 
                     text: str,
                     top_k: int = 3,
                     min_confidence: float = 0.1) -> Dict[str, Any]:
        """
        Restore missing or damaged text spans in the input text.
        
        Args:
            text: Input text with missing spans like <MISSING-0>
            top_k: Number of candidate restorations to return
            min_confidence: Minimum confidence threshold for candidates
            
        Returns:
            Dict with keys:
              - original_text: the input text
              - restored_text: best candidate text
              - candidates: list of {text, confidence}
        """
        try:
            # Normalize user markers then extract placeholders to fill
            normalized_text = self._normalize_missing_markers(text)
            spans = self.extract_spans(normalized_text)
            if not spans:
                # Fallback to single-shot if no placeholders were found
                single = self.generate_restorations(normalized_text, top_k=top_k)
                best = single[0]['text'] if single else normalized_text
                return {
                    'original_text': text,
                    'restored_text': best,
                    'candidates': single,
                }

            # For each span, generate short candidates (span only)
            per_span_candidates: Dict[int, List[Dict[str, Any]]] = {}

            for span in spans:
                sid = span['id']
                context = span['context']
                prompt = (
                    "Fill ONLY the missing Ge'ez span for the placeholder shown. "
                    "Respond with the span text only, no extra words.\n"
                    f"Context: {context}\n"
                    f"Missing: <MISSING-{sid}>\n"
                    "Span:"
                )
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=min(self.max_length, 256),
                ).to(self.device)

                out = self.model.generate(
                    **inputs,
                    max_new_tokens=12,
                    num_beams=max(4, top_k),
                    num_return_sequences=top_k,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2,
                    length_penalty=0.8,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

                cand_list: List[Dict[str, Any]] = []
                # sequences_scores may be present for beam search
                seq_scores = getattr(out, 'sequences_scores', None)
                for i, seq in enumerate(out.sequences):
                    decoded = self.tokenizer.decode(seq, skip_special_tokens=True).strip()
                    # Clean artifacts: remove echoed labels
                    for tag in ["Span:", "Missing:", "Context:"]:
                        if decoded.startswith(tag):
                            decoded = decoded[len(tag):].strip()
                    # Trim to first phrase (avoid run-ons)
                    decoded = decoded.split('\n')[0].strip()
                    decoded = decoded.split('።')[0].strip() or decoded
                    # Confidence heuristic
                    if seq_scores is not None and i < len(seq_scores):
                        conf = min(max(float(torch.exp(seq_scores[i]).item()), 0.0), 1.0)
                    else:
                        conf = 0.6
                    cand_list.append({'text': decoded, 'confidence': conf})

                # Deduplicate while keeping order
                seen = set()
                unique_list = []
                for c in cand_list:
                    if c['text'] in seen or not c['text']:
                        continue
                    seen.add(c['text'])
                    unique_list.append(c)

                per_span_candidates[sid] = unique_list[:max(1, top_k)]

            # Build the best restoration by using top-1 for each span
            restored_text = normalized_text
            span_confidences: Dict[int, float] = {}
            for span in sorted(spans, key=lambda s: s['start'], reverse=True):
                sid = span['id']
                replacement = per_span_candidates[sid][0]['text'] if per_span_candidates.get(sid) else ''
                conf = per_span_candidates[sid][0]['confidence'] if per_span_candidates.get(sid) else 0.5
                span_confidences[sid] = conf
                restored_text = (
                    restored_text[:span['start']] + replacement + restored_text[span['end'] :]
                )

            # Construct alternative full-text candidates by swapping per-span alternatives
            full_candidates: List[Dict[str, Any]] = []
            full_candidates.append({'text': restored_text, 'confidence': float(np.mean(list(span_confidences.values())) if span_confidences else 0.6)})

            # For additional candidates, vary one span at a time using its next-best option
            for sid, cand_list in per_span_candidates.items():
                for alt in cand_list[1:]:
                    variant = text
                    # Build using top-1 for all spans, except this sid uses alt
                    for span in sorted(spans, key=lambda s: s['start'], reverse=True):
                        use_text = per_span_candidates[span['id']][0]['text']
                        if span['id'] == sid:
                            use_text = alt['text']
                        variant = variant[:span['start']] + use_text + variant[span['end'] :]
                    avg_conf = float(np.mean([
                        (alt['confidence'] if span['id'] == sid else per_span_candidates[span['id']][0]['confidence'])
                        for span in spans
                    ]))
                    full_candidates.append({'text': variant, 'confidence': avg_conf})
                    if len(full_candidates) >= max(1, top_k):
                        break
                if len(full_candidates) >= max(1, top_k):
                    break

            # Filter by confidence threshold
            filtered = [c for c in full_candidates if c['confidence'] >= float(min_confidence)] or full_candidates

            return {
                'original_text': text,
                'restored_text': filtered[0]['text'] if filtered else text,
                'candidates': filtered,
            }

        except Exception as e:
            logger.error(f"Error in restore_text: {str(e)}")
            raise


def load_restorer(model_path: Optional[str] = None, **kwargs) -> GeEzTextRestorer:
    """
    Load a text restorer from a saved model.
    
    Args:
        model_path: Path to the fine-tuned model
        **kwargs: Additional arguments for GeEzTextRestorer
        
    Returns:
        GeEzTextRestorer instance
    """
    return GeEzTextRestorer(model_path=model_path, **kwargs)


def main():
    """Command-line interface for the text restorer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Restore missing text in Ge'ez manuscripts")
    parser.add_argument('text', nargs='?', help='Text with missing spans (if not provided, enter interactive mode)')
    parser.add_argument('--model', default='models/t5_geez_span',
                      help='Path to the fine-tuned model (default: models/t5_geez_span)')
    parser.add_argument('--device', default=None,
                      help='Device to run the model on (cuda, mps, cpu)')
    parser.add_argument('--top-k', type=int, default=3,
                      help='Number of top candidates to return (default: 3)')
    parser.add_argument('--min-confidence', type=float, default=0.1,
                      help='Minimum confidence threshold (default: 0.1)')
    parser.add_argument('--temperature', type=float, default=0.7,
                      help='Sampling temperature (default: 0.7)')
    
    args = parser.parse_args()
    
    # Load the restorer
    try:
        restorer = load_restorer(
            model_path=args.model,
            device=args.device,
            num_return_sequences=args.top_k,
        )
    except Exception as e:
        logger.error(f"Error loading restorer: {e}")
        return 1
    
    # Interactive mode
    if not args.text:
        print("Entering interactive mode. Type 'exit' or 'quit' to exit.")
        print("Enter text with placeholders like <MISSING-0>, <MISSING-1>, etc.")
        
        while True:
            try:
                text = input("\nEnter text with missing spans: ").strip()
                if text.lower() in ['exit', 'quit']:
                    break
                
                if not text:
                    continue
                
                # Restore the text
                result = restorer.restore_text(text)
                
                # Print results
                print("\nOriginal text:")
                print(result.get('original_text', text))
                
                print("\nRestored text:")
                print(result.get('restored_text', ''))
                
                cands = result.get('candidates', [])
                if cands:
                    print("\nAlternative candidates:")
                    for i, cand in enumerate(cands, 1):
                        print(f"{i}. [Confidence: {cand.get('confidence', 0.0):.2f}] {cand.get('text', '')}")
            
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
    else:
        # Single text mode
        result = restorer.restore_text(args.text)
        
        # Print results
        print("\nOriginal text:")
        print(result.get('original_text', args.text))
        
        print("\nRestored text:")
        print(result.get('restored_text', ''))
        
        cands = result.get('candidates', [])
        if cands:
            print("\nAlternative candidates:")
            for i, cand in enumerate(cands, 1):
                print(f"{i}. [Confidence: {cand.get('confidence', 0.0):.2f}] {cand.get('text', '')}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
