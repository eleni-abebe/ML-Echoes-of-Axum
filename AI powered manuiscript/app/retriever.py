#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BM25-based retriever for finding similar text passages.

This module provides functionality to index and search text passages using the BM25 algorithm,
which is effective for finding relevant documents based on keyword matching.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class BM25Retriever:
    """BM25-based retriever for finding similar text passages."""
    
    def __init__(self, corpus_path: Optional[str] = None, k1: float = 1.5, b: float = 0.75):
        """
        Initialize the BM25 retriever.
        
        Args:
            corpus_path: Path to the JSONL corpus file
            k1: BM25 parameter k1 (default: 1.5)
            b: BM25 parameter b (default: 0.75)
        """
        self.corpus: List[Dict[str, Any]] = []
        self.documents: List[str] = []
        self.ids: List[str] = []
        self.bm25: Optional[BM25Okapi] = None
        self.k1 = k1
        self.b = b
        
        if corpus_path:
            self.load_corpus(corpus_path)
    
    def load_corpus(self, corpus_path: Union[str, Path]) -> None:
        """
        Load a corpus from a JSONL file.
        
        Args:
            corpus_path: Path to the JSONL corpus file
        """
        corpus_path = Path(corpus_path)
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
        
        logger.info(f"Loading corpus from {corpus_path}...")
        self.corpus = []
        self.documents = []
        self.ids = []
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading documents"):
                try:
                    record = json.loads(line)
                    if 'text' not in record or not record['text'].strip():
                        continue
                    
                    self.corpus.append(record)
                    self.documents.append(record['text'])
                    self.ids.append(record.get('id', str(len(self.ids))))
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing JSON line: {e}")
                    continue
        
        logger.info(f"Loaded {len(self.documents)} documents from corpus")
        
        # Tokenize documents for BM25
        tokenized_docs = [self._tokenize(doc) for doc in tqdm(self.documents, desc="Tokenizing documents")]
        
        # Initialize BM25
        self.bm25 = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)
        logger.info("BM25 index created")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words (simple whitespace tokenizer).
        
        For Ge'ez script, we might want to use a more sophisticated tokenizer
        in the future, but whitespace tokenization is a good start.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        return text.split()
    
    def search(self, 
              query: str, 
              top_k: int = 5, 
              min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Search query
            top_k: Maximum number of results to return
            min_score: Minimum BM25 score for a document to be included
            
        Returns:
            List of matching documents with scores, sorted by relevance
        """
        if not self.bm25 or not self.corpus:
            raise ValueError("No corpus loaded. Call load_corpus() first.")
        
        # Tokenize query
        tokenized_query = self._tokenize(query)
        
        # Get scores for all documents
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Prepare results
        results = []
        for idx in top_indices:
            if scores[idx] < min_score:
                continue
                
            result = {
                'id': self.ids[idx],
                'text': self.documents[idx],
                'score': float(scores[idx]),
                'source': self.corpus[idx].get('source', ''),
                'metadata': {k: v for k, v in self.corpus[idx].items() 
                           if k not in ['text', 'id', 'source']},
            }
            results.append(result)
        
        # Sort by score (descending)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by its ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document dictionary or None if not found
        """
        try:
            idx = self.ids.index(doc_id)
            return {
                'id': self.ids[idx],
                'text': self.documents[idx],
                'source': self.corpus[idx].get('source', ''),
                'metadata': {k: v for k, v in self.corpus[idx].items() 
                           if k not in ['text', 'id', 'source']},
            }
        except ValueError:
            return None


def load_retriever(corpus_path: Optional[str] = None, 
                  cache_dir: Optional[str] = None,
                  **kwargs) -> BM25Retriever:
    """
    Load or create a BM25 retriever.
    
    Args:
        corpus_path: Path to the JSONL corpus file
        cache_dir: Directory to save/load the retriever cache
        **kwargs: Additional arguments for BM25Retriever
        
    Returns:
        BM25Retriever instance
    """
    import os
    import pickle
    
    # Try to load from cache if available
    if cache_dir and corpus_path:
        os.makedirs(cache_dir, exist_ok=True)
        corpus_mtime = os.path.getmtime(corpus_path)
        cache_file = os.path.join(cache_dir, f"bm25_retriever_{os.path.basename(corpus_path)}.pkl")
        
        if os.path.exists(cache_file) and os.path.getmtime(cache_file) > corpus_mtime:
            try:
                with open(cache_file, 'rb') as f:
                    logger.info(f"Loading retriever from cache: {cache_file}")
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Error loading retriever from cache: {e}")
    
    # Create new retriever
    retriever = BM25Retriever(corpus_path, **kwargs)
    
    # Save to cache if cache_dir is provided
    if cache_dir and corpus_path:
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(retriever, f)
            logger.info(f"Saved retriever to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Error saving retriever to cache: {e}")
    
    return retriever


def main():
    """Command-line interface for the retriever."""
    import argparse
    
    parser = argparse.ArgumentParser(description='BM25 Retriever for Ge\'ez Text')
    parser.add_argument('query', nargs='?', help='Search query (if not provided, enter interactive mode)')
    parser.add_argument('--corpus', default='data/corpus.jsonl',
                      help='Path to the corpus JSONL file (default: data/corpus.jsonl)')
    parser.add_argument('--top-k', type=int, default=5,
                      help='Number of results to return (default: 5)')
    parser.add_argument('--min-score', type=float, default=0.0,
                      help='Minimum BM25 score (default: 0.0)')
    parser.add_argument('--cache-dir', default='.cache',
                      help='Cache directory (default: .cache)')
    
    args = parser.parse_args()
    
    # Load the retriever
    try:
        retriever = load_retriever(
            corpus_path=args.corpus,
            cache_dir=args.cache_dir,
            k1=1.5,
            b=0.75
        )
    except Exception as e:
        logger.error(f"Error loading retriever: {e}")
        return 1
    
    # Interactive mode
    if not args.query:
        print("Entering interactive mode. Type 'exit' or 'quit' to exit.")
        while True:
            try:
                query = input("\nEnter search query: ").strip()
                if query.lower() in ['exit', 'quit']:
                    break
                
                if not query:
                    continue
                
                results = retriever.search(query, top_k=args.top_k, min_score=args.min_score)
                
                if not results:
                    print("No results found.")
                    continue
                
                print(f"\nFound {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. [Score: {result['score']:.4f}] (ID: {result['id']})")
                    print(f"   Text: {result['text']}")
                    if 'source' in result and result['source']:
                        print(f"   Source: {result['source']}")
            
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
    else:
        # Single query mode
        results = retriever.search(args.query, top_k=args.top_k, min_score=args.min_score)
        
        if not results:
            print("No results found.")
            return 0
        
        print(f"Found {len(results)} results for query: {args.query}")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. [Score: {result['score']:.4f}] (ID: {result['id']})")
            print(f"   Text: {result['text']}")
            if 'source' in result and result['source']:
                print(f"   Source: {result['source']}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
