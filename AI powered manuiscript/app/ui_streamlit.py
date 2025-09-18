#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Echoes of Axum - Streamlit Demo App

This module provides a Streamlit-based user interface for the Echoes of Axum project,
which focuses on restoring and translating Ge'ez manuscripts.
"""

import os
import sys
import json
import re
import logging
import base64
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import streamlit as st
from streamlit import session_state as state
import torch
from PIL import Image, UnidentifiedImageError

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

# Import our modules
from app.restore import GeEzTextRestorer, load_restorer
from app.translate import GeEzTranslator, load_translator
from app.retriever import BM25Retriever, load_retriever

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Path constants
BASE_DIR = Path(__file__).parent.parent.resolve()

# Default model/corpus locations
DEFAULT_MODEL_PATHS = {
    'restorer': 'google/byt5-small',  # Use HF repo by default (local dirs may be empty)
    'translator': 'google/mt5-small',  # Use HF repo by default (local dirs may be empty)
    'retriever': str((BASE_DIR / 'data' / 'corpus.jsonl').resolve()),
}

# Language options
LANGUAGES = {
    'gez': "Ge'ez",
    'amh': 'Amharic',
    'eng': 'English',
}

# Page config
st.set_page_config(
    page_title="Echoes of Axum - Ge'ez Manuscript Restoration",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Background image (minimal: load if exists, otherwise skip)
def _apply_background():
    try:
        assets_dir = Path(__file__).parent / 'assets'
        img_path = assets_dir / 'background.jpg'
        if img_path.exists():
            with open(img_path, 'rb') as f:
                data = base64.b64encode(f.read()).decode()
            st.markdown(
                f"""
                <style>
                .stApp {{
                    background-image: url('data:image/jpg;base64,{data}');
                    background-size: cover;
                    background-position: center center;
                    background-attachment: fixed;
                    background-repeat: no-repeat;
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )
    except Exception:
        # Fail silently to keep behavior unchanged if image cannot be loaded
        pass

_apply_background()

# Custom CSS
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .stTextArea textarea {
        min-height: 200px;
    }
    .result-box {
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


def load_models() -> Tuple[GeEzTextRestorer, GeEzTranslator, BM25Retriever]:
    """Load all required models."""
    # Check if models are already loaded
    if 'models_loaded' in state and state.models_loaded:
        return state.restorer, state.translator, state.retriever
    
    # Show loading message
    with st.spinner("Loading models (this may take a minute the first time)..."):
        try:
            # Load models
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load restorer
            restorer = load_restorer(
                model_path=state.get('model_paths', {}).get('restorer', DEFAULT_MODEL_PATHS['restorer']),
                device=device,
            )
            
            # Load translator
            translator = load_translator(
                model_path=state.get('model_paths', {}).get('translator', DEFAULT_MODEL_PATHS['translator']),
                device=device,
            )
            
            # Load retriever
            retriever = load_retriever(
                corpus_path=state.get('model_paths', {}).get('retriever', DEFAULT_MODEL_PATHS['retriever']),
                cache_dir='.cache',
            )
            
            # Store in session state
            state.restorer = restorer
            state.translator = translator
            state.retriever = retriever
            state.models_loaded = True
            
            return restorer, translator, retriever
            
        except Exception as e:
            st.error(f"Error loading models: {e}")
            st.stop()


def get_confidence_color(confidence: float) -> str:
    """Get CSS class for confidence level."""
    if confidence >= 0.7:
        return "confidence-high"
    elif confidence >= 0.4:
        return "confidence-medium"
    else:
        return "confidence-low"


def display_translation_result(result: Dict[str, Any]):
    """Display translation result in a formatted way."""
    with st.expander("Translation Results", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Source Text")
            st.markdown(f"**{LANGUAGES.get(result['source_lang'], result['source_lang'])}**")
            st.markdown(f"<div class='result-box'>{result['source_text']}</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Translated Text")
            st.markdown(f"**{LANGUAGES.get(result['target_lang'], result['target_lang'])}**")
            st.markdown(f"<div class='result-box'>{result['translated_text']}</div>", unsafe_allow_html=True)
        
        # Show confidence
        conf_class = get_confidence_color(result['confidence'])
        st.markdown(f"**Confidence:** <span class='{conf_class}'>{result['confidence']:.1%}</span>", unsafe_allow_html=True)


def display_restoration_result(result: Dict[str, Any]):
    """Display restoration result in a formatted way."""
    with st.expander("Restoration Results", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Original Text")
            st.markdown(f"<div class='result-box'>{result['original_text']}</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Restored Text")
            st.markdown(f"<div class='result-box'>{result['restored_text']}</div>", unsafe_allow_html=True)
        
        # Show candidates if available
        if 'candidates' in result and result['candidates']:
            st.markdown("### Alternative Restorations")
            for i, cand in enumerate(result['candidates'], 1):
                conf_class = get_confidence_color(cand['confidence'])
                st.markdown(f"**{i}.** <span class='{conf_class}'>({cand['confidence']:.1%})</span> {cand['text']}", 
                           unsafe_allow_html=True)


def display_evidence_results(results: List[Dict[str, Any]]):
    """Display evidence retrieval results."""
    if not results:
        st.warning("No relevant evidence found.")
        return
    
    st.markdown("### Supporting Evidence")
    
    for i, doc in enumerate(results, 1):
        with st.expander(f"Evidence #{i} (Score: {doc['score']:.2f})", expanded=i == 1):
            st.markdown(doc['text'])
            
            # Show source if available
            if 'source' in doc and doc['source']:
                st.caption(f"Source: {doc['source']}")
            
            # Show metadata if available
            if 'metadata' in doc and doc['metadata']:
                with st.expander("Metadata"):
                    st.json(doc['metadata'])


def main():
    """Main function for the Streamlit app."""
    # Initialize session state
    if 'models_loaded' not in state:
        state.models_loaded = False
    if 'model_paths' not in state:
        state.model_paths = DEFAULT_MODEL_PATHS.copy()
    
    # Sidebar with model settings
    with st.sidebar:
        st.title("⚙️ Settings")
        
        st.markdown("### Model Paths")
        state.model_paths['restorer'] = st.text_input(
            "Restoration Model Path",
            value=state.model_paths.get('restorer', DEFAULT_MODEL_PATHS['restorer']),
            help="Path to the fine-tuned ByT5 model for text restoration."
        )
        
        state.model_paths['translator'] = st.text_input(
            "Translation Model Path",
            value=state.model_paths.get('translator', DEFAULT_MODEL_PATHS['translator']),
            help="Path to the fine-tuned mT5 model for translation."
        )
        
        state.model_paths['retriever'] = st.text_input(
            "Corpus Path",
            value=state.model_paths.get('retriever', DEFAULT_MODEL_PATHS['retriever']),
            help="Path to the JSONL corpus file for evidence retrieval."
        )
        
        # Reset to defaults
        if st.button("Reset Paths to Defaults"):
            state.model_paths = DEFAULT_MODEL_PATHS.copy()
            state.models_loaded = False
            state.pop('restorer', None)
            state.pop('translator', None)
            state.pop('retriever', None)
            st.rerun()
        
        # Reload models if paths changed
        if st.button("Reload Models"):
            state.models_loaded = False
            state.pop('restorer', None)
            state.pop('translator', None)
            state.pop('retriever', None)
            st.rerun()
        
        st.markdown("---")
        
        # About section
        st.markdown("### About")
        st.markdown("""
        **Echoes of Axum** is an AI-powered tool for restoring and translating 
        Ge'ez manuscripts. It can help reconstruct missing text and translate 
        between Ge'ez, Amharic, and English.
        
        *Missing text should be marked with `<MISSING-0>`, `<MISSING-1>`, etc.*
        """)
    
    # Main content
    st.title("📜 Echoes of Axum")
    st.markdown("### Ge'ez Manuscript Restoration & Translation")
    
    # Tab interface
    tab1, tab2 = st.tabs(["Restore Text", "Translate Text"])
    
    with tab1:
        st.markdown("""
        Restore missing or damaged text in Ge'ez manuscripts. 
        Mark missing sections with `<MISSING-0>`, `<MISSING-1>`, etc.
        """)
        
        # Text input
        input_text = st.text_area(
            "Enter text with missing spans:",
            height=200,
            placeholder="ወይኩኑ ፡ ለአብርሆ ፡ ውስተ ፡ <missing-0>፡ ሰማይ ፡ ከመ ፡ <missing-1>፡ ዲበ ፡ ምድር ፡ ወኮነ ፡ ከማሁ ።",
            help="Mark missing sections with <MISSING-0>, <MISSING-1>, etc."
        )
        
        # Restore button
        if st.button("Restore Text"):
            if not input_text.strip():
                st.warning("Please enter some text to restore.")
            else:
                # Warn if no missing markers are present
                if not re.search(r"<MISSING-\d+>", input_text):
                    st.info("No <MISSING-N> markers found; the model will attempt a full rewrite, but results may be weak. Please add at least one marker like <MISSING-0>.")
                with st.spinner("Restoring missing text..."):
                    try:
                        # Load models if needed
                        restorer, _, retriever = load_models()
                        
                        # Restore text
                        result = restorer.restore_text(
                            input_text,
                            top_k=3,
                            min_confidence=0.1,
                        )
                        
                        # Display results
                        display_restoration_result(result)
                        
                        # Get evidence for the restored text
                        with st.spinner("Searching for supporting evidence..."):
                            evidence = retriever.search(
                                result['restored_text'],
                                top_k=3,
                                min_score=0.0,
                            )
                            display_evidence_results(evidence)
                        
                    except Exception as e:
                        st.error(f"Error during text restoration: {e}")
    
    with tab2:
        st.markdown("""
        Translate between Ge'ez, Amharic, and English. 
        The source language can be auto-detected.
        """)
        
        # Language selection
        col1, col2 = st.columns(2)
        
        with col1:
            source_lang = st.selectbox(
                "Source Language",
                options=["auto"] + list(LANGUAGES.keys()),
                format_func=lambda x: "Auto-detect" if x == "auto" else f"{x} - {LANGUAGES.get(x, 'Unknown')}",
                index=0,
            )
        
        with col2:
            target_lang = st.selectbox(
                "Target Language",
                options=[k for k in LANGUAGES.keys() if k != source_lang or source_lang == "auto"],
                format_func=lambda x: f"{x} - {LANGUAGES.get(x, 'Unknown')}",
                index=0,
            )
        
        # Text input
        input_text = st.text_area(
            "Enter text to translate:",
            height=200,
            placeholder="ይህ የጥንታዊ ጽሑፍ ነው።",
        )
        
        # Translate button
        if st.button("Translate"):
            if not input_text.strip():
                st.warning("Please enter some text to translate.")
            else:
                with st.spinner("Translating..."):
                    try:
                        # Load models if needed
                        _, translator, _ = load_models()
                        
                        # Translate text
                        result = translator.translate(
                            text=input_text,
                            source_lang=source_lang if source_lang != "auto" else None,
                            target_lang=target_lang,
                        )
                        
                        # Display results
                        display_translation_result(result)
                        
                    except Exception as e:
                        st.error(f"Error during translation: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.9em; margin-top: 2rem;'>
            <p>Echoes of Axum - AI-Powered Ge'ez Manuscript Restoration</p>
            <p> 2023 - KAIST Project</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
