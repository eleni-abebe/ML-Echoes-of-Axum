# Echoes of Axum: AI Reconstruction of Ge'ez Manuscripts

A deep learning system for restoring and analyzing Ge'ez manuscripts using state-of-the-art language models. This project focuses on text restoration, analysis, and translation of historical Ge'ez texts.

## ✨ Features

- **Text Restoration**: Reconstruct missing or damaged text in Ge'ez manuscripts
- **Model Analysis**: Tools for analyzing model performance and tokenization
- **Data Processing**: Utilities for cleaning and preparing Ge'ez text data
- **Evaluation**: Comprehensive evaluation scripts for model performance
- **Interactive Tools**: Scripts for testing and interacting with trained models

## 🏗️ Project Structure

```
.
├── app/                  # Web application components
├── data/                 # Data storage and processing
│   ├── raw/              # Raw text data
│   └── processed/        # Processed training data
├── models/               # Model checkpoints and configurations
├── training/             # Training logs and outputs
├── utils/                # Utility scripts
├── analyze_model.py      # Model analysis tools
├── analyze_tokenizer.py  # Tokenizer analysis
├── analyze_training_data.py  # Training data analysis
├── complete_text.py      # Text completion script
├── evaluate_model.py     # Model evaluation
├── requirements.txt      # Python dependencies
└── train_fast.py         # Main training script
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip
- CUDA-compatible GPU (recommended for training)

### Installation

1. **Clone the repository** (if not already cloned):
   ```bash
   git clone <repository-url>
   cd "AI powered manuiscript"
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # OR
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   For Ge'ez-specific requirements:
   ```bash
   pip install -r requirements_geez.txt
   ```

## 🛠️ Usage

### Training the Model

1. **Prepare your data**:
   - Place Ge'ez text files in `data/raw/`
   - Process the data using the provided scripts

2. **Start training**:
   ```bash
   python train_fast.py
   ```
   
   For more control, use the full training script:
   ```bash
   python train_geez_t5.py
   ```

### Text Completion

To generate completions for Ge'ez text:
```bash
python complete_text.py --model_path models/geez_t5_small --input "your text here"
```

### Model Evaluation

Evaluate model performance:
```bash
python evaluate_model.py --model_path models/geez_t5_small --test_data data/processed/test_data.jsonl
```

## 📊 Analysis Tools

- **Analyze tokenizer**: `python analyze_tokenizer.py`
- **Analyze training data**: `python analyze_training_data.py`
- **Inspect model**: `python inspect_model.py`

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Beta maṣāḥǝft** for Ge'ez texts
- **STEPBible** for Ge'ez scriptures
- **Hugging Face** for the transformer models
- **Abyssinica SIL** for the Ge'ez font
- **KAIST** for research support

## 📚 Resources

- [Ge'ez Script Wikipedia](https://en.wikipedia.org/wiki/Ge%27ez_script)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
