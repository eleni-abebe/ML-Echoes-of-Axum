import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
import numpy as np
from collections import defaultdict
import random
import torch.nn.functional as F

# Configure logging
import sys
import io

# Set stdout to handle UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(stream=sys.stdout),
        logging.FileHandler('geez_restore.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Set console to UTF-8 mode
if sys.platform.startswith('win'):
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleCP(65001)
    kernel32.SetConsoleOutputCP(65001)

class GeEzCharDataset(Dataset):
    def __init__(self, file_path, max_length=128):
        self.max_length = max_length
        self.chars = set()
        self.examples = []
        
        # First pass: collect all unique characters
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                self.chars.update(data['input'])
                self.chars.update(data['target'])
        
        # Create character to index mapping
        self.chars = sorted(list(self.chars))
        self.char_to_idx = {c: i+2 for i, c in enumerate(self.chars)}  # 0: pad, 1: sos, 2+: chars
        self.idx_to_char = {i+2: c for i, c in enumerate(self.chars)}
        self.char_to_idx['<pad>'] = 0
        self.char_to_idx['<sos>'] = 1
        self.char_to_idx['<eos>'] = 2  # Add EOS token
        self.idx_to_char[0] = '<pad>'
        self.idx_to_char[1] = '<sos>'
        self.idx_to_char[2] = '<eos>'
        
        # Second pass: process examples
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                input_text = data['input']
                target_text = data['target']
                
                # Convert to indices and add EOS token
                input_idx = [self.char_to_idx[c] for c in input_text[:self.max_length-2]]
                target_idx = [self.char_to_idx[c] for c in target_text[:self.max_length-2]]
                
                # Add start and end tokens
                input_idx = [self.char_to_idx['<sos>']] + input_idx + [self.char_to_idx['<eos>']]
                target_idx = [self.char_to_idx['<sos>']] + target_idx + [self.char_to_idx['<eos>']]
                
                # Pad sequences to max_length
                input_pad_len = max(0, self.max_length - len(input_idx))
                target_pad_len = max(0, self.max_length - len(target_idx))
                
                input_idx += [self.char_to_idx['<pad>']] * input_pad_len
                target_idx += [self.char_to_idx['<pad>']] * target_pad_len
                
                # Truncate if still too long (shouldn't happen with proper max_length)
                input_idx = input_idx[:self.max_length]
                target_idx = target_idx[:self.max_length]
                
                self.examples.append((input_idx, target_idx))
        
        self.vocab_size = len(self.char_to_idx)
        logger.info(f"Loaded {len(self.examples)} examples with {self.vocab_size} unique characters")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        input_idx, target_idx = self.examples[idx]
        return torch.tensor(input_idx, dtype=torch.long), torch.tensor(target_idx, dtype=torch.long)

    def decode_sequence(self, idx_sequence):
        return ''.join(self.idx_to_char[idx] for idx in idx_sequence if idx > 2)  # Skip <pad> and <sos>

class GeEzModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.LSTM(
            embed_dim, 
            hidden_dim,
            num_layers=num_layers,
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # Using unidirectional for simplicity
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            embed_dim + hidden_dim,  # Input: [embedding, context]
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention
        self.attention = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, x, y=None, max_length=128):
        batch_size = x.size(0)
        
        # Encoder
        x_embed = self.dropout(self.embedding(x))  # (batch_size, seq_len, embed_dim)
        encoder_outputs, (hidden, cell) = self.encoder(x_embed)
        
        # Decoder
        if y is not None:
            # Teacher forcing during training
            y_embed = self.dropout(self.embedding(y))  # (batch_size, seq_len, embed_dim)
            decoder_hidden = (hidden, cell)
            outputs = []
            
            # Initialize decoder input with <sos> token
            decoder_input = y_embed[:, :1, :]  # (batch_size, 1, embed_dim)
            
            # We'll generate one less token than the target length
            # since we're predicting the next token at each step
            for t in range(y.size(1) - 1):
                # Calculate attention weights
                attn_weights = self._attention(decoder_hidden[0][-1], encoder_outputs)  # (batch_size, seq_len)
                
                # Get context vector
                context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (batch_size, 1, hidden_dim)
                
                # Ensure tensors have correct dimensions [batch_size, 1, dim]
                if len(decoder_input.shape) < 3:
                    decoder_input = decoder_input.unsqueeze(1)
                if len(context.shape) < 3:
                    context = context.unsqueeze(1)
                
                # Combine input with context
                decoder_input_combined = torch.cat([
                    decoder_input.view(batch_size, 1, -1), 
                    context.view(batch_size, 1, -1)
                ], dim=2)  # (batch_size, 1, embed_dim + hidden_dim)
                
                # LSTM step
                output, decoder_hidden = self.decoder(decoder_input_combined, decoder_hidden)
                
                # Save output
                outputs.append(output)
                
                # Next input is current target (teacher forcing)
                if t + 1 < y_embed.size(1):
                    decoder_input = y_embed[:, t+1:t+2, :]
                else:
                    break
            
            # Stack all outputs along sequence dimension
            if outputs:
                outputs = torch.cat(outputs, dim=1)  # (batch_size, seq_len-1, hidden_dim)
                return self.fc(outputs)  # (batch_size, seq_len-1, vocab_size)
            return torch.zeros(batch_size, 0, self.fc.out_features, device=x.device)
            
        else:
            # Autoregressive generation during inference
            decoder_hidden = (hidden, cell)
            outputs = []
            
            # Start with <sos> token
            decoder_input = torch.ones(batch_size, 1, dtype=torch.long, 
                                    device=x.device) * 1  # <sos> token
            
            for _ in range(max_length):
                # Embed the input token
                decoder_embed = self.dropout(self.embedding(decoder_input))  # (batch_size, 1, embed_dim)
                
                # Calculate attention weights
                attn_weights = self._attention(decoder_hidden[0][-1], encoder_outputs)  # (batch_size, seq_len)
                
                # Get context vector
                context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (batch_size, 1, hidden_dim)
                
                # Ensure tensors have correct dimensions [batch_size, 1, dim]
                if len(decoder_embed.shape) < 3:
                    decoder_embed = decoder_embed.unsqueeze(1)
                if len(context.shape) < 3:
                    context = context.unsqueeze(1)
                
                # Combine input with context
                decoder_input_combined = torch.cat([
                    decoder_embed.view(batch_size, 1, -1), 
                    context.view(batch_size, 1, -1)
                ], dim=2)  # (batch_size, 1, embed_dim + hidden_dim)
                
                # LSTM step
                output, decoder_hidden = self.decoder(decoder_input_combined, decoder_hidden)
                
                # Get token probabilities
                output = self.fc(output)  # (batch_size, 1, vocab_size)
                outputs.append(output)
                
                # Greedy decoding
                _, topi = output.topk(1)
                decoder_input = topi.detach()
                
                # Stop if all sequences predict <eos>
                if (decoder_input == 2).all():  # <eos> token
                    break
            
            if outputs:
                return torch.cat(outputs, dim=1)  # (batch_size, seq_len, vocab_size)
            return torch.zeros(batch_size, 0, self.fc.out_features, device=x.device)
    
    def _attention(self, hidden, encoder_outputs):
        """Calculate attention weights."""
        # hidden: (batch_size, hidden_dim)
        # encoder_outputs: (batch_size, seq_len, hidden_dim)
        
        # Project hidden to same dimension as encoder_outputs
        hidden = self.attention(hidden)  # (batch_size, hidden_dim)
        
        # Calculate attention scores
        # Use matrix multiplication instead of bmm for simplicity
        scores = torch.bmm(encoder_outputs, hidden.unsqueeze(2))  # (batch_size, seq_len, 1)
        scores = scores.squeeze(2)  # (batch_size, seq_len)
        
        # Apply softmax to get attention weights
        return F.softmax(scores, dim=1)  # (batch_size, seq_len)

def train():
    # Configuration
    config = {
        'train_file': 'data/processed/small_training_data.jsonl',
        'val_file': 'data/processed/small_test_data.jsonl',
        'output_dir': 'models/geez_char_model',
        'batch_size': 32,
        'max_length': 128,
        'embed_dim': 128,
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'lr_decay': 0.95,
        'min_lr': 1e-5,
        'patience': 5,
        'num_epochs': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'grad_clip': 1.0,
        'teacher_forcing_ratio': 0.5
    }
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Save config
    with open(os.path.join(config['output_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Using device: {config['device']}")
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = GeEzCharDataset(config['train_file'], config['max_length'])
    val_dataset = GeEzCharDataset(config['val_file'], config['max_length'])
    
    # Save character mappings
    char_mappings = {
        'char_to_idx': train_dataset.char_to_idx,
        'idx_to_char': {int(k): v for k, v in train_dataset.idx_to_char.items()}
    }
    with open(os.path.join(config['output_dir'], 'char_mappings.json'), 'w', 
              encoding='utf-8') as f:
        json.dump(char_mappings, f, ensure_ascii=False, indent=2)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Initialize model
    model = GeEzModel(
        vocab_size=train_dataset.vocab_size,
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(config['device'])
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=config['lr_decay'],
        patience=config['patience'],
        min_lr=config['min_lr']
    )
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        total_batches = 0
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(config['device'])
            batch_y = batch_y.to(config['device'])
            
            # Forward pass with teacher forcing
            use_teacher_forcing = random.random() < config['teacher_forcing_ratio']
            
            if use_teacher_forcing:
                # Forward pass with teacher forcing
                outputs = model(batch_x, batch_y)
                
                # Calculate loss - outputs should be [batch_size, seq_len-1, vocab_size]
                # and targets should be [batch_size, seq_len-1]
                loss = criterion(
                    outputs.reshape(-1, outputs.size(-1)),  # [batch_size * (seq_len-1), vocab_size]
                    batch_y[:, 1:].reshape(-1)  # [batch_size * (seq_len-1)]
                )
            else:
                # Forward pass without teacher forcing
                outputs = model(batch_x, max_length=batch_y.size(1)-1)
                
                # Ensure we have outputs
                if outputs.size(1) == 0:
                    continue
                    
                # Calculate loss - align sequence lengths
                target_length = min(outputs.size(1), batch_y.size(1) - 1)
                if target_length == 0:
                    continue
                    
                loss = criterion(
                    outputs[:, :target_length].reshape(-1, outputs.size(-1)),
                    batch_y[:, 1:target_length+1].reshape(-1)
                )
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1
        
        # Calculate average loss for the epoch
        avg_train_loss = total_loss / total_batches if total_batches > 0 else 0
        
        # Validation
        model.eval()
        val_loss = 0
        total_val_batches = 0
        
        with torch.no_grad():
            for val_batch_x, val_batch_y in val_loader:
                val_batch_x = val_batch_x.to(config['device'])
                val_batch_y = val_batch_y.to(config['device'])
                
                # Forward pass without teacher forcing for validation
                val_outputs = model(val_batch_x, max_length=val_batch_y.size(1)-1)
                
                if val_outputs.size(1) == 0:
                    continue
                    
                # Calculate validation loss
                val_target_length = min(val_outputs.size(1), val_batch_y.size(1) - 1)
                if val_target_length == 0:
                    continue
                    
                val_loss += criterion(
                    val_outputs[:, :val_target_length].reshape(-1, val_outputs.size(-1)),
                    val_batch_y[:, 1:val_target_length+1].reshape(-1)
                ).item()
                total_val_batches += 1
        
        # Calculate average validation loss
        avg_val_loss = val_loss / total_val_batches if total_val_batches > 0 else float('inf')
        
        # Log training progress
        logger.info(f'Epoch {epoch+1}/{config["num_epochs"]}, ' 
                   f'Train Loss: {avg_train_loss:.4f}, '
                   f'Val Loss: {avg_val_loss:.4f}')
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'config': config,
                'char_mappings': char_mappings,
            }, os.path.join(config['output_dir'], 'best_model.pt'))
            
            logger.info(f"New best model saved with val loss: {avg_val_loss:.4f}")
            patience_counter = 0
            
            # Generate sample predictions
            sample_input = "ንጉሥ ፥ ቤተ ፥ ክርስትያን ። ነቢይ"
            predicted_text = generate_sample(
                model, 
                sample_input, 
                train_dataset.char_to_idx, 
                train_dataset.idx_to_char,
                config['device'],
                max_length=len(sample_input) * 2
            )
            
            # Log to file
            with open('predictions.txt', 'a', encoding='utf-8') as f:
                f.write(f"\nEpoch {epoch + 1}:")
                f.write(f"\n  Input: {sample_input}")
                f.write(f"\n  Predicted: {predicted_text}\n")
        else:
            patience_counter += 1
            if patience_counter >= config['patience'] * 2:  # Give it some more time with reduced LR
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break
    
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")

def evaluate_model(model, test_loader, device, max_examples=5):
    """Evaluate the model on the test set and print sample predictions."""
    model.eval()
    total_loss = 0
    total_examples = 0
    
    # Initialize metrics
    total_chars = 0
    correct_chars = 0
    total_words = 0
    correct_words = 0
    
    # For storing examples
    examples = []
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding
    
    with torch.no_grad():
        for batch_idx, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_x, max_length=batch_y.size(1)-1)
            
            # Calculate loss
            if outputs.size(1) > 0:
                target_length = min(outputs.size(1), batch_y.size(1) - 1)
                if target_length > 0:
                    loss = criterion(
                        outputs[:, :target_length].reshape(-1, outputs.size(-1)),
                        batch_y[:, 1:target_length+1].reshape(-1)
                    )
                    total_loss += loss.item() * batch_x.size(0)
                    total_examples += batch_x.size(0)
            
            # Get predictions
            _, predicted = torch.max(outputs, dim=2)  # [batch_size, seq_len]
            
            # Convert to CPU for processing
            batch_x = batch_x.cpu().numpy()
            batch_y = batch_y.cpu().numpy()
            predicted = predicted.cpu().numpy()
            
            # Process each example in the batch
            for i in range(min(len(batch_x), max_examples - len(examples))):
                # Convert token indices back to characters
                input_text = test_loader.dataset.decode_sequence(batch_x[i])
                target_text = test_loader.dataset.decode_sequence(batch_y[i][1:])  # Skip <sos>
                pred_text = test_loader.dataset.decode_sequence(predicted[i])
                
                # Calculate character-level accuracy
                min_len = min(len(target_text), len(pred_text))
                if min_len > 0:
                    char_matches = sum(1 for a, b in zip(target_text[:min_len], pred_text[:min_len]) if a == b)
                    total_chars += min_len
                    correct_chars += char_matches
                
                # Calculate word-level accuracy (space-separated)
                target_words = target_text.split()
                pred_words = pred_text.split()
                min_words = min(len(target_words), len(pred_words))
                if min_words > 0:
                    word_matches = sum(1 for a, b in zip(target_words[:min_words], pred_words[:min_words]) if a == b)
                    total_words += min_words
                    correct_words += word_matches
                
                # Store example
                examples.append((input_text, target_text, pred_text))
                
                if len(examples) >= max_examples:
                    break
    
    # Calculate metrics
    avg_loss = total_loss / total_examples if total_examples > 0 else float('inf')
    char_accuracy = (correct_chars / total_chars) * 100 if total_chars > 0 else 0
    word_accuracy = (correct_words / total_words) * 100 if total_words > 0 else 0
    
    # Print metrics
    print("\n" + "="*50)
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Character-level Accuracy: {char_accuracy:.2f}%")
    print(f"Word-level Accuracy: {word_accuracy:.2f}%")
    
    # Print examples
    print("\nSample Predictions:")
    print("-"*50)
    for i, (input_text, target, pred) in enumerate(examples, 1):
        print(f"Example {i}:")
        print(f"  Input:    {input_text}")
        print(f"  Target:   {target}")
        print(f"  Predicted: {pred}")
        print("-"*50)
    
    return avg_loss, char_accuracy, word_accuracy

def generate_sample(model, sample_input, char_to_idx, idx_to_char, device, max_length=128):
    model.eval()
    with torch.no_grad():
        # Convert input to indices
        input_idx = [char_to_idx.get(c, 1) for c in sample_input]  # 1 is <sos>
        input_tensor = torch.tensor([input_idx], device=device, dtype=torch.long)
        
        # Generate output
        output = model(input_tensor, max_length=max_length)
        
        # Convert output indices to text
        _, predicted = output.max(dim=-1)
        predicted = predicted[0].cpu().numpy()
        
        # Convert to text, stopping at EOS token
        result = []
        for idx in predicted:
            if idx == 2:  # EOS token
                break
            if idx > 2:  # Skip <pad> and <sos> tokens
                result.append(idx_to_char[idx])
        
        return ''.join(result)

if __name__ == "__main__":
    # Configure logging to both console and file
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # File handler
    file_handler = logging.FileHandler('training.log', mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Set up console encoding for Windows
    if sys.platform == 'win32':
        import io
        import sys
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    # Clear previous predictions file
    if os.path.exists('predictions.txt'):
        os.remove('predictions.txt')
    
    try:
        # Start training
        train()
        
        # After training completes, show sample predictions
        if os.path.exists('predictions.txt'):
            logger.info("\nSample predictions from training:")
            with open('predictions.txt', 'r', encoding='utf-8') as f:
                print(f.read())
                
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}", exc_info=True)
        raise
    
    # Load test dataset
    test_dataset = GeEzCharDataset('data/processed/small_test_data.jsonl')
    
    # Initialize model with test dataset's vocabulary
    model = GeEzModel(
        vocab_size=test_dataset.vocab_size,
        embed_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3
    )
    
    # Try to load the best model, if it exists
    model_path = 'best_model.pt'
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            logger.info(f"Loaded best model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            logger.info("Proceeding with randomly initialized model")
    else:
        logger.warning(f"Model file {model_path} not found. Using untrained model.")
    
    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create test data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    # Run evaluation
    test_loss, char_acc, word_acc = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        max_examples=10
    )
    
    logger.info(f"\nTest Results:")
    logger.info(f"- Loss: {test_loss:.4f}")
    logger.info(f"- Character Accuracy: {char_acc:.2f}%")
    logger.info(f"- Word Accuracy: {word_acc:.2f}%")
