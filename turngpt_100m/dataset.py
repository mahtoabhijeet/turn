"""
Dataset preparation and tokenization for TurnGPT training
Optimized for M1 MacBook Air with efficient data loading
"""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from typing import List, Dict, Iterator, Optional, Tuple
import numpy as np
from pathlib import Path
import json
import requests
import os
from tqdm import tqdm

class TurnGPTDataset(Dataset):
    """
    Efficient dataset for TurnGPT training with M1 optimization
    """
    def __init__(
        self,
        texts: List[str],
        tokenizer: GPT2Tokenizer,
        max_length: int = 512,
        stride: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        
        # Tokenize all texts and create overlapping chunks
        self.input_ids = []
        self._process_texts(texts)
        
    def _process_texts(self, texts: List[str]):
        """Process texts into tokenized chunks with overlap"""
        print(f"Processing {len(texts)} texts into training chunks...")
        
        for text in tqdm(texts, desc="Tokenizing"):
            # Tokenize the full text
            encoded = self.tokenizer.encode(text, add_special_tokens=True)
            
            # Create overlapping chunks
            for i in range(0, len(encoded) - self.max_length + 1, self.stride):
                chunk = encoded[i:i + self.max_length]
                if len(chunk) == self.max_length:
                    self.input_ids.append(chunk)
        
        print(f"Created {len(self.input_ids)} training examples")
    
    def __len__(self) -> int:
        return len(self.input_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor(self.input_ids[idx], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'labels': input_ids.clone(),  # For language modeling
        }

def download_sample_texts(cache_dir: str = "data") -> List[str]:
    """
    Download sample texts for training (optimized for quick demo)
    Uses small, high-quality datasets perfect for M1 training
    """
    os.makedirs(cache_dir, exist_ok=True)
    texts = []
    
    # Sample text sources (small but high-quality)
    sources = [
        {
            'name': 'shakespeare',
            'url': 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
            'file': 'shakespeare.txt'
        },
        {
            'name': 'wikipedia_sample', 
            'url': 'https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/wiki.train.tokens',
            'file': 'wiki_sample.txt'
        }
    ]
    
    for source in sources:
        file_path = os.path.join(cache_dir, source['file'])
        
        if not os.path.exists(file_path):
            print(f"Downloading {source['name']}...")
            try:
                response = requests.get(source['url'])
                response.raise_for_status()
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                print(f"Downloaded {source['name']} to {file_path}")
            except Exception as e:
                print(f"Failed to download {source['name']}: {e}")
                continue
        
        # Read the downloaded text
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            # Split into reasonable chunks (paragraphs or sentences)
            chunks = content.split('\n\n')  # Split by double newline
            chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 50]
            
            texts.extend(chunks)
            print(f"Loaded {len(chunks)} text chunks from {source['name']}")
            
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
    
    # Fallback: create some sample texts if downloads fail
    if not texts:
        print("Creating fallback sample texts...")
        fallback_texts = [
            "The quick brown fox jumps over the lazy dog. This is a sample sentence for training.",
            "Machine learning and artificial intelligence are transforming the world in remarkable ways.",
            "Shakespeare wrote many famous plays including Hamlet, Romeo and Juliet, and Macbeth.",
            "The history of science is filled with brilliant discoveries and remarkable innovations.",
            "Natural language processing helps computers understand and generate human language.",
            "Deep learning models can learn complex patterns from large amounts of data.",
            "The future of AI holds great promise for solving humanity's biggest challenges.",
        ]
        texts = fallback_texts * 100  # Repeat to create more training data
    
    return texts

def create_tokenizer(vocab_size: int = 5000) -> GPT2Tokenizer:
    """
    Create or load a GPT-2 tokenizer optimized for the target vocabulary size
    """
    try:
        # Try to use standard GPT-2 tokenizer first
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        # Add pad token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"Using GPT-2 tokenizer with vocab size: {tokenizer.vocab_size}")
        return tokenizer
        
    except Exception as e:
        print(f"Failed to load GPT-2 tokenizer: {e}")
        raise

def create_dataloaders(
    texts: List[str],
    tokenizer: GPT2Tokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    train_ratio: float = 0.9,
    num_workers: int = 0,  # Set to 0 for M1 compatibility
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders optimized for M1
    """
    # Split texts into train/val
    split_idx = int(len(texts) * train_ratio)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:] if split_idx < len(texts) else texts[:10]  # Ensure we have some val data
    
    print(f"Train texts: {len(train_texts)}, Val texts: {len(val_texts)}")
    
    # Create datasets
    train_dataset = TurnGPTDataset(
        texts=train_texts,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=max_length // 2,  # 50% overlap
    )
    
    val_dataset = TurnGPTDataset(
        texts=val_texts,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=max_length,  # No overlap for validation
    )
    
    # Create collator to handle attention_mask
    collator = M1OptimizedCollator(pad_token_id=tokenizer.pad_token_id)
    
    # Create dataloaders (num_workers=0 for M1 compatibility)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # Disable for unified memory
        drop_last=True,
        collate_fn=collator,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=collator,
    )
    
    print(f"Created dataloaders - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    return train_loader, val_loader

def create_vocab_mapping(tokenizer: GPT2Tokenizer) -> Dict[str, int]:
    """
    Create a mapping from words to token IDs for semantic initialization
    """
    vocab_mapping = {}
    
    # Get the vocabulary from tokenizer
    vocab = tokenizer.get_vocab()
    
    # Filter for actual words (not subword tokens)
    for word, token_id in vocab.items():
        # Skip special tokens and subword pieces (starting with 'Ġ' in GPT-2)
        if (not word.startswith('Ġ') and 
            word.isalpha() and 
            len(word) > 1 and 
            not word.startswith('<') and 
            not word.startswith('[')
        ):
            vocab_mapping[word.lower()] = token_id
    
    print(f"Created vocab mapping for {len(vocab_mapping)} words")
    return vocab_mapping

class M1OptimizedCollator:
    """
    Custom collate function optimized for M1 unified memory
    """
    def __init__(self, pad_token_id: int = 50256):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Extract input_ids and labels
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        # All sequences in TurnGPTDataset are already the same length
        # Stack them directly (no padding needed)
        input_ids = torch.stack(input_ids, dim=0)
        labels = torch.stack(labels, dim=0)
        
        # Create attention mask (all 1s since no padding)
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

def estimate_memory_usage(
    vocab_size: int,
    max_length: int,
    batch_size: int,
    n_layer: int = 12,
    n_embd: int = 768,
    n_turns: int = 8,
) -> Dict[str, float]:
    """
    Estimate memory usage for TurnGPT training on M1
    """
    # Model parameters (MB)
    turn_params = vocab_size * n_turns * 4 / 1024 / 1024  # 4 bytes per float
    poly_params = n_turns * 5 * n_embd * 4 / 1024 / 1024  # degree 4 polynomial
    transformer_params = n_layer * (
        4 * n_embd * n_embd + 2 * n_embd * n_embd * 4  # attention + MLP
    ) * 4 / 1024 / 1024
    
    total_model_mb = turn_params + poly_params + transformer_params
    
    # Activation memory during training (MB)
    activation_mb = batch_size * max_length * n_embd * n_layer * 4 / 1024 / 1024
    
    # Optimizer memory (Adam = 2x model params)
    optimizer_mb = total_model_mb * 2
    
    # Total memory estimate
    total_mb = total_model_mb + activation_mb + optimizer_mb
    
    return {
        'model_params_mb': total_model_mb,
        'activations_mb': activation_mb,
        'optimizer_mb': optimizer_mb,
        'total_mb': total_mb,
        'total_gb': total_mb / 1024,
        'fits_8gb': total_mb < 6000,  # Leave some buffer
        'fits_16gb': total_mb < 14000,
    }

def get_optimal_batch_size(
    vocab_size: int = 50257,
    max_length: int = 512,
    available_memory_gb: int = 8,
    n_layer: int = 12,
    n_embd: int = 768,
) -> int:
    """
    Find optimal batch size for M1 memory constraints
    """
    for batch_size in [1, 2, 4, 8, 16, 32]:
        memory_info = estimate_memory_usage(
            vocab_size=vocab_size,
            max_length=max_length, 
            batch_size=batch_size,
            n_layer=n_layer,
            n_embd=n_embd,
        )
        
        memory_gb = memory_info['total_gb']
        buffer_gb = available_memory_gb * 0.8  # 80% utilization
        
        if memory_gb <= buffer_gb:
            optimal_batch_size = batch_size
        else:
            break
    
    print(f"Optimal batch size for {available_memory_gb}GB: {optimal_batch_size}")
    print(f"  Estimated memory usage: {memory_info['total_gb']:.1f}GB")
    
    return optimal_batch_size

# Quick test function
def test_dataset_creation():
    """Test dataset creation with sample data"""
    print("Testing dataset creation...")
    
    # Create sample texts
    sample_texts = [
        "This is a test sentence for the TurnGPT model training.",
        "Semantic turn theory represents meaning as discrete integers.",
        "Machine learning and artificial intelligence are fascinating fields.",
        "The quick brown fox jumps over the lazy dog repeatedly.",
    ]
    
    try:
        # Create tokenizer
        tokenizer = create_tokenizer()
        
        # Create dataset
        dataset = TurnGPTDataset(
            texts=sample_texts,
            tokenizer=tokenizer,
            max_length=64,  # Small for testing
        )
        
        # Test data loading
        sample = dataset[0]
        print(f"✅ Dataset created successfully!")
        print(f"   Sample shape: {sample['input_ids'].shape}")
        print(f"   Total samples: {len(dataset)}")
        
        # Test dataloader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(dataloader))
        print(f"   Batch shape: {batch['input_ids'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return False

if __name__ == "__main__":
    # Run tests
    test_dataset_creation()
    
    # Show memory estimates for different configurations
    print("\n" + "="*50)
    print("MEMORY ESTIMATES FOR M1 CONFIGURATIONS")
    print("="*50)
    
    configs = [
        {"name": "Tiny", "n_layer": 6, "n_embd": 384, "batch_size": 16},
        {"name": "Small", "n_layer": 8, "n_embd": 512, "batch_size": 12}, 
        {"name": "Medium", "n_layer": 12, "n_embd": 768, "batch_size": 8},
        {"name": "Large", "n_layer": 16, "n_embd": 1024, "batch_size": 4},
    ]
    
    for config in configs:
        memory_info = estimate_memory_usage(
            vocab_size=50257,
            max_length=512,
            **config
        )
        
        print(f"\n{config['name']} TurnGPT:")
        print(f"  Parameters: {memory_info['model_params_mb']:.1f}MB")
        print(f"  Total memory: {memory_info['total_gb']:.1f}GB")
        print(f"  Fits 8GB M1: {'✅' if memory_info['fits_8gb'] else '❌'}")
        print(f"  Fits 16GB M1: {'✅' if memory_info['fits_16gb'] else '❌'}")
