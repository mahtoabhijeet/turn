"""
TurnGPT Training Script
Optimized for M1 MacBook Air with memory monitoring and checkpointing
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm

# Local imports
from model import TurnGPTLMHeadModel, create_turngpt_config
from turn_embedding_scaled import initialize_semantic_turns
from dataset import (
    download_sample_texts, create_tokenizer, create_dataloaders,
    create_vocab_mapping, estimate_memory_usage, get_optimal_batch_size
)

class TurnGPTTrainer:
    """
    Complete training pipeline for TurnGPT optimized for M1 MacBook
    """
    def __init__(
        self,
        model_size: str = "small",
        vocab_size: int = 50257,  # GPT-2 vocab size
        max_length: int = 256,    # Smaller for M1 efficiency
        batch_size: Optional[int] = None,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_steps: int = 5000,
        eval_steps: int = 250,
        save_steps: int = 1000,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        output_dir: str = "checkpoints",
    ):
        self.config = create_turngpt_config(vocab_size, model_size)
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.output_dir = output_dir
        
        # Device setup (M1 optimized)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("🚀 Using Apple M1 GPU (MPS)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("🚀 Using CUDA GPU")
        else:
            self.device = torch.device("cpu")
            print("🚀 Using CPU")
        
        # Auto-determine batch size if not provided
        if batch_size is None:
            available_memory = 8  # Assume 8GB, adjust if needed
            self.batch_size = get_optimal_batch_size(
                vocab_size=vocab_size,
                max_length=max_length,
                available_memory_gb=available_memory,
                n_layer=self.config.n_layer,
                n_embd=self.config.n_embd,
            )
            self.batch_size = max(1, min(self.batch_size, 8))  # Cap at 8 for M1
        else:
            self.batch_size = batch_size
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize tracking variables
        self.step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.turn_evolution = {}
    
    def setup_model_and_data(self):
        """Initialize model, tokenizer, and dataloaders"""
        print("🔧 Setting up model and data...")
        
        # Create tokenizer
        self.tokenizer = create_tokenizer()
        
        # Update config with actual vocab size
        self.config.vocab_size = self.tokenizer.vocab_size
        
        # Create model
        self.model = TurnGPTLMHeadModel(self.config).to(self.device)
        
        # Enable gradient checkpointing for memory efficiency
        self.model.transformer.enable_gradient_checkpointing()
        
        # Print model info
        memory_info = self.model.transformer.get_memory_footprint()
        print(f"📊 Model created:")
        print(f"   Total parameters: {memory_info['total_parameters']:,}")
        print(f"   Embedding compression: {memory_info['embedding_compression_ratio']:.1f}x")
        print(f"   Memory savings: {memory_info['embedding_memory_savings']:.1f}%")
        
        # Initialize semantic turns
        vocab_mapping = create_vocab_mapping(self.tokenizer)
        turn_embedding = self.model.get_semantic_calculator()
        initialized_count = initialize_semantic_turns(turn_embedding, vocab_mapping)
        print(f"   Semantically initialized: {initialized_count} words")
        
        # Create datasets
        print("📚 Loading training data...")
        texts = download_sample_texts("data")
        print(f"   Loaded {len(texts)} text samples")
        
        self.train_loader, self.val_loader = create_dataloaders(
            texts=texts,
            tokenizer=self.tokenizer,
            batch_size=self.batch_size,
            max_length=self.max_length,
            train_ratio=0.9,
        )
        
        # Setup optimizer and scheduler
        self.setup_optimizer()
        
        print("✅ Setup complete!")
    
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        # Separate parameters for different learning rates
        no_decay = ["bias", "LayerNorm.weight"]
        turn_params = ["turns"]  # Turn parameters might need different LR
        
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and not any(tp in n for tp in turn_params)
                ],
                "weight_decay": self.weight_decay,
                "lr": self.learning_rate,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": self.learning_rate,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(tp in n for tp in turn_params)
                ],
                "weight_decay": 0.0,  # Don't decay turn parameters
                "lr": self.learning_rate * 0.5,  # Lower LR for turns
            },
        ]
        
        # Use AdamW optimizer
        self.optimizer = optim.AdamW(optimizer_grouped_parameters, eps=1e-8)
        
        # Simple linear warmup scheduler
        self.warmup_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: min(1.0, step / self.warmup_steps)
        )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step"""
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs['loss']
        
        # Scale loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        return loss.item() * self.gradient_accumulation_steps
    
    def validation_step(self) -> Tuple[float, Dict[str, float]]:
        """Run validation loop"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs['loss'].item()
                num_batches += 1
                
                # Limit validation batches for speed
                if num_batches >= 10:
                    break
        
        avg_val_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        # Calculate perplexity
        perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
        
        return avg_val_loss, {'perplexity': perplexity}
    
    def test_semantic_arithmetic(self) -> Dict[str, float]:
        """Test semantic arithmetic capabilities"""
        turn_embedding = self.model.get_semantic_calculator()
        
        # Test cases (using token IDs from tokenizer)
        test_cases = [
            # Format: (word_a, word_b, word_c, expected_result)
            ("king", "man", "woman", "queen"),
            ("good", "bad", "terrible", "awful"),
            ("big", "small", "tiny", "huge"),
        ]
        
        results = {}
        for word_a, word_b, word_c, expected in test_cases:
            try:
                # Get token IDs
                a_id = self.tokenizer.encode(word_a, add_special_tokens=False)[0]
                b_id = self.tokenizer.encode(word_b, add_special_tokens=False)[0]
                c_id = self.tokenizer.encode(word_c, add_special_tokens=False)[0]
                expected_id = self.tokenizer.encode(expected, add_special_tokens=False)[0]
                
                # Perform semantic arithmetic
                result_turns, closest_id = turn_embedding.semantic_arithmetic(a_id, b_id, c_id)
                
                # Calculate distance to expected result
                expected_turns = turn_embedding.get_turn_vector(expected_id)
                distance = torch.norm(result_turns - expected_turns).item()
                
                results[f"{word_a}-{word_b}+{word_c}"] = distance
                
                # Decode the closest word found
                closest_word = self.tokenizer.decode([closest_id])
                print(f"   {word_a} - {word_b} + {word_c} = {closest_word} (expected: {expected}, distance: {distance:.3f})")
                
            except Exception as e:
                print(f"   Failed test {word_a}-{word_b}+{word_c}: {e}")
                results[f"{word_a}-{word_b}+{word_c}"] = float('inf')
        
        return results
    
    def generate_sample(self, prompt: str = "The", max_length: int = 50) -> str:
        """Generate a text sample"""
        self.model.eval()
        
        # Tokenize prompt
        input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=0.8,
                do_sample=True,
                top_k=50,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text
    
    def save_checkpoint(self, step: int, val_loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.warmup_scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config.__dict__,
        }
        
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint-{step}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = os.path.join(self.output_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"⭐ New best model saved: {best_path}")
    
    def train(self):
        """Main training loop"""
        print("🚀 Starting TurnGPT training...")
        
        # Setup
        self.setup_model_and_data()
        
        # Training loop
        self.model.train()
        running_loss = 0.0
        progress_bar = tqdm(total=self.max_steps, desc="Training")
        
        # Initialize data iterator
        train_iter = iter(self.train_loader)
        
        for step in range(self.max_steps):
            self.step = step
            
            try:
                batch = next(train_iter)
            except StopIteration:
                # Reset iterator when we run out of data
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            
            # Training step
            loss = self.train_step(batch)
            running_loss += loss
            
            # Gradient clipping and optimization step
            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                if step < self.warmup_steps:
                    self.warmup_scheduler.step()
                self.optimizer.zero_grad()
            
            # Update progress
            progress_bar.set_postfix({
                'loss': f'{running_loss/(step+1):.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                'step': step
            })
            progress_bar.update(1)
            
            # Evaluation
            if (step + 1) % self.eval_steps == 0:
                print(f"\n📊 Evaluation at step {step + 1}")
                val_loss, val_metrics = self.validation_step()
                
                print(f"   Validation loss: {val_loss:.4f}")
                print(f"   Perplexity: {val_metrics['perplexity']:.2f}")
                
                # Test semantic arithmetic
                print("   Testing semantic arithmetic:")
                arithmetic_results = self.test_semantic_arithmetic()
                
                # Generate sample
                sample_text = self.generate_sample("The king")
                print(f"   Sample generation: '{sample_text}'")
                
                # Save metrics
                self.train_losses.append(running_loss / (step + 1))
                self.val_losses.append(val_loss)
                
                self.model.train()  # Back to training mode
            
            # Save checkpoint
            if (step + 1) % self.save_steps == 0:
                val_loss, _ = self.validation_step()
                self.save_checkpoint(step + 1, val_loss)
        
        progress_bar.close()
        
        # Final evaluation
        print("\n🎯 Final evaluation:")
        val_loss, val_metrics = self.validation_step()
        print(f"Final validation loss: {val_loss:.4f}")
        print(f"Final perplexity: {val_metrics['perplexity']:.2f}")
        
        # Save final model
        self.save_checkpoint(self.max_steps, val_loss)
        
        # Test semantic arithmetic one more time
        print("\n🧮 Final semantic arithmetic test:")
        final_arithmetic = self.test_semantic_arithmetic()
        
        # Generate some samples
        print("\n📝 Final text generation samples:")
        for prompt in ["The", "Once upon", "In the"]:
            sample = self.generate_sample(prompt, max_length=40)
            print(f"   '{prompt}' → '{sample}'")
        
        print("🎉 Training complete!")
        
        return {
            'final_val_loss': val_loss,
            'final_perplexity': val_metrics['perplexity'],
            'semantic_arithmetic': final_arithmetic,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }

def main():
    """Main training function"""
    # Configuration
    trainer = TurnGPTTrainer(
        model_size="small",     # Start small for M1
        max_length=128,         # Shorter sequences for efficiency  
        batch_size=4,           # Conservative for M1
        learning_rate=3e-4,     # Slightly lower LR
        max_steps=2000,         # Shorter training for demo
        eval_steps=200,
        save_steps=500,
    )
    
    # Train
    results = trainer.train()
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Final validation loss: {results['final_val_loss']:.4f}")
    print(f"Final perplexity: {results['final_perplexity']:.2f}")
    print(f"Semantic arithmetic results:")
    for test, distance in results['semantic_arithmetic'].items():
        status = "✅" if distance < 2.0 else "⚠️" if distance < 5.0 else "❌"
        print(f"  {status} {test}: {distance:.3f}")
    
    return results

if __name__ == "__main__":
    main()
