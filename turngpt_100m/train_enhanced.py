"""
Enhanced TurnGPT Training Pipeline
3-Phase curriculum learning for conversation-quality AI
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
import numpy as np

# Local imports
from model import TurnGPTLMHeadModel, create_turngpt_config
from turn_embedding_scaled import initialize_semantic_turns
from dataset import (
    download_sample_texts, create_tokenizer, create_dataloaders,
    create_vocab_mapping, estimate_memory_usage, get_optimal_batch_size
)

class EnhancedTurnGPTTrainer:
    """
    Enhanced 3-Phase Training Pipeline for Conversation Quality
    Phase 1: Foundation (Grammar + Basic Structure)
    Phase 2: Conversation (Dialog Patterns + Context)  
    Phase 3: Refinement (Quality + Personality)
    """
    def __init__(
        self,
        model_size: str = "small",
        vocab_size: int = 50257,  
        output_dir: str = "checkpoints",
    ):
        self.model_size = model_size
        self.vocab_size = vocab_size
        self.output_dir = output_dir
        
        # 3-Phase Curriculum Configuration
        self.curriculum_phases = [
            {
                "name": "Foundation",
                "steps": (0, 3000),
                "max_length": 64,
                "batch_size": 6,
                "learning_rate": 3e-4,
                "focus": "grammar_structure",
                "target_perplexity": 50.0,
                "target_arithmetic_distance": 3.0,
            },
            {
                "name": "Conversation", 
                "steps": (3000, 8000),
                "max_length": 128,
                "batch_size": 4,
                "learning_rate": 2e-4,
                "focus": "dialog_patterns",
                "target_perplexity": 25.0,
                "target_arithmetic_distance": 2.0,
            },
            {
                "name": "Refinement",
                "steps": (8000, 12000), 
                "max_length": 256,
                "batch_size": 3,
                "learning_rate": 1e-4,
                "focus": "quality_personality",
                "target_perplexity": 15.0,
                "target_arithmetic_distance": 1.0,
            }
        ]
        
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
        
        # Training state
        self.step = 0
        self.current_phase = 0
        self.best_val_loss = float('inf')
        self.training_log = []
        
        # Metrics tracking
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'perplexity': [],
            'semantic_arithmetic': [],
            'conversation_quality': [],
            'phase_transitions': [],
        }
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize reporting
        self.report_path = os.path.join(self.output_dir, "training_progress.md")
        self.initialize_report()
    
    def initialize_report(self):
        """Initialize the training progress report"""
        with open(self.report_path, 'w') as f:
            f.write("# TurnGPT Enhanced Training Progress Report\n\n")
            f.write(f"**Training Started**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Model Size**: {self.model_size}\n")
            f.write(f"**Total Steps**: 12,000 (3 phases)\n")
            f.write(f"**Device**: {self.device}\n\n")
            f.write("---\n\n")
    
    def update_report(self, phase_info: Dict):
        """Update the progress report with current metrics"""
        with open(self.report_path, 'a') as f:
            f.write(f"## Phase {phase_info['phase_num']}: {phase_info['phase_name']}\n")
            f.write(f"**Steps**: {phase_info['steps_range']}\n")
            f.write(f"**Status**: {phase_info['status']}\n")
            f.write(f"**Current Step**: {self.step}\n")
            f.write(f"**Loss**: {phase_info['loss']:.4f}\n")
            f.write(f"**Perplexity**: {phase_info['perplexity']:.2f}\n")
            f.write(f"**Semantic Arithmetic**: {phase_info['arithmetic_distance']:.3f}\n\n")
            
            if 'sample_generation' in phase_info:
                f.write(f"**Sample Generation**:\n")
                f.write(f"```\n{phase_info['sample_generation']}\n```\n\n")
            
            f.write("---\n\n")
    
    def setup_model_and_data(self):
        """Initialize model, tokenizer, and enhanced semantic initialization"""
        print("🔧 Setting up enhanced model and data...")
        
        # Create tokenizer
        self.tokenizer = create_tokenizer()
        
        # Create model config
        self.config = create_turngpt_config(self.tokenizer.vocab_size, self.model_size)
        
        # Create model
        self.model = TurnGPTLMHeadModel(self.config).to(self.device)
        self.model.transformer.enable_gradient_checkpointing()
        
        # Enhanced semantic initialization (19 → 500+ words)
        print("🧠 Applying enhanced semantic initialization...")
        vocab_mapping = create_vocab_mapping(self.tokenizer)
        turn_embedding = self.model.get_semantic_calculator()
        initialized_count = initialize_semantic_turns(turn_embedding, vocab_mapping)
        print(f"   ✅ Initialized {initialized_count} words with semantic patterns")
        
        # Print model info
        memory_info = self.model.transformer.get_memory_footprint()
        compression_stats = turn_embedding.get_compression_stats()
        
        print(f"📊 Enhanced Model Statistics:")
        print(f"   Total parameters: {memory_info['total_parameters']:,}")
        print(f"   Compression ratio: {compression_stats['compression_ratio']:.1f}x")
        print(f"   Memory savings: {compression_stats['memory_savings_percent']:.1f}%")
        print(f"   Semantic coverage: {initialized_count} words")
        
        # Load training data
        print("📚 Loading enhanced training data...")
        texts = download_sample_texts("data")
        print(f"   Loaded {len(texts)} text samples")
        
        # Store for phase-specific data loading
        self.training_texts = texts
        
        print("✅ Enhanced setup complete!")
    
    def setup_phase(self, phase_num: int):
        """Setup training parameters for current phase"""
        phase = self.curriculum_phases[phase_num]
        self.current_phase = phase_num
        
        print(f"\n🎯 Starting Phase {phase_num + 1}: {phase['name']}")
        print(f"   Steps: {phase['steps'][0]} → {phase['steps'][1]}")
        print(f"   Focus: {phase['focus']}")
        print(f"   Max length: {phase['max_length']}")
        print(f"   Batch size: {phase['batch_size']}")
        print(f"   Learning rate: {phase['learning_rate']}")
        
        # Create phase-specific data loaders
        self.train_loader, self.val_loader = create_dataloaders(
            texts=self.training_texts,
            tokenizer=self.tokenizer,
            batch_size=phase['batch_size'],
            max_length=phase['max_length'],
            train_ratio=0.9,
        )
        
        # Setup optimizer for this phase
        self.setup_optimizer(phase['learning_rate'])
        
        # Log phase transition
        self.metrics_history['phase_transitions'].append({
            'step': self.step,
            'phase': phase['name'],
            'phase_num': phase_num
        })
        
        # Update report
        self.update_report({
            'phase_num': phase_num + 1,
            'phase_name': phase['name'],
            'steps_range': f"{phase['steps'][0]}-{phase['steps'][1]}",
            'status': 'Starting',
            'loss': 0.0,
            'perplexity': 0.0,
            'arithmetic_distance': 0.0
        })
    
    def setup_optimizer(self, learning_rate: float):
        """Setup optimizer with different learning rates for turn parameters"""
        no_decay = ["bias", "LayerNorm.weight"]
        turn_params = ["turns"]
        
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and not any(tp in n for tp in turn_params)
                ],
                "weight_decay": 0.01,
                "lr": learning_rate,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": learning_rate,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(tp in n for tp in turn_params)
                ],
                "weight_decay": 0.0,
                "lr": learning_rate * 0.5,  # Lower LR for turns
            },
        ]
        
        self.optimizer = optim.AdamW(optimizer_grouped_parameters, eps=1e-8)
    
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
        
        # Backward pass
        loss.backward()
        
        return loss.item()
    
    def validation_step(self) -> Tuple[float, Dict[str, float]]:
        """Run validation with enhanced metrics"""
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
                if num_batches >= 20:  # More validation than original
                    break
        
        avg_val_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
        
        return avg_val_loss, {'perplexity': perplexity}
    
    def test_semantic_arithmetic(self) -> Dict[str, float]:
        """Enhanced semantic arithmetic testing"""
        turn_embedding = self.model.get_semantic_calculator()
        
        # Comprehensive test cases
        test_cases = [
            # Classic cases
            ("king", "man", "woman", "queen"),
            ("good", "bad", "terrible", "awful"),
            ("big", "small", "tiny", "huge"),
            
            # Conversation-specific cases
            ("hello", "formal", "informal", "hi"),
            ("happy", "positive", "negative", "sad"),
            ("today", "present", "past", "yesterday"),
            
            # New categories from enhanced init
            ("computer", "technology", "nature", "tree"),
            ("doctor", "formal", "informal", "friend"),
        ]
        
        results = {}
        total_distance = 0
        successful_tests = 0
        
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
                total_distance += distance
                successful_tests += 1
                
                # Decode the closest word found
                closest_word = self.tokenizer.decode([closest_id])
                
            except Exception as e:
                results[f"{word_a}-{word_b}+{word_c}"] = float('inf')
        
        # Calculate average distance
        avg_distance = total_distance / successful_tests if successful_tests > 0 else float('inf')
        results['average_distance'] = avg_distance
        
        return results
    
    def test_conversation_quality(self) -> Dict[str, float]:
        """Test conversation-specific capabilities"""
        test_prompts = [
            "Hello, how are you?",
            "What's your favorite color?", 
            "Can you help me?",
            "Tell me about yourself.",
            "What do you think about AI?",
        ]
        
        quality_scores = []
        self.model.eval()
        
        for prompt in test_prompts:
            try:
                # Generate response
                input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)
                
                with torch.no_grad():
                    output_ids = self.model.generate(
                        input_ids,
                        max_length=len(input_ids[0]) + 30,
                        temperature=0.7,
                        do_sample=True,
                        top_k=50,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                
                response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                response = response[len(prompt):].strip()
                
                # Simple quality heuristics
                score = 0.0
                if len(response) > 5:  # Has substantial response
                    score += 0.3
                if any(word in response.lower() for word in ['i', 'my', 'me']):  # Personal response
                    score += 0.2
                if '?' in response:  # Asks follow-up questions
                    score += 0.2
                if not any(char in response for char in ['@', '#', '\\', '|']):  # No gibberish chars
                    score += 0.3
                
                quality_scores.append(score)
                
            except Exception:
                quality_scores.append(0.0)
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        return {
            'conversation_quality_score': avg_quality,
            'individual_scores': quality_scores
        }
    
    def generate_sample(self, prompt: str = "Hello", max_length: int = 50) -> str:
        """Generate enhanced text sample"""
        self.model.eval()
        
        input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=0.8,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text
    
    def evaluate_phase_completion(self, phase_num: int, val_loss: float, arithmetic_results: Dict) -> bool:
        """Check if phase completion criteria are met"""
        phase = self.curriculum_phases[phase_num]
        
        # Calculate metrics
        perplexity = torch.exp(torch.tensor(val_loss)).item()
        avg_arithmetic_distance = arithmetic_results.get('average_distance', float('inf'))
        
        # Check completion criteria
        perplexity_met = perplexity <= phase['target_perplexity']
        arithmetic_met = avg_arithmetic_distance <= phase['target_arithmetic_distance']
        
        print(f"   📊 Phase {phase_num + 1} Progress:")
        print(f"      Perplexity: {perplexity:.2f} (target: ≤{phase['target_perplexity']})")
        print(f"      Arithmetic: {avg_arithmetic_distance:.3f} (target: ≤{phase['target_arithmetic_distance']})")
        print(f"      Completion: {'✅' if (perplexity_met and arithmetic_met) else '⏳'}")
        
        return perplexity_met and arithmetic_met
    
    def save_checkpoint(self, step: int, val_loss: float, phase_num: int):
        """Save enhanced checkpoint with phase info"""
        checkpoint = {
            'step': step,
            'phase': phase_num,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config.__dict__,
            'metrics_history': self.metrics_history,
            'curriculum_phases': self.curriculum_phases,
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint-phase{phase_num+1}-step{step}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = os.path.join(self.output_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"⭐ New best model saved: {best_path}")
    
    def train(self):
        """Main enhanced training loop with 3-phase curriculum"""
        print("🚀 Starting Enhanced 3-Phase TurnGPT Training...")
        
        # Initial setup
        self.setup_model_and_data()
        
        total_steps = self.curriculum_phases[-1]['steps'][1]  # 12000
        
        # Phase-by-phase training
        for phase_num, phase in enumerate(self.curriculum_phases):
            self.setup_phase(phase_num)
            
            phase_start_step = phase['steps'][0]
            phase_end_step = phase['steps'][1]
            phase_steps = phase_end_step - phase_start_step
            
            # Initialize phase progress bar
            phase_progress = tqdm(
                total=phase_steps, 
                desc=f"Phase {phase_num + 1}: {phase['name']}", 
                initial=0
            )
            
            # Initialize data iterator
            train_iter = iter(self.train_loader)
            
            # Phase training loop
            running_loss = 0.0
            phase_step = 0
            
            while self.step < phase_end_step:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)
                
                # Training step
                loss = self.train_step(batch)
                running_loss += loss
                
                # Optimizer step
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                self.step += 1
                phase_step += 1
                
                # Update progress
                phase_progress.set_postfix({
                    'loss': f'{running_loss/phase_step:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                })
                phase_progress.update(1)
                
                # Evaluation (more frequent in later phases)
                eval_frequency = 500 if phase_num == 0 else 250  # More frequent later
                
                if self.step % eval_frequency == 0:
                    print(f"\n📊 Evaluation at step {self.step} (Phase {phase_num + 1})")
                    
                    # Run validation
                    val_loss, val_metrics = self.validation_step()
                    perplexity = val_metrics['perplexity']
                    
                    # Test semantic arithmetic
                    arithmetic_results = self.test_semantic_arithmetic()
                    avg_arithmetic = arithmetic_results.get('average_distance', float('inf'))
                    
                    # Test conversation quality
                    conversation_results = self.test_conversation_quality()
                    conv_quality = conversation_results['conversation_quality_score']
                    
                    # Generate sample
                    sample_text = self.generate_sample("Hello, I")
                    
                    print(f"   Validation loss: {val_loss:.4f}")
                    print(f"   Perplexity: {perplexity:.2f}")
                    print(f"   Arithmetic distance: {avg_arithmetic:.3f}")
                    print(f"   Conversation quality: {conv_quality:.3f}")
                    print(f"   Sample: '{sample_text}'")
                    
                    # Store metrics
                    self.metrics_history['train_loss'].append(running_loss / phase_step)
                    self.metrics_history['val_loss'].append(val_loss)
                    self.metrics_history['perplexity'].append(perplexity)
                    self.metrics_history['semantic_arithmetic'].append(avg_arithmetic)
                    self.metrics_history['conversation_quality'].append(conv_quality)
                    
                    # Update report
                    self.update_report({
                        'phase_num': phase_num + 1,
                        'phase_name': phase['name'],
                        'steps_range': f"{phase['steps'][0]}-{phase['steps'][1]}",
                        'status': 'In Progress',
                        'loss': val_loss,
                        'perplexity': perplexity,
                        'arithmetic_distance': avg_arithmetic,
                        'sample_generation': sample_text
                    })
                    
                    # Check phase completion
                    if self.evaluate_phase_completion(phase_num, val_loss, arithmetic_results):
                        print(f"🎉 Phase {phase_num + 1} completion criteria met!")
                    
                    self.model.train()  # Back to training mode
                
                # Save checkpoints
                if self.step % 1000 == 0:
                    val_loss, _ = self.validation_step()
                    self.save_checkpoint(self.step, val_loss, phase_num)
            
            phase_progress.close()
            
            # Phase completion
            print(f"\n🏁 Phase {phase_num + 1}: {phase['name']} Complete!")
            val_loss, val_metrics = self.validation_step()
            arithmetic_results = self.test_semantic_arithmetic()
            
            # Update report with phase completion
            self.update_report({
                'phase_num': phase_num + 1,
                'phase_name': phase['name'],
                'steps_range': f"{phase['steps'][0]}-{phase['steps'][1]}",
                'status': 'Completed',
                'loss': val_loss,
                'perplexity': val_metrics['perplexity'],
                'arithmetic_distance': arithmetic_results.get('average_distance', 0),
            })
            
            # Save phase completion checkpoint
            self.save_checkpoint(self.step, val_loss, phase_num)
        
        # Final evaluation
        print("\n🎯 Final Enhanced Evaluation:")
        final_val_loss, final_val_metrics = self.validation_step()
        final_arithmetic = self.test_semantic_arithmetic()
        final_conversation = self.test_conversation_quality()
        
        print(f"Final validation loss: {final_val_loss:.4f}")
        print(f"Final perplexity: {final_val_metrics['perplexity']:.2f}")
        print(f"Final arithmetic distance: {final_arithmetic['average_distance']:.3f}")
        print(f"Final conversation quality: {final_conversation['conversation_quality_score']:.3f}")
        
        # Generate final samples
        print("\n📝 Final Enhanced Generation Samples:")
        test_prompts = ["Hello", "What is", "I think", "Can you", "The future"]
        for prompt in test_prompts:
            sample = self.generate_sample(prompt, max_length=60)
            print(f"   '{prompt}' → '{sample}'")
        
        # Final report update
        with open(self.report_path, 'a') as f:
            f.write("## 🎉 Training Complete!\n\n")
            f.write(f"**Final Validation Loss**: {final_val_loss:.4f}\n")
            f.write(f"**Final Perplexity**: {final_val_metrics['perplexity']:.2f}\n")
            f.write(f"**Final Semantic Arithmetic**: {final_arithmetic['average_distance']:.3f}\n")
            f.write(f"**Final Conversation Quality**: {final_conversation['conversation_quality_score']:.3f}\n\n")
            f.write(f"**Training Completed**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print("🎉 Enhanced 3-Phase Training Complete!")
        
        return {
            'final_val_loss': final_val_loss,
            'final_perplexity': final_val_metrics['perplexity'],
            'semantic_arithmetic': final_arithmetic,
            'conversation_quality': final_conversation,
            'metrics_history': self.metrics_history,
        }

def main():
    """Enhanced training with conversation quality focus"""
    print("🌟 TurnGPT Enhanced Training Pipeline")
    print("=====================================")
    
    # Enhanced trainer configuration
    trainer = EnhancedTurnGPTTrainer(
        model_size="small",  # Optimized for M1
        output_dir="enhanced_checkpoints",
    )
    
    # Start enhanced training
    results = trainer.train()
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("🏆 ENHANCED TRAINING SUMMARY")
    print("="*80)
    print(f"Final validation loss: {results['final_val_loss']:.4f}")
    print(f"Final perplexity: {results['final_perplexity']:.2f}")
    print(f"Semantic arithmetic average distance: {results['semantic_arithmetic']['average_distance']:.3f}")
    print(f"Conversation quality score: {results['conversation_quality']['conversation_quality_score']:.3f}")
    
    print("\n🧮 Semantic Arithmetic Results:")
    for test, distance in results['semantic_arithmetic'].items():
        if test != 'average_distance':
            status = "✅" if distance < 2.0 else "⚠️" if distance < 5.0 else "❌"
            print(f"  {status} {test}: {distance:.3f}")
    
    print("\n🎯 Training Success Criteria:")
    criteria_met = 0
    total_criteria = 4
    
    if results['final_perplexity'] < 25:
        print("  ✅ Perplexity < 25")
        criteria_met += 1
    else:
        print(f"  ❌ Perplexity: {results['final_perplexity']:.2f} (target: < 25)")
    
    if results['semantic_arithmetic']['average_distance'] < 2.0:
        print("  ✅ Semantic arithmetic distance < 2.0")  
        criteria_met += 1
    else:
        print(f"  ❌ Arithmetic distance: {results['semantic_arithmetic']['average_distance']:.3f} (target: < 2.0)")
    
    if results['conversation_quality']['conversation_quality_score'] > 0.5:
        print("  ✅ Conversation quality > 0.5")
        criteria_met += 1
    else:
        print(f"  ❌ Conversation quality: {results['conversation_quality']['conversation_quality_score']:.3f} (target: > 0.5)")
    
    if results['final_val_loss'] < 3.0:
        print("  ✅ Final validation loss < 3.0")
        criteria_met += 1
    else:
        print(f"  ❌ Final loss: {results['final_val_loss']:.4f} (target: < 3.0)")
    
    success_rate = (criteria_met / total_criteria) * 100
    print(f"\n🎉 Overall Success Rate: {criteria_met}/{total_criteria} ({success_rate:.1f}%)")
    
    if success_rate >= 75:
        print("🌟 EXCELLENT: TurnGPT achieved conversation-quality performance!")
    elif success_rate >= 50:
        print("👍 GOOD: TurnGPT shows promising conversation capabilities!")
    else:
        print("🔧 NEEDS WORK: More training needed for conversation quality!")
    
    return results

if __name__ == "__main__":
    main()
