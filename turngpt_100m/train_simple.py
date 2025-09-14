"""
Simple Enhanced TurnGPT Training
Fixes gibberish generation with minimal code complexity
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
from model import TurnGPTLMHeadModel, create_turngpt_config
from turn_embedding_scaled import initialize_semantic_turns
from dataset import create_tokenizer, create_dataloaders, download_sample_texts, create_vocab_mapping

class SimpleTurnGPTTrainer:
    def __init__(self):
        # Device setup
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("🚀 Using Apple M1 GPU")
        else:
            self.device = torch.device("cpu")
            print("🚀 Using CPU")
        
        self.step = 0
        self.best_loss = float('inf')
    
    def setup(self):
        """Setup model, data, and enhanced initialization"""
        print("🔧 Setting up TurnGPT...")
        
        # Create tokenizer and model
        self.tokenizer = create_tokenizer()
        config = create_turngpt_config(self.tokenizer.vocab_size, "small")
        self.model = TurnGPTLMHeadModel(config).to(self.device)
        
        # Enhanced semantic initialization (19 → 500+ words)
        print("🧠 Enhanced semantic initialization...")
        vocab_mapping = create_vocab_mapping(self.tokenizer)
        turn_embedding = self.model.get_semantic_calculator()
        count = initialize_semantic_turns(turn_embedding, vocab_mapping)
        print(f"   ✅ Initialized {count} words with semantic patterns")
        
        # Load data with better parameters
        texts = download_sample_texts("data")
        self.train_loader, self.val_loader = create_dataloaders(
            texts=texts,
            tokenizer=self.tokenizer,
            batch_size=4,        # Better for conversation learning
            max_length=128,      # Longer context
            train_ratio=0.9,
        )
        
        # Optimizer with different rates for turns
        turn_params = [p for n, p in self.model.named_parameters() if "turns" in n]
        other_params = [p for n, p in self.model.named_parameters() if "turns" not in n]
        
        self.optimizer = optim.AdamW([
            {'params': other_params, 'lr': 2e-4},
            {'params': turn_params, 'lr': 1e-4},  # Lower LR for turns
        ])
        
        print("✅ Setup complete!")
    
    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()
    
    def evaluate(self):
        """Quick evaluation"""
        self.model.eval()
        total_loss = 0
        count = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs['loss'].item()
                count += 1
                
                if count >= 10:  # Quick eval
                    break
        
        return total_loss / count if count > 0 else float('inf')
    
    def test_arithmetic(self):
        """Test semantic arithmetic"""
        turn_embedding = self.model.get_semantic_calculator()
        
        test_cases = [
            ("king", "man", "woman", "queen"),
            ("happy", "positive", "negative", "sad"),
            ("big", "small", "tiny", "huge"),
        ]
        
        total_distance = 0
        successful = 0
        
        for word_a, word_b, word_c, expected in test_cases:
            try:
                a_id = self.tokenizer.encode(word_a, add_special_tokens=False)[0]
                b_id = self.tokenizer.encode(word_b, add_special_tokens=False)[0]
                c_id = self.tokenizer.encode(word_c, add_special_tokens=False)[0]
                expected_id = self.tokenizer.encode(expected, add_special_tokens=False)[0]
                
                result_turns, _ = turn_embedding.semantic_arithmetic(a_id, b_id, c_id)
                expected_turns = turn_embedding.get_turn_vector(expected_id)
                distance = torch.norm(result_turns - expected_turns).item()
                
                total_distance += distance
                successful += 1
                
            except:
                pass
        
        return total_distance / successful if successful > 0 else float('inf')
    
    def generate_sample(self, prompt="Hello"):
        """Generate text sample"""
        self.model.eval()
        input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_length=len(input_ids[0]) + 30,
                temperature=0.8,
                do_sample=True,
                top_k=50,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    def train(self, steps=5000):
        """Main training loop - simple and effective"""
        print(f"🚀 Training TurnGPT for {steps} steps...")
        
        self.setup()
        
        train_iter = iter(self.train_loader)
        progress = tqdm(range(steps), desc="Training")
        
        for step in progress:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            
            loss = self.train_step(batch)
            self.step += 1
            
            # Update progress
            progress.set_postfix({'loss': f'{loss:.4f}'})
            
            # Evaluate every 500 steps
            if (step + 1) % 500 == 0:
                val_loss = self.evaluate()
                arithmetic_dist = self.test_arithmetic()
                sample = self.generate_sample("Hello")
                
                print(f"\n📊 Step {step + 1}:")
                print(f"   Val loss: {val_loss:.4f}")
                print(f"   Arithmetic: {arithmetic_dist:.3f}")
                print(f"   Sample: '{sample}'")
                
                # Save if best
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    torch.save(self.model.state_dict(), "best_model.pt")
                    print("   💾 Saved best model")
        
        # Final test
        print("\n🎯 Final Results:")
        final_loss = self.evaluate()
        final_arithmetic = self.test_arithmetic()
        
        print(f"Final loss: {final_loss:.4f}")
        print(f"Final arithmetic: {final_arithmetic:.3f}")
        
        # Generate samples
        print("\n📝 Final samples:")
        for prompt in ["Hello", "What is", "I think"]:
            sample = self.generate_sample(prompt)
            print(f"   '{prompt}' → '{sample}'")
        
        print("✅ Training complete!")
        return {'loss': final_loss, 'arithmetic': final_arithmetic}

def main():
    """Simple training - just run it!"""
    trainer = SimpleTurnGPTTrainer()
    results = trainer.train(steps=5000)  # 5K steps should show improvement
    
    # Simple success check
    if results['loss'] < 4.0 and results['arithmetic'] < 3.0:
        print("🎉 SUCCESS: Model shows conversation potential!")
    else:
        print("🔧 Try running longer or check data quality")

if __name__ == "__main__":
    main()
