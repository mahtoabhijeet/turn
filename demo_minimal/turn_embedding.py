"""
Semantic Turn Theory - Core Implementation
The minimal 20% that shows 80% of the breakthrough
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional

class TurnEmbedding(nn.Module):
    """
    The core breakthrough: 4 integers → 128D semantic embeddings via polynomials
    """
    def __init__(self, vocab_size: int, n_turns: int = 4, output_dim: int = 128, poly_degree: int = 3):
        super().__init__()
        # Each word = n_turns integers (the "hydrogen atoms" of meaning)
        self.turns = nn.Parameter(torch.randint(-5, 6, (vocab_size, n_turns)).float())
        # Polynomial coefficients: the "semantic wormholes"
        self.poly_coeffs = nn.Parameter(torch.randn(n_turns, poly_degree + 1, output_dim) * 0.1)
        
        self.n_turns = n_turns
        self.poly_degree = poly_degree
        self.output_dim = output_dim

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Generate embeddings from turn integers via polynomial transformation"""
        base_turns = self.turns[token_ids]  # [B, S, n_turns]
        batch_size, seq_len = base_turns.shape[:2]
        embeddings = torch.zeros(batch_size, seq_len, self.output_dim, device=base_turns.device)
        
        for turn_idx in range(self.n_turns):
            x = base_turns[..., turn_idx].unsqueeze(-1)  # [B, S, 1]
            # Generate polynomial: 1, x, x², x³...
            powers = torch.cat([x**d for d in range(self.poly_degree + 1)], dim=-1)
            # Apply coefficients → [B, S, output_dim]
            embeddings += torch.einsum('bsp,pdo->bso', powers, self.poly_coeffs[turn_idx])
        
        return embeddings
    
    def get_turn_vector(self, token_id: int) -> torch.Tensor:
        """Get the turn vector for a specific token - the core semantic representation"""
        return self.turns[token_id].detach()
    
    def semantic_arithmetic(self, word_a: str, word_b: str, word_c: str, vocab: Dict[str, int]) -> Tuple[torch.Tensor, str]:
        """Perform semantic arithmetic: word_a - word_b + word_c = ?"""
        turn_a = self.turns[vocab[word_a]]
        turn_b = self.turns[vocab[word_b]] 
        turn_c = self.turns[vocab[word_c]]
        
        result_turns = turn_a - turn_b + turn_c
        
        # Find closest word by turn distance
        distances = torch.norm(self.turns - result_turns.unsqueeze(0), dim=1)
        closest_id = torch.argmin(distances).item()
        
        # Convert back to word
        reverse_vocab = {v: k for k, v in vocab.items()}
        closest_word = reverse_vocab[closest_id]
        
        return result_turns.detach(), closest_word

class SemanticCalculator:
    """
    The demo interface - semantic arithmetic that actually works
    """
    def __init__(self, model: TurnEmbedding, vocab: Dict[str, int]):
        self.model = model
        self.vocab = vocab
        self.reverse_vocab = {v: k for k, v in vocab.items()}
    
    def calculate(self, equation: str) -> Dict:
        """
        Parse and solve semantic equations like "king - man + woman"
        Returns the result with interpretable turn breakdown
        """
        # Simple parser - could be enhanced but this shows the concept
        parts = equation.replace(' ', '').split('+')
        if len(parts) != 2:
            raise ValueError("Expected format: 'word1 - word2 + word3'")
        
        left_part = parts[0].split('-')
        if len(left_part) != 2:
            raise ValueError("Expected format: 'word1 - word2 + word3'")
        
        word_a, word_b, word_c = left_part[0], left_part[1], parts[1]
        
        # Get the turn vectors
        turn_a = self.model.turns[self.vocab[word_a]].detach()
        turn_b = self.model.turns[self.vocab[word_b]].detach()
        turn_c = self.model.turns[self.vocab[word_c]].detach()
        
        # Perform the arithmetic
        result_turns = turn_a - turn_b + turn_c
        
        # Find closest word
        distances = torch.norm(self.model.turns - result_turns.unsqueeze(0), dim=1)
        closest_id = torch.argmin(distances).item()
        closest_word = self.reverse_vocab[closest_id]
        distance = distances[closest_id].item()
        
        return {
            'equation': equation,
            'result_word': closest_word,
            'distance': distance,
            'turn_breakdown': {
                word_a: turn_a.numpy().round(3),
                word_b: turn_b.numpy().round(3), 
                word_c: turn_c.numpy().round(3),
                'result': result_turns.numpy().round(3)
            },
            'arithmetic': f"{turn_a.numpy().round(2)} - {turn_b.numpy().round(2)} + {turn_c.numpy().round(2)} = {result_turns.numpy().round(2)}"
        }

def create_semantic_vocab() -> Dict[str, int]:
    """Create a comprehensive 100-word vocabulary for strong proof of concept"""
    words = [
        # Royalty & Authority (8 words)
        "king", "queen", "prince", "princess", "emperor", "empress", "ruler", "leader",
        
        # People & Relationships (12 words)
        "man", "woman", "boy", "girl", "child", "adult", "friend", "enemy", "family", "parent", "teacher", "student",
        
        # Animals (12 words)
        "cat", "dog", "kitten", "puppy", "lion", "tiger", "bird", "fish", "horse", "cow", "pig", "sheep",
        
        # Size & Scale (8 words)
        "small", "big", "tiny", "huge", "large", "mini", "giant", "massive",
        
        # Temperature & Weather (8 words)
        "hot", "cold", "warm", "cool", "freezing", "boiling", "sunny", "rainy",
        
        # Colors (8 words)
        "red", "blue", "green", "yellow", "black", "white", "purple", "orange",
        
        # Actions & Movement (12 words)
        "run", "ran", "walk", "walked", "jump", "flew", "swim", "drove", "climb", "fall", "sit", "stand",
        
        # Emotions & Feelings (12 words)
        "happy", "sad", "joy", "anger", "love", "hate", "fear", "calm", "excited", "worried", "proud", "ashamed",
        
        # Qualities & States (12 words)
        "good", "bad", "better", "worse", "strong", "weak", "fast", "slow", "smart", "dumb", "beautiful", "ugly",
        
        # Objects & Things (8 words)
        "house", "car", "book", "food", "water", "tree", "mountain", "ocean"
    ]
    return {word: i for i, word in enumerate(words)}

def initialize_semantic_turns(model: TurnEmbedding, vocab: Dict[str, int]):
    """Initialize turns with semantic structure - the secret sauce"""
    # Format: [Conceptual, Behavioral, Size, Context]
    semantic_init = {
        # Royalty & Authority (8 words)
        "king": [5.0, 0.0, 2.0, 0.0],      # Royal, Neutral, Large, Present
        "queen": [5.0, 0.0, 2.0, 0.0],     # Royal, Neutral, Large, Present
        "prince": [5.0, 0.0, 1.0, 0.0],    # Royal, Neutral, Medium, Present
        "princess": [5.0, 0.0, 1.0, 0.0],  # Royal, Neutral, Medium, Present
        "emperor": [6.0, 0.0, 3.0, 0.0],   # Imperial, Neutral, Very Large, Present
        "empress": [6.0, 0.0, 3.0, 0.0],   # Imperial, Neutral, Very Large, Present
        "ruler": [4.0, 0.0, 2.0, 0.0],     # Authority, Neutral, Large, Present
        "leader": [4.0, 2.0, 1.0, 0.0],    # Authority, Social, Medium, Present
        
        # People & Relationships (12 words)
        "man": [2.0, 0.0, 0.0, 0.0],       # Human, Neutral, Medium, Present
        "woman": [2.0, 0.0, 0.0, 0.0],     # Human, Neutral, Medium, Present
        "boy": [2.0, 0.0, -1.0, 0.0],      # Human, Neutral, Small, Present
        "girl": [2.0, 0.0, -1.0, 0.0],     # Human, Neutral, Small, Present
        "child": [2.0, 0.0, -2.0, 0.0],    # Human, Neutral, Very Small, Present
        "adult": [2.0, 0.0, 1.0, 0.0],     # Human, Neutral, Large, Present
        "friend": [2.0, 3.0, 0.0, 0.0],    # Human, Very Social, Medium, Present
        "enemy": [2.0, -3.0, 0.0, 0.0],    # Human, Anti-Social, Medium, Present
        "family": [2.0, 2.0, 0.0, 0.0],    # Human, Social, Medium, Present
        "parent": [2.0, 1.0, 1.0, 0.0],    # Human, Social, Large, Present
        "teacher": [2.0, 2.0, 0.0, 0.0],   # Human, Social, Medium, Present
        "student": [2.0, 0.0, -1.0, 0.0],  # Human, Neutral, Small, Present
        
        # Animals (12 words)
        "cat": [3.0, -2.0, 0.0, 0.0],      # Animal, Independent, Medium, Present
        "dog": [3.0, 2.0, 0.0, 0.0],       # Animal, Social, Medium, Present
        "kitten": [3.0, -2.0, -2.0, 0.0],  # Animal, Independent, Small, Present
        "puppy": [3.0, 2.0, -2.0, 0.0],    # Animal, Social, Small, Present
        "lion": [3.0, -2.0, 3.0, 0.0],     # Animal, Independent, Large, Present
        "tiger": [3.0, -2.0, 2.0, 0.0],    # Animal, Independent, Large, Present
        "bird": [3.0, 0.0, -1.0, 1.0],     # Animal, Neutral, Small, Air
        "fish": [3.0, 0.0, 0.0, -1.0],     # Animal, Neutral, Medium, Water
        "horse": [3.0, 1.0, 2.0, 0.0],     # Animal, Social, Large, Present
        "cow": [3.0, 0.0, 2.0, 0.0],       # Animal, Neutral, Large, Present
        "pig": [3.0, 0.0, 1.0, 0.0],       # Animal, Neutral, Medium, Present
        "sheep": [3.0, 1.0, 1.0, 0.0],     # Animal, Social, Medium, Present
        
        # Size & Scale (8 words)
        "small": [0.0, 0.0, -2.0, 0.0],    # Modifier, Neutral, Small, Present
        "big": [0.0, 0.0, 2.0, 0.0],       # Modifier, Neutral, Large, Present
        "tiny": [0.0, 0.0, -3.0, 0.0],     # Modifier, Neutral, Very Small, Present
        "huge": [0.0, 0.0, 3.0, 0.0],      # Modifier, Neutral, Very Large, Present
        "large": [0.0, 0.0, 2.0, 0.0],     # Modifier, Neutral, Large, Present
        "mini": [0.0, 0.0, -3.0, 0.0],     # Modifier, Neutral, Very Small, Present
        "giant": [0.0, 0.0, 4.0, 0.0],     # Modifier, Neutral, Huge, Present
        "massive": [0.0, 0.0, 4.0, 0.0],   # Modifier, Neutral, Huge, Present
        
        # Temperature & Weather (8 words)
        "hot": [0.0, 0.0, 0.0, 3.0],       # Modifier, Neutral, Medium, Hot
        "cold": [0.0, 0.0, 0.0, -3.0],     # Modifier, Neutral, Medium, Cold
        "warm": [0.0, 0.0, 0.0, 1.0],      # Modifier, Neutral, Medium, Warm
        "cool": [0.0, 0.0, 0.0, -1.0],     # Modifier, Neutral, Medium, Cool
        "freezing": [0.0, 0.0, 0.0, -4.0], # Modifier, Neutral, Medium, Very Cold
        "boiling": [0.0, 0.0, 0.0, 4.0],   # Modifier, Neutral, Medium, Very Hot
        "sunny": [0.0, 0.0, 0.0, 2.0],     # Modifier, Neutral, Medium, Bright
        "rainy": [0.0, 0.0, 0.0, -2.0],    # Modifier, Neutral, Medium, Wet
        
        # Colors (8 words)
        "red": [0.0, 0.0, 0.0, 0.0],       # Color, Neutral, Medium, Present
        "blue": [0.0, 0.0, 0.0, 0.0],      # Color, Neutral, Medium, Present
        "green": [0.0, 0.0, 0.0, 0.0],     # Color, Neutral, Medium, Present
        "yellow": [0.0, 0.0, 0.0, 0.0],    # Color, Neutral, Medium, Present
        "black": [0.0, 0.0, 0.0, 0.0],     # Color, Neutral, Medium, Present
        "white": [0.0, 0.0, 0.0, 0.0],     # Color, Neutral, Medium, Present
        "purple": [0.0, 0.0, 0.0, 0.0],    # Color, Neutral, Medium, Present
        "orange": [0.0, 0.0, 0.0, 0.0],    # Color, Neutral, Medium, Present
        
        # Actions & Movement (12 words)
        "run": [1.0, 0.0, 0.0, 1.0],       # Action, Neutral, Medium, Present
        "ran": [1.0, 0.0, 0.0, -1.0],      # Action, Neutral, Medium, Past
        "walk": [1.0, 0.0, 0.0, 1.0],      # Action, Neutral, Medium, Present
        "walked": [1.0, 0.0, 0.0, -1.0],   # Action, Neutral, Medium, Past
        "jump": [1.0, 0.0, 0.0, 1.0],      # Action, Neutral, Medium, Present
        "flew": [1.0, 0.0, 0.0, -1.0],     # Action, Neutral, Medium, Past
        "swim": [1.0, 0.0, 0.0, 1.0],      # Action, Neutral, Medium, Present
        "drove": [1.0, 0.0, 0.0, -1.0],    # Action, Neutral, Medium, Past
        "climb": [1.0, 0.0, 0.0, 1.0],     # Action, Neutral, Medium, Present
        "fall": [1.0, 0.0, 0.0, 1.0],      # Action, Neutral, Medium, Present
        "sit": [1.0, 0.0, 0.0, 1.0],       # Action, Neutral, Medium, Present
        "stand": [1.0, 0.0, 0.0, 1.0],     # Action, Neutral, Medium, Present
        
        # Emotions & Feelings (12 words)
        "happy": [0.0, 3.0, 0.0, 0.0],     # Emotion, Positive, Medium, Present
        "sad": [0.0, -3.0, 0.0, 0.0],      # Emotion, Negative, Medium, Present
        "joy": [0.0, 4.0, 0.0, 0.0],       # Emotion, Very Positive, Medium, Present
        "anger": [0.0, -4.0, 0.0, 0.0],    # Emotion, Very Negative, Medium, Present
        "love": [0.0, 4.0, 0.0, 0.0],      # Emotion, Very Positive, Medium, Present
        "hate": [0.0, -4.0, 0.0, 0.0],     # Emotion, Very Negative, Medium, Present
        "fear": [0.0, -2.0, 0.0, 0.0],     # Emotion, Negative, Medium, Present
        "calm": [0.0, 0.0, 0.0, 0.0],      # Emotion, Neutral, Medium, Present
        "excited": [0.0, 3.0, 0.0, 0.0],   # Emotion, Positive, Medium, Present
        "worried": [0.0, -2.0, 0.0, 0.0],  # Emotion, Negative, Medium, Present
        "proud": [0.0, 2.0, 0.0, 0.0],     # Emotion, Positive, Medium, Present
        "ashamed": [0.0, -3.0, 0.0, 0.0],  # Emotion, Negative, Medium, Present
        
        # Qualities & States (12 words)
        "good": [0.0, 2.0, 0.0, 0.0],      # Quality, Positive, Medium, Present
        "bad": [0.0, -2.0, 0.0, 0.0],      # Quality, Negative, Medium, Present
        "better": [0.0, 2.0, 0.0, 0.0],    # Quality, Positive, Medium, Present
        "worse": [0.0, -2.0, 0.0, 0.0],    # Quality, Negative, Medium, Present
        "strong": [0.0, 0.0, 2.0, 0.0],    # Quality, Neutral, Large, Present
        "weak": [0.0, 0.0, -2.0, 0.0],     # Quality, Neutral, Small, Present
        "fast": [0.0, 0.0, 0.0, 2.0],      # Quality, Neutral, Medium, Fast
        "slow": [0.0, 0.0, 0.0, -2.0],     # Quality, Neutral, Medium, Slow
        "smart": [0.0, 1.0, 0.0, 0.0],     # Quality, Positive, Medium, Present
        "dumb": [0.0, -1.0, 0.0, 0.0],     # Quality, Negative, Medium, Present
        "beautiful": [0.0, 2.0, 0.0, 0.0], # Quality, Positive, Medium, Present
        "ugly": [0.0, -2.0, 0.0, 0.0],     # Quality, Negative, Medium, Present
        
        # Objects & Things (8 words)
        "house": [4.0, 0.0, 2.0, 0.0],     # Object, Neutral, Large, Present
        "car": [4.0, 0.0, 1.0, 0.0],       # Object, Neutral, Medium, Present
        "book": [4.0, 0.0, 0.0, 0.0],      # Object, Neutral, Medium, Present
        "food": [4.0, 0.0, 0.0, 0.0],      # Object, Neutral, Medium, Present
        "water": [4.0, 0.0, 0.0, -1.0],    # Object, Neutral, Medium, Liquid
        "tree": [4.0, 0.0, 2.0, 0.0],      # Object, Neutral, Large, Present
        "mountain": [4.0, 0.0, 4.0, 0.0],  # Object, Neutral, Huge, Present
        "ocean": [4.0, 0.0, 4.0, -1.0],    # Object, Neutral, Huge, Liquid
    }
    
    # Initialize with semantic structure
    for word, turns in semantic_init.items():
        if word in vocab:
            model.turns.data[vocab[word]] = torch.tensor(turns, dtype=torch.float32)
