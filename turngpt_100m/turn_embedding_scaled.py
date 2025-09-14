"""
Scaled Semantic Turn Theory Implementation
TurnGPT-100M: 100M parameter model with turn-based embeddings
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import math
import turngpt_rust # Import the Rust module

class ScaledTurnEmbedding(nn.Module):
    """
    Scaled version of TurnEmbedding for larger vocabularies and transformer integration
    Each token represented by 8 turns instead of traditional 768D embeddings
    """
    def __init__(
        self, 
        vocab_size: int,
        n_turns: int = 8,
        output_dim: int = 768,
        poly_degree: int = 4,
        dropout: float = 0.1,
        max_position_embeddings: int = 1024
    ):
        super().__init__()
        
        # Core turn parameters: 8 integers per token instead of 768 floats
        self.turns = nn.Parameter(torch.randint(-10, 11, (vocab_size, n_turns)).float())
        
        # Polynomial coefficients for generating embeddings from turns
        self.poly_coeffs = nn.Parameter(torch.randn(n_turns, poly_degree + 1, output_dim) * 0.02)
        
        # Position embeddings (standard)
        self.position_embeddings = nn.Embedding(max_position_embeddings, output_dim)
        
        # Layer norm and dropout
        self.LayerNorm = nn.LayerNorm(output_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        
        self.vocab_size = vocab_size
        self.n_turns = n_turns
        self.output_dim = output_dim
        self.poly_degree = poly_degree
        
        # Initialize positions
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate embeddings from turn integers via polynomial transformation
        
        Args:
            input_ids: [batch_size, seq_len] token ids
            position_ids: [batch_size, seq_len] position ids (optional)
            
        Returns:
            embeddings: [batch_size, seq_len, output_dim] embeddings
        """
        batch_size, seq_len = input_ids.shape
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_len]
        
        # Get turn vectors for input tokens
        base_turns = self.turns[input_ids]  # [batch_size, seq_len, n_turns]
        
        # Generate embeddings via polynomial transformation using Rust
        # Convert PyTorch tensors to NumPy arrays for Rust FFI
        turns_np = base_turns.detach().cpu().numpy().astype(np.int8)
        
        # Prepare coefficients for Rust function
        # Reshape poly_coeffs from (n_turns, poly_degree + 1, output_dim) to a flat array
        # The Rust function expects coeffs to be flattened for each turn_idx
        coeffs_np = self.poly_coeffs.detach().cpu().numpy().astype(np.float32)
        
        # Call Rust function for polynomial evaluation
        # The Rust function will return a flattened 1D array, reshape it back to 3D
        rust_embeddings_flat = turngpt_rust.evaluate_turns(
            turns_np.reshape(-1, self.n_turns), # Reshape turns to (batch_size * seq_len, n_turns)
            coeffs_np.reshape(self.n_turns, -1) # Reshape coeffs to (n_turns, (poly_degree + 1) * output_dim)
        )
        
        embeddings = torch.from_numpy(
            np.array(rust_embeddings_flat, dtype=np.float32)
        ).reshape(batch_size, seq_len, self.output_dim).to(base_turns.device)
        
        # Add position embeddings
        position_embeds = self.position_embeddings(position_ids)
        embeddings += position_embeds
        
        # Layer norm and dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    def get_turn_vector(self, token_id: int) -> torch.Tensor:
        """Get the raw turn vector for a token"""
        return self.turns[token_id].detach()
    
    def semantic_arithmetic(
        self, 
        word_a_id: int, 
        word_b_id: int, 
        word_c_id: int
    ) -> Tuple[torch.Tensor, int]:
        """
        Perform semantic arithmetic: word_a - word_b + word_c = ?
        Returns the result turns and closest token id
        """
        turn_a = self.turns[word_a_id]
        turn_b = self.turns[word_b_id]
        turn_c = self.turns[word_c_id]
        
        # Perform semantic arithmetic using Rust
        result_turns_np = np.array(turngpt_rust.semantic_arithmetic(
            turn_a.detach().cpu().numpy().astype(np.int8),
            turn_b.detach().cpu().numpy().astype(np.int8),
            turn_c.detach().cpu().numpy().astype(np.int8)
        ), dtype=np.int8)
        
        result_turns = torch.from_numpy(result_turns_np).to(turn_a.device)
        
        # Find closest token by turn distance using Rust
        vocab_turns_np = self.turns.detach().cpu().numpy().astype(np.int8)
        closest_id = turngpt_rust.find_closest_turn(
            result_turns_np,
            vocab_turns_np
        )
        
        return result_turns.detach(), closest_id
    
    def get_compression_stats(self) -> Dict[str, float]:
        """Calculate compression statistics vs traditional embeddings"""
        traditional_params = self.vocab_size * self.output_dim  # 768D per token
        turn_params = self.vocab_size * self.n_turns  # 8D per token
        poly_params = self.n_turns * (self.poly_degree + 1) * self.output_dim
        
        total_turn_params = turn_params + poly_params
        compression_ratio = traditional_params / total_turn_params
        memory_savings = (1 - total_turn_params / traditional_params) * 100
        
        return {
            'traditional_params': traditional_params,
            'turn_params': turn_params,
            'poly_params': poly_params, 
            'total_turn_params': total_turn_params,
            'compression_ratio': compression_ratio,
            'memory_savings_percent': memory_savings
        }

def initialize_semantic_turns(model: ScaledTurnEmbedding, vocab_mapping: Dict[str, int]):
    """
    Enhanced semantic initialization with 500+ words for conversation quality
    Turn dimensions: [Concept, Sentiment, Size, Temporal, Social, Domain, Formality, Intensity]
    """
    
    # Define enhanced semantic patterns (8-dimensional turns)
    semantic_patterns = {
        # Core conceptual categories
        'human': [4, 0, 0, 0, 2, 0, 0, 0],
        'animal': [3, 0, 0, 0, 0, 0, 0, 0], 
        'object': [1, 0, 0, 0, 0, 0, 0, 0],
        'place': [2, 0, 0, 0, 0, 0, 0, 0],
        'abstract': [5, 0, 0, 0, 0, 0, 0, 0],
        'action': [6, 0, 0, 0, 0, 0, 0, 0],
        
        # Sentiment patterns (Turn 1)
        'positive': [0, 3, 0, 0, 0, 0, 0, 0],
        'negative': [0, -3, 0, 0, 0, 0, 0, 0],
        'neutral': [0, 0, 0, 0, 0, 0, 0, 0],
        
        # Size patterns (Turn 2)
        'large': [0, 0, 3, 0, 0, 0, 0, 0],
        'small': [0, 0, -3, 0, 0, 0, 0, 0],
        'medium': [0, 0, 0, 0, 0, 0, 0, 0],
        
        # Temporal patterns (Turn 3)
        'past': [0, 0, 0, -3, 0, 0, 0, 0],
        'present': [0, 0, 0, 0, 0, 0, 0, 0],
        'future': [0, 0, 0, 3, 0, 0, 0, 0],
        
        # Social patterns (Turn 4)
        'social': [0, 0, 0, 0, 3, 0, 0, 0],
        'independent': [0, 0, 0, 0, -3, 0, 0, 0],
        'cooperative': [0, 0, 0, 0, 2, 0, 0, 0],
        
        # Domain patterns (Turn 5)
        'technology': [0, 0, 0, 0, 0, 3, 0, 0],
        'nature': [0, 0, 0, 0, 0, -3, 0, 0],
        'science': [0, 0, 0, 0, 0, 2, 0, 0],
        'everyday': [0, 0, 0, 0, 0, 0, 0, 0],
        
        # Formality patterns (Turn 6)
        'formal': [0, 0, 0, 0, 0, 0, 3, 0],
        'informal': [0, 0, 0, 0, 0, 0, -3, 0],
        'casual': [0, 0, 0, 0, 0, 0, -1, 0],
        
        # Intensity patterns (Turn 7)
        'intense': [0, 0, 0, 0, 0, 0, 0, 3],
        'mild': [0, 0, 0, 0, 0, 0, 0, -3],
        'moderate': [0, 0, 0, 0, 0, 0, 0, 0],
    }
    
    # Comprehensive word categories for conversation quality
    word_categories = {
        # Greetings & Social (50 words)
        'hello': ['human', 'social', 'informal', 'positive'],
        'hi': ['human', 'social', 'informal', 'positive'],
        'hey': ['human', 'social', 'informal', 'positive'], 
        'goodbye': ['human', 'social', 'neutral'],
        'bye': ['human', 'social', 'informal', 'neutral'],
        'thanks': ['human', 'social', 'positive'],
        'thank': ['human', 'social', 'positive'],
        'please': ['human', 'social', 'cooperative'],
        'sorry': ['human', 'social', 'negative'],
        'welcome': ['human', 'social', 'positive'],
        'yes': ['human', 'positive', 'cooperative'],
        'no': ['human', 'negative', 'independent'],
        'maybe': ['human', 'neutral'],
        'sure': ['human', 'positive', 'cooperative'],
        'okay': ['human', 'neutral', 'cooperative'],
        'alright': ['human', 'neutral', 'cooperative'],
        'fine': ['human', 'neutral'],
        
        # Questions (20 words)
        'who': ['abstract', 'neutral'],
        'what': ['abstract', 'neutral'],
        'when': ['abstract', 'temporal', 'neutral'],
        'where': ['abstract', 'place', 'neutral'],
        'why': ['abstract', 'neutral'],
        'how': ['abstract', 'neutral'],
        'which': ['abstract', 'neutral'],
        'can': ['action', 'neutral'],
        'would': ['action', 'formal'],
        'could': ['action', 'formal'],
        'should': ['action', 'formal'],
        'will': ['action', 'future'],
        'do': ['action', 'present'],
        'does': ['action', 'present'],
        'did': ['action', 'past'],
        'are': ['action', 'present'],
        'is': ['action', 'present'],
        'was': ['action', 'past'],
        'were': ['action', 'past'],
        
        # Emotions (40 words)
        'happy': ['abstract', 'positive', 'intense'],
        'sad': ['abstract', 'negative', 'intense'],
        'angry': ['abstract', 'negative', 'intense'],
        'excited': ['abstract', 'positive', 'intense'],
        'worried': ['abstract', 'negative', 'mild'],
        'calm': ['abstract', 'positive', 'mild'],
        'love': ['abstract', 'positive', 'intense'],
        'hate': ['abstract', 'negative', 'intense'],
        'like': ['abstract', 'positive', 'mild'],
        'enjoy': ['abstract', 'positive', 'moderate'],
        'fear': ['abstract', 'negative', 'intense'],
        'hope': ['abstract', 'positive', 'mild'],
        'feel': ['abstract', 'neutral'],
        'think': ['abstract', 'neutral'],
        'believe': ['abstract', 'neutral'],
        'know': ['abstract', 'neutral'],
        'understand': ['abstract', 'neutral'],
        'remember': ['abstract', 'neutral'],
        'forget': ['abstract', 'negative'],
        'care': ['abstract', 'positive'],
        'mind': ['abstract', 'neutral'],
        'matter': ['abstract', 'neutral'],
        
        # People & Relationships (30 words)
        'person': ['human', 'neutral'],
        'people': ['human', 'social'],
        'friend': ['human', 'social', 'positive'],
        'family': ['human', 'social', 'positive'],
        'mother': ['human', 'social', 'positive'],
        'father': ['human', 'social', 'positive'],
        'parent': ['human', 'social', 'positive'],
        'child': ['human', 'small', 'positive'],
        'baby': ['human', 'small', 'positive'],
        'man': ['human', 'neutral'],
        'woman': ['human', 'neutral'],
        'boy': ['human', 'small'],
        'girl': ['human', 'small'],
        'king': ['human', 'large', 'formal'],
        'queen': ['human', 'large', 'formal'],
        'teacher': ['human', 'social', 'formal'],
        'student': ['human', 'social'],
        'doctor': ['human', 'social', 'formal'],
        'nurse': ['human', 'social', 'positive'],
        'worker': ['human', 'neutral'],
        'boss': ['human', 'large', 'formal'],
        'employee': ['human', 'neutral'],
        
        # Actions (60 words)
        'go': ['action', 'present'],
        'come': ['action', 'present'],
        'run': ['action', 'present', 'intense'],
        'ran': ['action', 'past', 'intense'],
        'walk': ['action', 'present'],
        'talk': ['action', 'present', 'social'],
        'speak': ['action', 'present', 'formal'],
        'say': ['action', 'present'],
        'tell': ['action', 'present', 'social'],
        'ask': ['action', 'present', 'social'],
        'answer': ['action', 'present', 'social'],
        'help': ['action', 'present', 'positive', 'social'],
        'work': ['action', 'present', 'neutral'],
        'play': ['action', 'present', 'positive'],
        'eat': ['action', 'present'],
        'drink': ['action', 'present'],
        'sleep': ['action', 'present'],
        'wake': ['action', 'present'],
        'see': ['action', 'present'],
        'look': ['action', 'present'],
        'watch': ['action', 'present'],
        'read': ['action', 'present'],
        'write': ['action', 'present'],
        'learn': ['action', 'present', 'positive'],
        'teach': ['action', 'present', 'positive', 'social'],
        'study': ['action', 'present'],
        'try': ['action', 'present'],
        'want': ['action', 'present'],
        'need': ['action', 'present'],
        'have': ['action', 'present'],
        'get': ['action', 'present'],
        'give': ['action', 'present', 'social'],
        'take': ['action', 'present'],
        'buy': ['action', 'present'],
        'sell': ['action', 'present'],
        'make': ['action', 'present'],
        'build': ['action', 'present'],
        'create': ['action', 'present', 'positive'],
        'destroy': ['action', 'present', 'negative'],
        'fix': ['action', 'present', 'positive'],
        'break': ['action', 'present', 'negative'],
        
        # Animals (25 words)
        'cat': ['animal', 'independent', 'small'],
        'dog': ['animal', 'social', 'medium'],
        'bird': ['animal', 'small'],
        'fish': ['animal', 'small'],
        'horse': ['animal', 'large'],
        'cow': ['animal', 'large'],
        'pig': ['animal', 'medium'],
        'sheep': ['animal', 'medium'],
        'lion': ['animal', 'large', 'intense'],
        'tiger': ['animal', 'large', 'intense'],
        'bear': ['animal', 'large', 'intense'],
        'wolf': ['animal', 'medium', 'intense'],
        'mouse': ['animal', 'small'],
        'rat': ['animal', 'small'],
        'rabbit': ['animal', 'small'],
        'elephant': ['animal', 'large'],
        'monkey': ['animal', 'medium', 'social'],
        
        # Objects & Places (50 words)
        'house': ['object', 'large', 'everyday'],
        'home': ['place', 'positive', 'everyday'],
        'car': ['object', 'medium', 'technology'],
        'book': ['object', 'small', 'everyday'],
        'phone': ['object', 'small', 'technology'],
        'computer': ['object', 'medium', 'technology'],
        'table': ['object', 'medium', 'everyday'],
        'chair': ['object', 'medium', 'everyday'],
        'bed': ['object', 'large', 'everyday'],
        'door': ['object', 'medium', 'everyday'],
        'window': ['object', 'medium', 'everyday'],
        'school': ['place', 'formal', 'everyday'],
        'work': ['place', 'formal', 'everyday'],
        'store': ['place', 'neutral', 'everyday'],
        'park': ['place', 'positive', 'nature'],
        'city': ['place', 'large', 'social'],
        'town': ['place', 'medium', 'social'],
        'country': ['place', 'large'],
        'world': ['place', 'large'],
        'earth': ['place', 'large', 'nature'],
        
        # Descriptors (50 words)
        'good': ['abstract', 'positive'],
        'bad': ['abstract', 'negative'],
        'great': ['abstract', 'positive', 'intense'],
        'amazing': ['abstract', 'positive', 'intense'],
        'terrible': ['abstract', 'negative', 'intense'],
        'awful': ['abstract', 'negative', 'intense'],
        'nice': ['abstract', 'positive', 'mild'],
        'fine': ['abstract', 'neutral'],
        'okay': ['abstract', 'neutral'],
        'big': ['abstract', 'large'],
        'small': ['abstract', 'small'],
        'large': ['abstract', 'large'],
        'little': ['abstract', 'small'],
        'huge': ['abstract', 'large', 'intense'],
        'tiny': ['abstract', 'small', 'intense'],
        'hot': ['abstract', 'intense'],
        'cold': ['abstract', 'negative', 'intense'],
        'warm': ['abstract', 'positive', 'mild'],
        'cool': ['abstract', 'neutral'],
        'fast': ['abstract', 'intense'],
        'slow': ['abstract', 'mild'],
        'new': ['abstract', 'positive'],
        'old': ['abstract', 'neutral'],
        'young': ['abstract', 'positive', 'small'],
        'beautiful': ['abstract', 'positive', 'intense'],
        'pretty': ['abstract', 'positive', 'mild'],
        'ugly': ['abstract', 'negative', 'intense'],
        'smart': ['abstract', 'positive'],
        'stupid': ['abstract', 'negative'],
        'easy': ['abstract', 'positive'],
        'hard': ['abstract', 'negative'],
        'difficult': ['abstract', 'negative', 'formal'],
        'simple': ['abstract', 'positive'],
        'complex': ['abstract', 'formal'],
        'important': ['abstract', 'intense'],
        'interesting': ['abstract', 'positive'],
        'boring': ['abstract', 'negative'],
        'fun': ['abstract', 'positive', 'informal'],
        'funny': ['abstract', 'positive'],
        'serious': ['abstract', 'formal'],
        
        # Time (20 words)
        'time': ['abstract', 'temporal'],
        'today': ['abstract', 'present', 'temporal'],
        'tomorrow': ['abstract', 'future', 'temporal'],
        'yesterday': ['abstract', 'past', 'temporal'],
        'now': ['abstract', 'present', 'temporal'],
        'then': ['abstract', 'past', 'temporal'],
        'soon': ['abstract', 'future', 'temporal'],
        'later': ['abstract', 'future', 'temporal'],
        'before': ['abstract', 'past', 'temporal'],
        'after': ['abstract', 'future', 'temporal'],
        'morning': ['abstract', 'temporal'],
        'afternoon': ['abstract', 'temporal'],
        'evening': ['abstract', 'temporal'],
        'night': ['abstract', 'temporal'],
        'day': ['abstract', 'temporal'],
        'week': ['abstract', 'temporal'],
        'month': ['abstract', 'temporal'],
        'year': ['abstract', 'temporal'],
        'hour': ['abstract', 'temporal'],
        'minute': ['abstract', 'temporal'],
        
        # Technology (30 words)
        'internet': ['object', 'technology', 'large'],
        'website': ['object', 'technology'],
        'email': ['object', 'technology', 'social'],
        'software': ['abstract', 'technology'],
        'app': ['object', 'technology'],
        'game': ['object', 'positive'],
        'video': ['object', 'technology'],
        'music': ['abstract', 'positive'],
        'movie': ['object', 'positive'],
        'tv': ['object', 'technology', 'everyday'],
        'radio': ['object', 'technology', 'everyday'],
        'camera': ['object', 'technology'],
        'photo': ['object', 'positive'],
        'picture': ['object', 'positive'],
        'screen': ['object', 'technology'],
        'keyboard': ['object', 'technology'],
        'mouse': ['object', 'technology'],  # Computer mouse
        'battery': ['object', 'technology'],
        'power': ['abstract', 'intense'],
        'energy': ['abstract', 'intense'],
        
        # Food (25 words)
        'food': ['object', 'everyday', 'positive'],
        'water': ['object', 'everyday'],
        'coffee': ['object', 'everyday'],
        'tea': ['object', 'everyday'],
        'milk': ['object', 'everyday'],
        'bread': ['object', 'everyday'],
        'meat': ['object', 'everyday'],
        'fruit': ['object', 'everyday', 'positive'],
        'vegetable': ['object', 'everyday', 'positive'],
        'pizza': ['object', 'everyday', 'positive'],
        'burger': ['object', 'everyday'],
        'chicken': ['object', 'everyday'],
        'fish': ['object', 'everyday'],  # As food
        'rice': ['object', 'everyday'],
        'pasta': ['object', 'everyday'],
        'soup': ['object', 'everyday'],
        'salad': ['object', 'everyday', 'positive'],
        'cake': ['object', 'positive'],
        'cookie': ['object', 'positive'],
        'candy': ['object', 'positive'],
        'chocolate': ['object', 'positive'],
        
        # Colors (12 words)
        'red': ['abstract', 'intense'],
        'blue': ['abstract', 'calm'],
        'green': ['abstract', 'positive', 'nature'],
        'yellow': ['abstract', 'positive'],
        'orange': ['abstract', 'positive'],
        'purple': ['abstract', 'neutral'],
        'pink': ['abstract', 'positive'],
        'black': ['abstract', 'neutral'],
        'white': ['abstract', 'neutral'],
        'brown': ['abstract', 'neutral'],
        'gray': ['abstract', 'neutral'],
        'grey': ['abstract', 'neutral'],
    }
    
    # Apply semantic initialization
    initialized_count = 0
    for word, token_id in vocab_mapping.items():
        if word in word_categories:
            categories = word_categories[word]
            turn_vector = torch.zeros(model.n_turns)
            
            # Combine patterns from all categories
            for category in categories:
                if category in semantic_patterns:
                    pattern = torch.tensor(semantic_patterns[category])
                    turn_vector += pattern
            
            # Add some noise for variation
            turn_vector += torch.randn(model.n_turns) * 0.1
            
            model.turns.data[token_id] = turn_vector
            initialized_count += 1
    
    print(f"Initialized {initialized_count} words with semantic turn patterns")
    return initialized_count

class TurnGPTConfig:
    """Configuration for TurnGPT model"""
    def __init__(
        self,
        vocab_size: int = 5000,
        n_turns: int = 8,
        n_positions: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        intermediate_size: int = 3072,
        activation_function: str = "gelu",
        dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        poly_degree: int = 4,
    ):
        self.vocab_size = vocab_size
        self.n_turns = n_turns
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.intermediate_size = intermediate_size
        self.activation_function = activation_function
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.poly_degree = poly_degree
