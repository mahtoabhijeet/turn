"""
TurnGPT Model: GPT-2 Architecture with Semantic Turn Embeddings
Combines the breakthrough turn theory with proven transformer architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math

from turn_embedding_scaled import ScaledTurnEmbedding, TurnGPTConfig

class TurnGPTAttention(nn.Module):
    """Multi-head attention optimized for M1 MacBook"""
    def __init__(self, config: TurnGPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        assert self.n_embd % self.n_head == 0
        self.head_dim = self.n_embd // self.n_head
        
        # Combined QKV projection for efficiency
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)
        
        # Dropout
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        
        # Causal mask for autoregressive generation
        self.register_buffer("bias", torch.tril(torch.ones(config.n_positions, config.n_positions))
                            .view(1, 1, config.n_positions, config.n_positions))

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.size()  # batch, sequence length, embedding dimensionality
        
        # Calculate QKV in one go
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        
        # Attention weights
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply causal mask
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            att = att + attention_mask
        
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Apply attention to values
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # Reassemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        
        return y

class TurnGPTMLP(nn.Module):
    """Feed-forward network"""
    def __init__(self, config: TurnGPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.intermediate_size)
        self.c_proj = nn.Linear(config.intermediate_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
        # Use GELU activation for better performance
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class TurnGPTBlock(nn.Module):
    """Transformer block with turn-aware processing"""
    def __init__(self, config: TurnGPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = TurnGPTAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = TurnGPTMLP(config)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture (more stable)
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class TurnGPTModel(nn.Module):
    """
    The main TurnGPT model: GPT-2 architecture with semantic turn embeddings
    This is where the breakthrough happens - meaning as counting + transformers
    """
    def __init__(self, config: TurnGPTConfig):
        super().__init__()
        self.config = config
        
        # Core breakthrough: Turn-based embeddings instead of traditional embeddings
        self.wte = ScaledTurnEmbedding(
            vocab_size=config.vocab_size,
            n_turns=config.n_turns,
            output_dim=config.n_embd,
            poly_degree=config.poly_degree,
            dropout=config.dropout,
            max_position_embeddings=config.n_positions
        )
        
        # Standard transformer blocks
        self.h = nn.ModuleList([TurnGPTBlock(config) for _ in range(config.n_layer)])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # Enable gradient checkpointing for memory efficiency on M1
        self.gradient_checkpointing = False
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights using Xavier/Kaiming initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        
        batch_size, seq_length = input_ids.shape
        
        # Generate embeddings from turn integers (the breakthrough!)
        hidden_states = self.wte(input_ids, position_ids)
        
        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
        
        # Pass through transformer blocks
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        for i, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    attention_mask,
                )
            else:
                hidden_states = block(hidden_states, attention_mask)
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        return {
            'last_hidden_state': hidden_states,
            'hidden_states': all_hidden_states,
            'attentions': all_attentions,
        }
    
    def get_turn_embeddings(self) -> ScaledTurnEmbedding:
        """Access the turn embedding layer for semantic arithmetic"""
        return self.wte
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        self.gradient_checkpointing = True
    
    def get_memory_footprint(self) -> Dict[str, int]:
        """Calculate memory footprint for M1 optimization"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Embedding comparison
        compression_stats = self.wte.get_compression_stats()
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'embedding_compression_ratio': compression_stats['compression_ratio'],
            'embedding_memory_savings': compression_stats['memory_savings_percent'],
        }

class TurnGPTLMHeadModel(nn.Module):
    """
    TurnGPT with language modeling head for text generation
    This is the complete model ready for training and inference
    """
    def __init__(self, config: TurnGPTConfig):
        super().__init__()
        self.config = config
        
        # Core TurnGPT model
        self.transformer = TurnGPTModel(config)
        
        # Language modeling head (no bias to reduce parameters)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie weights between embeddings and lm_head for efficiency
        self.tie_weights()
        
        # Initialize weights
        self.apply(self._init_weights)

    def tie_weights(self):
        """Tie input and output embeddings for parameter efficiency"""
        # This is tricky with turn embeddings, so we'll skip for now
        # Could be implemented by sharing polynomial coefficients
        pass

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        
        # Forward pass through transformer
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs
        )
        
        hidden_states = transformer_outputs['last_hidden_state']
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': transformer_outputs.get('hidden_states'),
            'attentions': transformer_outputs.get('attentions'),
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text using the TurnGPT model
        """
        self.eval()
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        # Generate tokens one by one
        with torch.no_grad():
            for _ in range(max_length - seq_len):
                # Forward pass
                outputs = self.forward(input_ids)
                logits = outputs['logits']
                
                # Get logits for the last token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = torch.gather(sorted_indices_to_remove, 1, sorted_indices.argsort(1))
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to input_ids
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # Check for EOS token
                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break
        
        return input_ids
    
    def get_semantic_calculator(self):
        """Get semantic calculator for turn arithmetic"""
        return self.transformer.get_turn_embeddings()

def create_turngpt_config(
    vocab_size: int = 5000,
    model_size: str = "medium"
) -> TurnGPTConfig:
    """Create TurnGPT configuration based on target model size"""
    
    size_configs = {
        "tiny": {
            "n_layer": 4,
            "n_head": 4,
            "n_embd": 256,
            "intermediate_size": 1024,
        },
        "small": {
            "n_layer": 8,
            "n_head": 8,
            "n_embd": 512,
            "intermediate_size": 2048,
        },
        "medium": {
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
            "intermediate_size": 3072,
        },
        "large": {
            "n_layer": 16,
            "n_head": 16,
            "n_embd": 1024,
            "intermediate_size": 4096,
        }
    }
    
    config_dict = size_configs.get(model_size, size_configs["medium"])
    config_dict["vocab_size"] = vocab_size
    
    return TurnGPTConfig(**config_dict)
