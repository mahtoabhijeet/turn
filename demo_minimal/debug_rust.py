#!/usr/bin/env python3
"""
Debug the Rust semantic arithmetic
"""

import torch
import torch.nn as nn
import numpy as np
import turngpt_rust

def debug_semantic_arithmetic():
    """Debug the semantic arithmetic step by step"""
    print("🔍 DEBUGGING SEMANTIC ARITHMETIC")
    print("=" * 50)
    
    # Create simple vocabulary
    vocab = {"man": 0, "woman": 1, "boy": 2, "girl": 3}
    
    # Create model with semantic initialization
    model = nn.Module()
    model.turns = nn.Parameter(torch.tensor([
        [2.0, 0.0, 0.0, 0.0],  # man: human, neutral, medium, present
        [2.0, 0.0, 0.0, 1.0],  # woman: human, neutral, medium, female
        [2.0, 0.0, -1.0, 0.0], # boy: human, neutral, small, present
        [2.0, 0.0, -1.0, 1.0], # girl: human, neutral, small, female
    ]))
    
    print("Initial turn vectors:")
    for word, idx in vocab.items():
        print(f"  {word}: {model.turns[idx].detach().numpy()}")
    
    # Test: man - woman + boy = ?
    print(f"\nTesting: man - woman + boy = ?")
    
    turn_a = model.turns[vocab["man"]].detach().cpu().numpy().astype(np.int8)
    turn_b = model.turns[vocab["woman"]].detach().cpu().numpy().astype(np.int8)
    turn_c = model.turns[vocab["boy"]].detach().cpu().numpy().astype(np.int8)
    
    print(f"  man:   {turn_a}")
    print(f"  woman: {turn_b}")
    print(f"  boy:   {turn_c}")
    
    # Perform arithmetic: a - b + c
    result_turns = turn_a - turn_b + turn_c
    print(f"  result: {result_turns}")
    print(f"  Expected: [2, 0, -1, 1] (girl)")
    
    # Get all vocabulary turns
    vocab_turns = model.turns.detach().cpu().numpy().astype(np.int8)
    print(f"\nVocabulary turns:")
    for word, idx in vocab.items():
        print(f"  {word}: {vocab_turns[idx]}")
    
    # Find closest using Rust
    closest_id = turngpt_rust.find_closest_turn(result_turns, vocab_turns)
    closest_word = list(vocab.keys())[closest_id]
    
    print(f"\nClosest word: {closest_word} (index {closest_id})")
    print(f"Expected: girl")
    
    # Calculate distances manually
    print(f"\nManual distance calculation:")
    for word, idx in vocab.items():
        distance = np.linalg.norm(result_turns - vocab_turns[idx])
        print(f"  {word}: {distance:.3f}")
    
    # Test with floating point precision
    print(f"\nTesting with floating point precision:")
    turn_a_fp = model.turns[vocab["man"]].detach().cpu().numpy()
    turn_b_fp = model.turns[vocab["woman"]].detach().cpu().numpy()
    turn_c_fp = model.turns[vocab["boy"]].detach().cpu().numpy()
    
    result_fp = turn_a_fp - turn_b_fp + turn_c_fp
    print(f"  result (fp): {result_fp}")
    
    # Find closest with floating point
    vocab_turns_fp = model.turns.detach().cpu().numpy()
    distances = [np.linalg.norm(result_fp - vocab_turns_fp[idx]) for idx in range(len(vocab))]
    closest_idx_fp = np.argmin(distances)
    closest_word_fp = list(vocab.keys())[closest_idx_fp]
    
    print(f"  Closest (fp): {closest_word_fp}")
    print(f"  Distances: {distances}")

if __name__ == "__main__":
    debug_semantic_arithmetic()
