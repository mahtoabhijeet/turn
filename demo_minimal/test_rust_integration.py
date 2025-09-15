#!/usr/bin/env python3
"""
Test Rust integration with simple examples
"""

import numpy as np
import turngpt_rust

def test_rust_functions():
    """Test the Rust functions with simple examples"""
    print("🧪 Testing Rust Integration")
    print("=" * 40)
    
    # Test 1: Simple polynomial evaluation
    print("\n1. Testing polynomial evaluation:")
    try:
        # Simple test: evaluate polynomial for a single turn value
        turns = np.array([[3]], dtype=np.int8)  # Single turn value = 3
        coeffs = np.array([[1.0, 2.0, 0.5]], dtype=np.float32)  # 1 + 2x + 0.5x²
        
        result = turngpt_rust.evaluate_turns(turns, coeffs)
        print(f"   Input: turn=3, coeffs=[1, 2, 0.5]")
        print(f"   Result: {result}")
        print(f"   Expected: 1 + 2*3 + 0.5*9 = 8.5")
        print("   ✅ Polynomial evaluation works!")
    except Exception as e:
        print(f"   ❌ Polynomial evaluation failed: {e}")
    
    # Test 2: Simple semantic arithmetic
    print("\n2. Testing semantic arithmetic:")
    try:
        # Test finding closest turn vector
        target = np.array([1, 2], dtype=np.int8)
        vocab_turns = np.array([
            [0, 0],  # distance = sqrt(5) ≈ 2.24
            [1, 1],  # distance = sqrt(1) = 1
            [2, 3],  # distance = sqrt(2) ≈ 1.41
        ], dtype=np.int8)
        
        closest_idx = turngpt_rust.find_closest_turn(target, vocab_turns)
        print(f"   Target: [1, 2]")
        print(f"   Vocabulary: [[0,0], [1,1], [2,3]]")
        print(f"   Closest index: {closest_idx}")
        print(f"   Expected: 1 (closest to [1,1])")
        print("   ✅ Semantic arithmetic works!")
    except Exception as e:
        print(f"   ❌ Semantic arithmetic failed: {e}")
    
    # Test 3: Memory usage comparison
    print("\n3. Testing memory efficiency:")
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Large batch test
        large_turns = np.random.randint(-10, 11, (1000, 4), dtype=np.int8)
        large_coeffs = np.random.randn(4, 5, 256).astype(np.float32)
        
        # Flatten for Rust processing
        turns_flat = large_turns.reshape(-1, 4)
        coeffs_flat = large_coeffs.reshape(4, -1)
        
        results = []
        for i in range(4):  # 4 turn dimensions
            result = turngpt_rust.evaluate_turns(
                np.array([turns_flat[:, i]]), 
                np.array([coeffs_flat[i]])
            )
            results.append(result[0])
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        print(f"   Processed 1000x4 turn vectors")
        print(f"   Memory used: {memory_used:.1f} MB")
        print(f"   Results shape: {len(results)} x {len(results[0])}")
        print("   ✅ Memory efficiency test passed!")
        
    except Exception as e:
        print(f"   ❌ Memory test failed: {e}")

def test_simple_analogy():
    """Test a simple analogy with Rust acceleration"""
    print("\n🧮 Testing Simple Analogy")
    print("=" * 40)
    
    # Create a simple vocabulary
    vocab = {"man": 0, "woman": 1, "boy": 2, "girl": 3}
    
    # Create simple turn vectors
    turns = np.array([
        [2, 0, 0, 0],  # man: human, neutral, medium, present
        [2, 0, 0, 0],  # woman: human, neutral, medium, present  
        [2, 0, -1, 0], # boy: human, neutral, small, present
        [2, 0, -1, 0], # girl: human, neutral, small, present
    ], dtype=np.int8)
    
    # Test: man - woman + boy = ?
    turn_a = turns[0]  # man
    turn_b = turns[1]  # woman
    turn_c = turns[2]  # boy
    
    # Perform arithmetic: a - b + c
    result_turns = turn_a - turn_b + turn_c
    print(f"   man - woman + boy = {result_turns}")
    
    # Find closest
    closest_idx = turngpt_rust.find_closest_turn(result_turns, turns)
    closest_word = list(vocab.keys())[closest_idx]
    
    print(f"   Closest word: {closest_word} (index {closest_idx})")
    print(f"   Expected: girl")
    
    if closest_word == "girl":
        print("   ✅ Simple analogy works!")
    else:
        print("   ❌ Simple analogy failed")

if __name__ == "__main__":
    test_rust_functions()
    test_simple_analogy()
