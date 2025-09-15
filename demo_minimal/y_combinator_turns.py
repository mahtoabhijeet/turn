#!/usr/bin/env python3
"""
Y-Combinator Self-Recursive Turn Theory Implementation
Based on semantic notes insight: "Y-combinator bootstraps self-reference from nothing"
"""

import numpy as np
from typing import Dict, List, Callable, Tuple
import torch
import torch.nn as nn

class YCombinatorTurnVector:
    """
    Turn vector with Y-combinator self-recursive capabilities
    """
    def __init__(self, coords: np.ndarray):
        self.coords = coords.copy()
        self.history = [coords.copy()]  # Track evolution
    
    def __repr__(self):
        return f"YTurnVector({self.coords.round(3)})"
    
    def distance(self, other: 'YCombinatorTurnVector') -> float:
        """Semantic distance between turn vectors"""
        return np.linalg.norm(self.coords - other.coords)
    
    def add(self, other: 'YCombinatorTurnVector') -> 'YCombinatorTurnVector':
        """Vector addition"""
        return YCombinatorTurnVector(self.coords + other.coords)
    
    def subtract(self, other: 'YCombinatorTurnVector') -> 'YCombinatorTurnVector':
        """Vector subtraction"""
        return YCombinatorTurnVector(self.coords - other.coords)
    
    def normalize(self) -> 'YCombinatorTurnVector':
        """Normalize to unit vector"""
        norm = np.linalg.norm(self.coords)
        if norm > 0:
            return YCombinatorTurnVector(self.coords / norm)
        return YCombinatorTurnVector(self.coords)
    
    def apply_to_self(self) -> 'YCombinatorTurnVector':
        """
        Core self-recursive operation: f(x) where f uses x's own structure
        This is the Y-combinator in action
        """
        # Each dimension influences how the vector transforms itself
        new_coords = np.zeros_like(self.coords)
        
        for i in range(len(self.coords)):
            # Dimension i is transformed by a function of all other dimensions
            influence = 0.0
            for j in range(len(self.coords)):
                if i != j:
                    # Cross-dimensional influence with dampening
                    influence += self.coords[j] * 0.1 * np.sin(self.coords[i] + j)
            
            # Self-transformation: coordinate transforms itself based on its own value
            self_transform = self.coords[i] * (1.0 + 0.05 * np.tanh(self.coords[i]))
            
            new_coords[i] = self_transform + influence
        
        result = YCombinatorTurnVector(new_coords)
        result.history = self.history + [new_coords.copy()]
        return result
    
    def y_combinator_fixed_point(self, max_iterations: int = 100, tolerance: float = 1e-6) -> 'YCombinatorTurnVector':
        """
        Find fixed point using Y-combinator: Y f = f(Y f)
        Keep applying self-transformation until convergence
        """
        current = YCombinatorTurnVector(self.coords)
        
        for iteration in range(max_iterations):
            next_state = current.apply_to_self()
            
            # Check for convergence
            if current.distance(next_state) < tolerance:
                print(f"   Fixed point found after {iteration + 1} iterations")
                return next_state
            
            current = next_state
        
        print(f"   No convergence after {max_iterations} iterations")
        return current
    
    def is_fixed_point(self, tolerance: float = 1e-6) -> bool:
        """Check if this vector is a fixed point under self-application"""
        transformed = self.apply_to_self()
        return self.distance(transformed) < tolerance

class SelfRecursiveTurnSpace:
    """
    Turn space that evolves through self-recursive Y-combinator operations
    """
    def __init__(self):
        self.word_to_turn: Dict[str, YCombinatorTurnVector] = {}
        self.evolution_history: List[Dict[str, YCombinatorTurnVector]] = []
    
    def add_word(self, word: str, coords: np.ndarray):
        """Add a word with its turn coordinates"""
        self.word_to_turn[word] = YCombinatorTurnVector(coords)
    
    def get_turn(self, word: str) -> YCombinatorTurnVector:
        """Get turn vector for a word"""
        return self.word_to_turn.get(word)
    
    def semantic_arithmetic(self, a: str, b: str, c: str) -> Tuple[YCombinatorTurnVector, str, float]:
        """Perform semantic arithmetic: a - b + c"""
        turn_a = self.get_turn(a)
        turn_b = self.get_turn(b)
        turn_c = self.get_turn(c)
        
        if not all([turn_a, turn_b, turn_c]):
            return None, "unknown", float('inf')
        
        # Compute: a - b + c
        result = turn_a.subtract(turn_b).add(turn_c)
        
        # Find closest word
        min_distance = float('inf')
        closest_word = "unknown"
        
        for word, turn_vec in self.word_to_turn.items():
            if word not in [a, b, c]:  # Exclude input words
                distance = result.distance(turn_vec)
                if distance < min_distance:
                    min_distance = distance
                    closest_word = word
        
        return result, closest_word, min_distance
    
    def evolve_through_self_recursion(self, iterations: int = 10):
        """
        Evolve the entire turn space through Y-combinator self-recursion
        Each word's meaning becomes a fixed point of self-application
        """
        print(f"🌀 Evolving turn space through {iterations} self-recursive iterations...")
        
        for iteration in range(iterations):
            print(f"\n   Iteration {iteration + 1}:")
            
            # Store current state
            current_state = {word: YCombinatorTurnVector(turn.coords) 
                           for word, turn in self.word_to_turn.items()}
            self.evolution_history.append(current_state)
            
            # Apply Y-combinator to each word
            evolved_words = {}
            for word, turn_vec in self.word_to_turn.items():
                # Each word evolves by applying itself to itself (Y-combinator)
                evolved = turn_vec.apply_to_self()
                evolved_words[word] = evolved
                
                # Show evolution for a few key words
                if word in ["man", "woman", "king", "queen"] and iteration < 3:
                    distance_moved = turn_vec.distance(evolved)
                    print(f"     {word:8}: moved {distance_moved:.4f}")
            
            # Update the turn space
            self.word_to_turn = evolved_words
            
            # Test semantic arithmetic after evolution
            if iteration % 3 == 0:
                self.test_semantic_arithmetic_evolution()
    
    def test_semantic_arithmetic_evolution(self):
        """Test how semantic arithmetic improves through evolution"""
        test_analogies = [
            ("man", "woman", "boy", "girl"),
            ("king", "queen", "prince", "princess"),
        ]
        
        correct = 0
        total = 0
        
        for a, b, c, expected in test_analogies:
            if all(word in self.word_to_turn for word in [a, b, c, expected]):
                result_vec, predicted, distance = self.semantic_arithmetic(a, b, c)
                total += 1
                if predicted == expected:
                    correct += 1
                    print(f"     ✅ {a} - {b} + {c} = {predicted} (distance: {distance:.3f})")
                else:
                    print(f"     ❌ {a} - {b} + {c} = {predicted} (expected: {expected}, distance: {distance:.3f})")
        
        accuracy = (correct / total * 100) if total > 0 else 0
        print(f"     Evolution accuracy: {accuracy:.1f}%")
    
    def find_semantic_fixed_points(self) -> Dict[str, YCombinatorTurnVector]:
        """Find words that are fixed points under self-application"""
        print("\n🎯 Finding semantic fixed points...")
        
        fixed_points = {}
        
        for word, turn_vec in self.word_to_turn.items():
            # Try to find fixed point for this word
            fixed_point = turn_vec.y_combinator_fixed_point()
            
            if fixed_point.is_fixed_point():
                fixed_points[word] = fixed_point
                print(f"   ✅ {word}: Fixed point found")
            else:
                print(f"   ❌ {word}: No stable fixed point")
        
        return fixed_points

def bootstrap_with_y_combinator():
    """
    Bootstrap turn coordinates using Y-combinator self-recursion
    """
    print("🌟 Y-COMBINATOR BOOTSTRAP EXPERIMENT")
    print("Testing self-recursive meaning generation")
    print("=" * 50)
    
    # Load the discovered 8D coordinates from bootstrap experiment
    try:
        coords_dict = np.load("discovered_8d_coordinates.npy", allow_pickle=True).item()
        print(f"✅ Loaded {len(coords_dict)} bootstrapped 8D coordinates")
    except FileNotFoundError:
        print("❌ No bootstrap coordinates found. Run bootstrap_8d.py first.")
        return
    
    # Create self-recursive turn space
    turn_space = SelfRecursiveTurnSpace()
    
    # Add words with their coordinates
    for word, coords in coords_dict.items():
        turn_space.add_word(word, coords)
    
    # Test initial semantic arithmetic
    print("\n🧮 Initial semantic arithmetic (before Y-combinator):")
    turn_space.test_semantic_arithmetic_evolution()
    
    # Evolve through self-recursion
    turn_space.evolve_through_self_recursion(iterations=15)
    
    # Find fixed points
    fixed_points = turn_space.find_semantic_fixed_points()
    
    # Final test
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Found {len(fixed_points)} semantic fixed points")
    turn_space.test_semantic_arithmetic_evolution()
    
    # Analyze evolution patterns
    print(f"\n📊 Evolution Analysis:")
    for word in ["man", "woman", "king", "queen"]:
        if word in turn_space.word_to_turn:
            turn_vec = turn_space.word_to_turn[word]
            if len(turn_vec.history) > 1:
                initial = turn_vec.history[0]
                final = turn_vec.history[-1]
                total_movement = np.linalg.norm(final - initial)
                print(f"   {word:8}: Total evolution distance = {total_movement:.4f}")
    
    return turn_space

if __name__ == "__main__":
    evolved_space = bootstrap_with_y_combinator()
