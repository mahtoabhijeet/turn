#!/usr/bin/env python3
"""
Test Polysemantic-Derived Turn Coordinates
Validate that coordinates extracted from polysemantic neurons work better
than manually initialized ones
"""

import numpy as np
from turn_embedding import TurnEmbedding, SemanticCalculator
import torch
from typing import Dict, List, Tuple

def load_polysemantic_coordinates() -> Dict[str, np.ndarray]:
    """Load coordinates derived from polysemantic neuron analysis"""
    try:
        coords_dict = np.load("polysemantic_turn_coordinates.npy", allow_pickle=True).item()
        print(f"✅ Loaded {len(coords_dict)} polysemantic-derived coordinates")
        return coords_dict
    except FileNotFoundError:
        print("❌ Polysemantic coordinates not found. Run polysemantic_mining.py first.")
        return {}

def create_polysemantic_turn_space(coords_dict: Dict[str, np.ndarray]) -> TurnEmbedding:
    """Create TurnEmbedding model using polysemantic-derived coordinates"""
    
    # Convert to the format expected by TurnEmbedding
    vocab = list(coords_dict.keys())
    vocab_size = len(vocab)
    
    # Create model with 8 turns (matching our 8D coordinates)
    model = TurnEmbedding(vocab_size=vocab_size, n_turns=8, output_dim=128, poly_degree=3)
    
    # Initialize with polysemantic coordinates
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    
    # Set turn parameters from discovered coordinates
    with torch.no_grad():
        for word, coords in coords_dict.items():
            if word in word_to_idx:
                idx = word_to_idx[word]
                # Convert to integers in range [-5, 5] for turn parameters (matching TurnEmbedding range)
                turn_coords = np.clip(np.round(coords * 5), -5, 5).astype(float)
                model.turns[idx] = torch.tensor(turn_coords, dtype=torch.float32)
    
    return model, vocab, word_to_idx

def test_semantic_arithmetic_polysemantic(model: TurnEmbedding, vocab: List[str], word_to_idx: Dict[str, int]):
    """Test semantic arithmetic using polysemantic-derived coordinates"""
    
    print("\n🧮 Testing Semantic Arithmetic with Polysemantic Coordinates")
    print("=" * 60)
    
    # Create vocab dictionary for semantic arithmetic
    vocab_dict = {word: idx for idx, word in enumerate(vocab)}
    
    # Test analogies based on the topological invariants we discovered
    test_analogies = [
        # Warmth/Luminosity manifold tests
        ("golden", "honey", "sunset", "amber"),     # Within same manifold
        ("warmth", "comfort", "glow", "light"),     # Within same manifold
        
        # Hierarchy/Dominance manifold tests  
        ("king", "crown", "authority", "power"),    # Within same manifold
        ("lion", "dominant", "ruler", "peak"),      # Within same manifold
        
        # Dynamic Continuity manifold tests
        ("flow", "river", "stream", "current"),     # Within same manifold
        ("music", "rhythm", "dance", "movement"),   # Within same manifold
        
        # Cross-manifold tests (should be harder)
        ("king", "small", "large", "ruler"),        # Hierarchy + Scale
        ("love", "tiny", "huge", "passion"),        # Emotion + Scale
        ("flow", "golden", "silver", "stream"),     # Dynamic + Warmth
        
        # Traditional analogies if available
        ("king", "man", "woman", "queen"),          # If we have these words
        ("love", "hate", "joy", "sadness"),         # Emotional opposites
    ]
    
    results = []
    successful_tests = 0
    total_tests = 0
    
    for a, b, c, expected_d in test_analogies:
        # Check if all words are in vocabulary
        if all(word in vocab_dict for word in [a, b, c, expected_d]):
            result_turns, result_word = model.semantic_arithmetic(a, b, c, vocab_dict)
            
            # Calculate distance to expected word
            expected_turns = model.get_turn_vector(vocab_dict[expected_d])
            distance = torch.norm(result_turns - expected_turns).item()
            
            total_tests += 1
            success = result_word == expected_d
            if success:
                successful_tests += 1
            
            status = "✅" if success else "❌"
            print(f"{status} {a} - {b} + {c} = {result_word} (expected: {expected_d}, distance: {distance:.3f})")
            
            results.append({
                'analogy': (a, b, c, expected_d),
                'predicted': result_word,
                'distance': distance,
                'success': success
            })
        else:
            missing = [w for w in [a, b, c, expected_d] if w not in vocab_dict]
            print(f"⏭️  Skipping {a} - {b} + {c} = {expected_d} (missing: {missing})")
    
    accuracy = (successful_tests / total_tests * 100) if total_tests > 0 else 0
    avg_distance = np.mean([r['distance'] for r in results]) if results else float('inf')
    
    print(f"\n📊 POLYSEMANTIC COORDINATE RESULTS:")
    print(f"   Tests completed: {total_tests}")
    print(f"   Successful: {successful_tests}")
    print(f"   Accuracy: {accuracy:.1f}%")
    print(f"   Average distance: {avg_distance:.3f}")
    
    return results, accuracy, avg_distance

def test_topological_clustering(coords_dict: Dict[str, np.ndarray]):
    """Test if polysemantic coordinates show expected topological clustering"""
    
    print("\n🔍 Testing Topological Clustering of Polysemantic Coordinates")
    print("=" * 60)
    
    # Define expected clusters based on our discovered invariants
    expected_clusters = {
        "Warmth/Luminosity": ["golden", "sunset", "honey", "amber", "warmth", "glow", "light", "comfort"],
        "Hierarchy/Dominance": ["king", "authority", "crown", "power", "lion", "dominant", "ruler", "peak"],
        "Dynamic Continuity": ["flow", "river", "music", "dance", "movement", "rhythm", "stream", "current"],
        "Emotional Intensity": ["love", "passion", "fire", "heart", "intensity", "desire", "energy", "strong"],
        "Scale/Development": ["small", "child", "cute", "young", "delicate", "tiny", "innocent", "gentle"]
    }
    
    cluster_coherence_scores = {}
    
    for cluster_name, expected_words in expected_clusters.items():
        # Get coordinates for words in this cluster
        cluster_coords = []
        available_words = []
        
        for word in expected_words:
            if word in coords_dict:
                cluster_coords.append(coords_dict[word])
                available_words.append(word)
        
        if len(cluster_coords) < 2:
            print(f"⏭️  Skipping {cluster_name} (insufficient words: {available_words})")
            continue
        
        # Calculate intra-cluster distances
        cluster_coords = np.array(cluster_coords)
        intra_distances = []
        
        for i in range(len(cluster_coords)):
            for j in range(i+1, len(cluster_coords)):
                distance = np.linalg.norm(cluster_coords[i] - cluster_coords[j])
                intra_distances.append(distance)
        
        avg_intra_distance = np.mean(intra_distances)
        
        # Calculate distances to other clusters (for comparison)
        inter_distances = []
        other_coords = []
        
        for other_cluster, other_words in expected_clusters.items():
            if other_cluster != cluster_name:
                for word in other_words:
                    if word in coords_dict:
                        other_coords.append(coords_dict[word])
        
        if other_coords:
            other_coords = np.array(other_coords)
            for cluster_coord in cluster_coords:
                for other_coord in other_coords:
                    distance = np.linalg.norm(cluster_coord - other_coord)
                    inter_distances.append(distance)
            
            avg_inter_distance = np.mean(inter_distances)
            
            # Coherence score: ratio of inter-cluster to intra-cluster distance
            # Higher is better (tight clusters, well separated)
            coherence = avg_inter_distance / (avg_intra_distance + 1e-8)
            cluster_coherence_scores[cluster_name] = coherence
            
            print(f"📊 {cluster_name}:")
            print(f"   Words: {', '.join(available_words)}")
            print(f"   Intra-cluster distance: {avg_intra_distance:.3f}")
            print(f"   Inter-cluster distance: {avg_inter_distance:.3f}")
            print(f"   Coherence score: {coherence:.3f}")
        
        else:
            print(f"⚠️  {cluster_name}: No other clusters for comparison")
    
    if cluster_coherence_scores:
        avg_coherence = np.mean(list(cluster_coherence_scores.values()))
        print(f"\n🎯 CLUSTERING ANALYSIS:")
        print(f"   Average coherence score: {avg_coherence:.3f}")
        print(f"   Best cluster: {max(cluster_coherence_scores, key=cluster_coherence_scores.get)}")
        print(f"   Worst cluster: {min(cluster_coherence_scores, key=cluster_coherence_scores.get)}")
        
        return cluster_coherence_scores, avg_coherence
    
    return {}, 0.0

def compare_with_baseline():
    """Compare polysemantic coordinates with previous approaches"""
    
    print("\n📈 Comparison with Previous Approaches")
    print("=" * 50)
    
    # Previous results from our experiments
    baseline_results = {
        "Manual 100-word": {"accuracy": 70.6, "method": "Hand-crafted semantic initialization"},
        "1000-word scaled": {"accuracy": 17.6, "method": "Scaled manual initialization"},
        "Bootstrap PCA": {"accuracy": 0.0, "method": "PCA dimension reduction"},
        "Bootstrap ICA": {"accuracy": 0.0, "method": "ICA dimension reduction"},
        "Y-Combinator": {"accuracy": 0.0, "method": "Self-recursive evolution"},
    }
    
    print("Previous results:")
    for method, data in baseline_results.items():
        print(f"   {method:20}: {data['accuracy']:5.1f}% ({data['method']})")
    
    return baseline_results

def main():
    """
    Main function to test polysemantic-derived coordinates
    """
    print("🌟 TESTING POLYSEMANTIC-DERIVED TURN COORDINATES")
    print("Revolutionary test: Do extracted topological invariants work better?")
    print("=" * 70)
    
    # Load polysemantic coordinates
    coords_dict = load_polysemantic_coordinates()
    if not coords_dict:
        return
    
    # Create Turn Theory model with polysemantic coordinates
    print(f"\n🏗️  Creating TurnEmbedding model with {len(coords_dict)} polysemantic coordinates...")
    model, vocab, word_to_idx = create_polysemantic_turn_space(coords_dict)
    
    # Test semantic arithmetic
    arithmetic_results, accuracy, avg_distance = test_semantic_arithmetic_polysemantic(model, vocab, word_to_idx)
    
    # Test topological clustering
    cluster_scores, avg_coherence = test_topological_clustering(coords_dict)
    
    # Compare with baseline
    baseline_results = compare_with_baseline()
    
    # Final analysis
    print(f"\n🎯 FINAL ANALYSIS:")
    print(f"   Polysemantic accuracy: {accuracy:.1f}%")
    print(f"   Average semantic distance: {avg_distance:.3f}")
    print(f"   Topological coherence: {avg_coherence:.3f}")
    
    # Determine if this is a breakthrough
    best_baseline = max(baseline_results.values(), key=lambda x: x['accuracy'])['accuracy']
    
    if accuracy > best_baseline:
        improvement = accuracy - best_baseline
        print(f"\n🚀 BREAKTHROUGH ACHIEVED!")
        print(f"   Improvement over best baseline: +{improvement:.1f} percentage points")
        print(f"   Polysemantic mining successfully extracted better semantic structure!")
    elif accuracy > 0:
        print(f"\n🔄 PARTIAL SUCCESS:")
        print(f"   Polysemantic coordinates show semantic structure ({accuracy:.1f}% accuracy)")
        print(f"   Need refinement to beat baseline ({best_baseline:.1f}%)")
    else:
        print(f"\n❌ APPROACH NEEDS REVISION:")
        print(f"   Polysemantic extraction didn't capture usable semantic structure")
        print(f"   Consider different invariant discovery methods")
    
    # Save results for progress tracking
    results_summary = {
        "polysemantic_accuracy": accuracy,
        "avg_distance": avg_distance,
        "topological_coherence": avg_coherence,
        "total_words": len(coords_dict),
        "successful_tests": len([r for r in arithmetic_results if r['success']]),
        "total_tests": len(arithmetic_results)
    }
    
    np.save("polysemantic_test_results.npy", results_summary)
    print(f"\n💾 Results saved to 'polysemantic_test_results.npy'")
    
    return results_summary

if __name__ == "__main__":
    results = main()
