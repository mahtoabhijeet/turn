#!/usr/bin/env python3
"""
Simple visualization utilities for Semantic Turn Theory
Shows turn space geometry and semantic relationships
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from turn_embedding import TurnEmbedding, create_semantic_vocab, initialize_semantic_turns

def plot_turn_space_2d(model, vocab, save_path="turn_space.png"):
    """Create a 2D visualization of turn space using PCA"""
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("⚠️  Install sklearn for visualization: pip install scikit-learn")
        return
    
    # Get turn vectors for all words
    words = list(vocab.keys())
    turn_vectors = []
    
    for word in words:
        turns = model.turns[vocab[word]].detach().numpy()
        turn_vectors.append(turns)
    
    turn_matrix = np.array(turn_vectors)
    
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    turns_2d = pca.fit_transform(turn_matrix)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.scatter(turns_2d[:, 0], turns_2d[:, 1], s=100, alpha=0.7, c='blue')
    
    # Label each point
    for i, word in enumerate(words):
        plt.annotate(word, (turns_2d[i, 0], turns_2d[i, 1]), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, alpha=0.8)
    
    plt.title("Semantic Turn Space (2D PCA Projection)", fontsize=16, fontweight='bold')
    plt.xlabel(f"Turn Component 1 (explains {pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"Turn Component 2 (explains {pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.grid(True, alpha=0.3)
    
    # Highlight some key relationships
    highlight_pairs = [("king", "queen"), ("cat", "dog"), ("happy", "sad"), ("big", "small")]
    colors = ['red', 'green', 'orange', 'purple']
    
    for i, (word1, word2) in enumerate(highlight_pairs):
        if word1 in vocab and word2 in vocab:
            idx1, idx2 = words.index(word1), words.index(word2)
            plt.plot([turns_2d[idx1, 0], turns_2d[idx2, 0]], 
                    [turns_2d[idx1, 1], turns_2d[idx2, 1]], 
                    colors[i], alpha=0.6, linewidth=2,
                    label=f"{word1} ↔ {word2}")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Turn space visualization saved to {save_path}")
    print(f"🔍 PCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")

def show_semantic_arithmetic_visual(model, vocab, equation_parts):
    """Visualize semantic arithmetic in turn space"""
    word_a, word_b, word_c = equation_parts
    
    if not all(w in vocab for w in [word_a, word_b, word_c]):
        print("⚠️  Some words not in vocabulary")
        return
    
    turns_a = model.turns[vocab[word_a]].detach().numpy()
    turns_b = model.turns[vocab[word_b]].detach().numpy()  
    turns_c = model.turns[vocab[word_c]].detach().numpy()
    result_turns = turns_a - turns_b + turns_c
    
    # Create bar chart showing the arithmetic
    dimensions = ['Turn 0\n(Concept)', 'Turn 1\n(Behavior)', 'Turn 2\n(Size)', 'Turn 3\n(Context)']
    x = np.arange(len(dimensions))
    width = 0.15
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - 1.5*width, turns_a, width, label=word_a, alpha=0.8, color='blue')
    plt.bar(x - 0.5*width, -turns_b, width, label=f"-{word_b}", alpha=0.8, color='red')
    plt.bar(x + 0.5*width, turns_c, width, label=word_c, alpha=0.8, color='green')
    plt.bar(x + 1.5*width, result_turns, width, label='Result', alpha=0.8, color='orange')
    
    plt.xlabel('Turn Dimensions')
    plt.ylabel('Turn Values')
    plt.title(f'Semantic Arithmetic: {word_a} - {word_b} + {word_c}', fontweight='bold')
    plt.xticks(x, dimensions)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"🧮 Arithmetic breakdown:")
    print(f"  {word_a}: {turns_a.round(2)}")
    print(f"  -{word_b}: {(-turns_b).round(2)}")  
    print(f"  +{word_c}: {turns_c.round(2)}")
    print(f"  Result: {result_turns.round(2)}")

def main():
    """Quick visualization demo"""
    print("📊 SEMANTIC TURN THEORY - VISUALIZATION DEMO")
    print("=" * 50)
    
    # Create and initialize model
    vocab = create_semantic_vocab()
    model = TurnEmbedding(vocab_size=len(vocab), n_turns=4, output_dim=128)
    initialize_semantic_turns(model, vocab)
    
    print("1. Creating turn space visualization...")
    plot_turn_space_2d(model, vocab)
    
    print("\n2. Showing semantic arithmetic visualization...")
    show_semantic_arithmetic_visual(model, vocab, ["king", "man", "woman"])
    
    print("\n✨ Visualizations complete!")

if __name__ == "__main__":
    main()
