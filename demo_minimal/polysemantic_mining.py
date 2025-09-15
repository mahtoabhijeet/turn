#!/usr/bin/env python3
"""
Polysemantic Neuron Mining for Topological Invariant Discovery
Revolutionary approach: Dense neurons aren't bugs - they're topological detectors
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
import json
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class PolysematicNeuron:
    """A neuron that activates for multiple seemingly unrelated concepts"""
    neuron_id: str
    activations: List[Tuple[str, float]]  # (concept, activation_strength)
    layer: int
    model_name: str
    polysemy_score: float  # How "confused" this neuron appears
    
    def get_top_concepts(self, n: int = 20) -> List[str]:
        """Get the top N concepts this neuron activates for"""
        sorted_activations = sorted(self.activations, key=lambda x: x[1], reverse=True)
        return [concept for concept, _ in sorted_activations[:n]]
    
    def get_activation_strength(self, concept: str) -> float:
        """Get activation strength for a specific concept"""
        for c, strength in self.activations:
            if c == concept:
                return strength
        return 0.0

@dataclass 
class TopologicalInvariant:
    """A discovered topological invariant from polysemantic analysis"""
    invariant_id: str
    concepts: List[str]  # Concepts that share this invariant
    geometric_signature: np.ndarray  # 8D signature of this invariant
    persistence_score: float  # How stable this invariant is
    description: str  # Human-readable description
    
    def applies_to(self, concept: str) -> bool:
        """Check if this invariant applies to a concept"""
        return concept in self.concepts

class PolysematicMiner:
    """
    Mines existing neural networks for polysemantic neurons and extracts
    topological invariants from their activation patterns
    """
    
    def __init__(self):
        self.discovered_neurons: List[PolysematicNeuron] = []
        self.extracted_invariants: List[TopologicalInvariant] = []
        
    def simulate_gpt_neuron_extraction(self) -> List[PolysematicNeuron]:
        """
        Simulate extracting polysemantic neurons from GPT-4
        In practice, this would interface with actual model internals
        """
        print("🔍 Simulating polysemantic neuron extraction from large models...")
        
        # Simulate highly polysemantic neurons discovered in real models
        simulated_neurons = [
            PolysematicNeuron(
                neuron_id="gpt4_layer12_neuron_1847",
                activations=[
                    ("golden", 0.89), ("sunset", 0.87), ("honey", 0.85), ("amber", 0.83),
                    ("warmth", 0.81), ("retriever", 0.79), ("autumn", 0.77), ("wheat", 0.75),
                    ("glow", 0.73), ("comfort", 0.71), ("treasure", 0.69), ("light", 0.67)
                ],
                layer=12,
                model_name="GPT-4",
                polysemy_score=0.92  # Very polysemantic
            ),
            
            PolysematicNeuron(
                neuron_id="gpt4_layer8_neuron_2341", 
                activations=[
                    ("king", 0.91), ("authority", 0.88), ("crown", 0.86), ("power", 0.84),
                    ("chess", 0.82), ("lion", 0.80), ("mountain", 0.78), ("tower", 0.76),
                    ("dominant", 0.74), ("peak", 0.72), ("ruler", 0.70), ("height", 0.68)
                ],
                layer=8,
                model_name="GPT-4", 
                polysemy_score=0.87
            ),
            
            PolysematicNeuron(
                neuron_id="gpt4_layer15_neuron_892",
                activations=[
                    ("flow", 0.93), ("river", 0.90), ("music", 0.88), ("dance", 0.86),
                    ("time", 0.84), ("movement", 0.82), ("rhythm", 0.80), ("stream", 0.78),
                    ("grace", 0.76), ("current", 0.74), ("melody", 0.72), ("fluid", 0.70)
                ],
                layer=15,
                model_name="GPT-4",
                polysemy_score=0.95  # Extremely polysemantic
            ),
            
            PolysematicNeuron(
                neuron_id="claude_layer10_neuron_1523",
                activations=[
                    ("love", 0.94), ("passion", 0.91), ("fire", 0.89), ("red", 0.87),
                    ("heart", 0.85), ("intensity", 0.83), ("desire", 0.81), ("warmth", 0.79),
                    ("energy", 0.77), ("life", 0.75), ("vibrant", 0.73), ("strong", 0.71)
                ],
                layer=10,
                model_name="Claude",
                polysemy_score=0.89
            ),
            
            PolysematicNeuron(
                neuron_id="gpt4_layer6_neuron_3156",
                activations=[
                    ("small", 0.88), ("child", 0.86), ("cute", 0.84), ("young", 0.82),
                    ("delicate", 0.80), ("tiny", 0.78), ("innocent", 0.76), ("new", 0.74),
                    ("fragile", 0.72), ("beginning", 0.70), ("soft", 0.68), ("gentle", 0.66)
                ],
                layer=6,
                model_name="GPT-4",
                polysemy_score=0.85
            )
        ]
        
        self.discovered_neurons = simulated_neurons
        print(f"   ✅ Extracted {len(simulated_neurons)} highly polysemantic neurons")
        
        # Analyze polysemy patterns
        avg_polysemy = np.mean([n.polysemy_score for n in simulated_neurons])
        print(f"   📊 Average polysemy score: {avg_polysemy:.3f}")
        
        return simulated_neurons
    
    def analyze_topological_patterns(self, neurons: List[PolysematicNeuron]) -> List[TopologicalInvariant]:
        """
        Analyze polysemantic neurons to discover topological invariants
        """
        print("\n🧬 Analyzing topological patterns in polysemantic activations...")
        
        invariants = []
        
        for neuron in neurons:
            concepts = neuron.get_top_concepts(12)
            
            # Analyze what geometric property these concepts might share
            invariant = self.discover_shared_invariant(neuron, concepts)
            
            if invariant:
                invariants.append(invariant)
                print(f"   🎯 Discovered invariant: {invariant.description}")
                print(f"      Concepts: {', '.join(concepts[:6])}...")
                print(f"      Persistence: {invariant.persistence_score:.3f}")
        
        self.extracted_invariants = invariants
        return invariants
    
    def discover_shared_invariant(self, neuron: PolysematicNeuron, concepts: List[str]) -> Optional[TopologicalInvariant]:
        """
        Discover what topological invariant a set of concepts might share
        """
        # Analyze the semantic patterns to infer topological properties
        concept_features = self.extract_concept_features(concepts)
        
        # Look for geometric clustering patterns
        geometric_signature = self.compute_geometric_signature(concept_features)
        
        # Determine the invariant type and description
        invariant_type, description = self.classify_invariant(concepts, geometric_signature)
        
        if invariant_type:
            return TopologicalInvariant(
                invariant_id=f"invariant_{neuron.neuron_id}",
                concepts=concepts,
                geometric_signature=geometric_signature,
                persistence_score=neuron.polysemy_score,  # High polysemy = strong invariant
                description=description
            )
        
        return None
    
    def extract_concept_features(self, concepts: List[str]) -> Dict[str, np.ndarray]:
        """
        Extract semantic features for concepts to analyze their relationships
        """
        features = {}
        
        # Simulate semantic feature extraction (in practice, use real embeddings)
        for concept in concepts:
            # Create feature vectors based on semantic properties
            feature_vector = np.zeros(20)  # 20 semantic dimensions
            
            # Assign features based on concept properties (simplified simulation)
            if concept in ["golden", "sunset", "honey", "amber", "warmth", "glow", "light"]:
                feature_vector[0] = 1.0  # Warmth/luminosity
                feature_vector[1] = 0.8  # Positive valence
                feature_vector[2] = 0.6  # Organic/natural
                
            elif concept in ["king", "authority", "crown", "power", "dominant", "ruler"]:
                feature_vector[3] = 1.0  # Hierarchy/dominance
                feature_vector[4] = 0.9  # Social structure
                feature_vector[5] = 0.7  # Human institution
                
            elif concept in ["flow", "river", "movement", "stream", "current", "fluid"]:
                feature_vector[6] = 1.0  # Dynamic motion
                feature_vector[7] = 0.8  # Continuity
                feature_vector[8] = 0.6  # Natural process
                
            elif concept in ["love", "passion", "heart", "intensity", "desire", "energy"]:
                feature_vector[9] = 1.0   # Emotional intensity
                feature_vector[10] = 0.9  # Human experience
                feature_vector[11] = 0.8  # Internal state
                
            elif concept in ["small", "child", "cute", "young", "tiny", "delicate"]:
                feature_vector[12] = 1.0  # Size/scale
                feature_vector[13] = 0.9  # Developmental stage
                feature_vector[14] = 0.7  # Protective instinct trigger
            
            # Add noise to make it more realistic
            feature_vector += np.random.normal(0, 0.1, 20)
            features[concept] = feature_vector
            
        return features
    
    def compute_geometric_signature(self, concept_features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute an 8D geometric signature for the topological invariant
        """
        if not concept_features:
            return np.zeros(8)
        
        # Stack all feature vectors
        feature_matrix = np.array(list(concept_features.values()))
        
        # Use PCA to reduce to 8D signature
        if len(feature_matrix) > 1:
            pca = PCA(n_components=min(8, feature_matrix.shape[0], feature_matrix.shape[1]))
            reduced_features = pca.fit_transform(feature_matrix)
            
            # Compute the centroid as the geometric signature
            signature = np.mean(reduced_features, axis=0)
            
            # Pad or truncate to exactly 8 dimensions
            if len(signature) < 8:
                signature = np.pad(signature, (0, 8 - len(signature)))
            else:
                signature = signature[:8]
        else:
            signature = np.zeros(8)
        
        # Normalize to reasonable range
        signature = np.tanh(signature * 2.0)  # Map to [-1, 1] range
        
        return signature
    
    def classify_invariant(self, concepts: List[str], signature: np.ndarray) -> Tuple[Optional[str], str]:
        """
        Classify what type of topological invariant this represents
        """
        # Analyze concept patterns to determine invariant type
        concept_str = " ".join(concepts).lower()
        
        if any(word in concept_str for word in ["golden", "warm", "glow", "light", "sunset"]):
            return "warmth_luminosity", "Warmth/Luminosity Manifold: Concepts sharing warm, glowing, comforting properties"
            
        elif any(word in concept_str for word in ["king", "power", "authority", "dominant"]):
            return "hierarchy_dominance", "Hierarchy/Dominance Manifold: Concepts related to power and social dominance"
            
        elif any(word in concept_str for word in ["flow", "movement", "stream", "current"]):
            return "dynamic_continuity", "Dynamic Continuity Manifold: Concepts involving continuous motion and flow"
            
        elif any(word in concept_str for word in ["love", "passion", "intensity", "heart"]):
            return "emotional_intensity", "Emotional Intensity Manifold: Concepts involving deep emotional states"
            
        elif any(word in concept_str for word in ["small", "young", "tiny", "delicate"]):
            return "scale_development", "Scale/Development Manifold: Concepts related to size and developmental stages"
            
        else:
            return "unknown", f"Unknown Invariant: {len(concepts)} concepts with unclear topological relationship"
    
    def validate_invariants_with_turn_space(self, invariants: List[TopologicalInvariant]) -> Dict[str, float]:
        """
        Test how well discovered invariants map to 8D turn space
        """
        print("\n🎯 Validating invariants against 8D turn space...")
        
        validation_results = {}
        
        for invariant in invariants:
            # Test if the 8D signature produces meaningful semantic arithmetic
            signature = invariant.geometric_signature
            
            # Simulate testing semantic relationships
            coherence_score = self.test_semantic_coherence(invariant.concepts, signature)
            arithmetic_score = self.test_semantic_arithmetic(invariant.concepts, signature)
            
            overall_score = (coherence_score + arithmetic_score) / 2
            validation_results[invariant.invariant_id] = overall_score
            
            print(f"   📊 {invariant.description}")
            print(f"      Coherence: {coherence_score:.3f}, Arithmetic: {arithmetic_score:.3f}")
            print(f"      Overall validation: {overall_score:.3f}")
        
        return validation_results
    
    def test_semantic_coherence(self, concepts: List[str], signature: np.ndarray) -> float:
        """
        Test if concepts with the same signature are semantically coherent
        """
        # Simulate coherence testing
        # In practice, this would test if the signature predicts semantic similarity
        
        # Simple heuristic: concepts should have similar "semantic fingerprints"
        concept_similarities = []
        
        for i, concept_a in enumerate(concepts):
            for j, concept_b in enumerate(concepts[i+1:], i+1):
                # Simulate semantic similarity based on concept properties
                sim = self.simulate_concept_similarity(concept_a, concept_b)
                concept_similarities.append(sim)
        
        if concept_similarities:
            avg_similarity = np.mean(concept_similarities)
            # High similarity = high coherence
            return min(1.0, avg_similarity * 1.2)
        
        return 0.0
    
    def test_semantic_arithmetic(self, concepts: List[str], signature: np.ndarray) -> float:
        """
        Test if the signature enables semantic arithmetic
        """
        # Simulate arithmetic testing
        # In practice, test if signature-based operations produce meaningful results
        
        if len(concepts) < 3:
            return 0.0
        
        # Test a few arithmetic operations
        scores = []
        
        for i in range(min(3, len(concepts) - 2)):
            # Simulate: concept_a - concept_b + concept_c should be meaningful
            concept_a, concept_b, concept_c = concepts[i], concepts[i+1], concepts[i+2]
            
            # Simple heuristic based on signature properties
            arithmetic_coherence = np.abs(np.sum(signature)) / (np.linalg.norm(signature) + 1e-8)
            scores.append(min(1.0, arithmetic_coherence))
        
        return np.mean(scores) if scores else 0.0
    
    def simulate_concept_similarity(self, concept_a: str, concept_b: str) -> float:
        """
        Simulate semantic similarity between concepts
        """
        # Simple similarity heuristics
        similar_groups = [
            ["golden", "sunset", "honey", "amber", "warmth", "glow", "light"],
            ["king", "authority", "crown", "power", "dominant", "ruler"],
            ["flow", "river", "movement", "stream", "current", "fluid"],
            ["love", "passion", "heart", "intensity", "desire", "energy"],
            ["small", "child", "cute", "young", "tiny", "delicate"]
        ]
        
        for group in similar_groups:
            if concept_a in group and concept_b in group:
                return 0.8 + np.random.normal(0, 0.1)
        
        # Different groups
        return 0.2 + np.random.normal(0, 0.1)
    
    def generate_turn_space_initialization(self, invariants: List[TopologicalInvariant]) -> Dict[str, np.ndarray]:
        """
        Generate 8D turn space coordinates based on discovered invariants
        """
        print("\n🌟 Generating Turn Space initialization from topological invariants...")
        
        turn_coordinates = {}
        
        for invariant in invariants:
            base_signature = invariant.geometric_signature
            
            # Generate coordinates for each concept in this invariant
            for i, concept in enumerate(invariant.concepts):
                # Add small variations to the base signature for each concept
                variation = np.random.normal(0, 0.2, 8)  # Small random variation
                concept_coords = base_signature + variation
                
                # Ensure coordinates stay in reasonable range
                concept_coords = np.clip(concept_coords, -3.0, 3.0)
                
                turn_coordinates[concept] = concept_coords
                
                print(f"   📍 {concept:12}: {concept_coords.round(2)}")
        
        print(f"\n✅ Generated {len(turn_coordinates)} turn space coordinates from invariants")
        return turn_coordinates
    
    def run_complete_mining_pipeline(self) -> Dict[str, np.ndarray]:
        """
        Run the complete polysemantic mining pipeline
        """
        print("🚀 POLYSEMANTIC MINING PIPELINE")
        print("Revolutionary hypothesis: Dense neurons encode topological invariants")
        print("=" * 70)
        
        # Step 1: Extract polysemantic neurons
        neurons = self.simulate_gpt_neuron_extraction()
        
        # Step 2: Discover topological invariants
        invariants = self.analyze_topological_patterns(neurons)
        
        # Step 3: Validate invariants
        validation_results = self.validate_invariants_with_turn_space(invariants)
        
        # Step 4: Generate turn space coordinates
        turn_coordinates = self.generate_turn_space_initialization(invariants)
        
        # Summary
        print(f"\n🎯 MINING RESULTS SUMMARY:")
        print(f"   Neurons analyzed: {len(neurons)}")
        print(f"   Invariants discovered: {len(invariants)}")
        print(f"   Average validation score: {np.mean(list(validation_results.values())):.3f}")
        print(f"   Turn coordinates generated: {len(turn_coordinates)}")
        
        # Save results
        np.save("polysemantic_turn_coordinates.npy", turn_coordinates)
        print(f"\n💾 Saved polysemantic-derived coordinates to 'polysemantic_turn_coordinates.npy'")
        
        return turn_coordinates

def main():
    """
    Main function to run polysemantic mining experiment
    """
    miner = PolysematicMiner()
    
    # Run the complete pipeline
    coordinates = miner.run_complete_mining_pipeline()
    
    print(f"\n🌟 POLYSEMANTIC MINING COMPLETE!")
    print(f"   Ready to test these coordinates in Turn Theory system")
    print(f"   Next step: Validate semantic arithmetic with discovered coordinates")
    
    return coordinates, miner

if __name__ == "__main__":
    coords, miner = main()
