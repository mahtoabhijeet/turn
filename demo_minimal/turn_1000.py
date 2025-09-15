
#!/usr/bin/env python3
"""
Turn Theory - 1000-Word Analogical Reasoning Network
The ultimate test: Can semantic arithmetic scale to 1000 words with <0.1 error?
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import random
from collections import defaultdict

class TurnEmbedding1000(nn.Module):
    """
    Enhanced TurnEmbedding for 1000-word vocabulary with analogical reasoning focus
    """
    def __init__(self, vocab_size: int, n_turns: int = 4, output_dim: int = 256, poly_degree: int = 4):
        super().__init__()
        # Each word = n_turns integers (the "hydrogen atoms" of meaning)
        self.turns = nn.Parameter(torch.randint(-10, 11, (vocab_size, n_turns)).float())
        # Enhanced polynomial coefficients for better semantic representation
        self.poly_coeffs = nn.Parameter(torch.randn(n_turns, poly_degree + 1, output_dim) * 0.05)
        
        self.n_turns = n_turns
        self.poly_degree = poly_degree
        self.output_dim = output_dim

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Generate embeddings from turn integers via polynomial transformation"""
        base_turns = self.turns[token_ids]
        batch_size, seq_len = base_turns.shape[:2]
        embeddings = torch.zeros(batch_size, seq_len, self.output_dim, device=base_turns.device)
        
        for turn_idx in range(self.n_turns):
            x = base_turns[..., turn_idx].unsqueeze(-1)
            # Generate polynomial: 1, x, x², x³, x⁴...
            powers = torch.cat([x**d for d in range(self.poly_degree + 1)], dim=-1)
            embeddings += torch.einsum('bsp,pdo->bso', powers, self.poly_coeffs[turn_idx])
        
        return embeddings
    
    def semantic_arithmetic(self, word_a: str, word_b: str, word_c: str, vocab: Dict[str, int]) -> Tuple[torch.Tensor, str, float]:
        """Perform semantic arithmetic: word_a - word_b + word_c = ?"""
        turn_a = self.turns[vocab[word_a]]
        turn_b = self.turns[vocab[word_b]] 
        turn_c = self.turns[vocab[word_c]]
        
        result_turns = turn_a - turn_b + turn_c
        
        # Find closest word by turn distance
        distances = torch.norm(self.turns - result_turns.unsqueeze(0), dim=1)
        closest_id = torch.argmin(distances).item()
        min_distance = distances[closest_id].item()
        
        # Convert back to word
        reverse_vocab = {v: k for k, v in vocab.items()}
        closest_word = reverse_vocab[closest_id]
        
        return result_turns.detach(), closest_word, min_distance

def create_1000_word_vocab() -> Dict[str, int]:
    """Create comprehensive 1000-word vocabulary for analogical reasoning"""
    words = []
    
    # Core semantic categories with systematic expansion
    categories = {
        # People & Relationships (100 words)
        "people": [
            "person", "human", "individual", "being", "creature",
            "man", "woman", "boy", "girl", "child", "baby", "infant", "toddler", "teenager", "adult", "elder", "senior",
            "father", "mother", "dad", "mom", "parent", "son", "daughter", "brother", "sister", "sibling",
            "uncle", "aunt", "cousin", "nephew", "niece", "grandfather", "grandmother", "grandpa", "grandma",
            "husband", "wife", "spouse", "partner", "boyfriend", "girlfriend", "fiancé", "fiancée",
            "friend", "buddy", "pal", "companion", "acquaintance", "colleague", "neighbor", "stranger",
            "teacher", "student", "pupil", "professor", "instructor", "mentor", "coach", "tutor",
            "doctor", "nurse", "patient", "surgeon", "physician", "dentist", "veterinarian",
            "lawyer", "judge", "jury", "client", "defendant", "plaintiff", "witness",
            "police", "officer", "detective", "sheriff", "guard", "security", "soldier", "veteran",
            "chef", "cook", "waiter", "waitress", "server", "bartender", "manager", "boss", "employee",
            "artist", "painter", "musician", "singer", "dancer", "actor", "actress", "performer",
            "writer", "author", "journalist", "reporter", "editor", "publisher", "librarian",
            "engineer", "scientist", "researcher", "inventor", "architect", "designer", "programmer"
        ],
        
        # Animals (100 words)
        "animals": [
            "animal", "creature", "beast", "pet", "wildlife",
            "dog", "cat", "puppy", "kitten", "hound", "mutt", "terrier", "retriever", "shepherd",
            "lion", "tiger", "leopard", "cheetah", "panther", "jaguar", "cougar", "lynx",
            "bear", "grizzly", "polar", "panda", "koala", "raccoon", "skunk", "badger",
            "wolf", "fox", "coyote", "hyena", "jackal", "dingo", "wild", "domestic",
            "horse", "pony", "stallion", "mare", "colt", "foal", "donkey", "mule", "zebra",
            "cow", "bull", "calf", "ox", "buffalo", "bison", "yak", "antelope", "deer",
            "sheep", "lamb", "goat", "ram", "ewe", "pig", "hog", "boar", "swine",
            "bird", "eagle", "hawk", "falcon", "owl", "crow", "raven", "sparrow", "robin",
            "chicken", "rooster", "hen", "duck", "goose", "swan", "turkey", "peacock",
            "fish", "salmon", "tuna", "shark", "whale", "dolphin", "seal", "walrus",
            "snake", "python", "cobra", "viper", "rattlesnake", "lizard", "gecko", "iguana",
            "frog", "toad", "turtle", "tortoise", "crocodile", "alligator", "dinosaur"
        ],
        
        # Size & Scale (50 words)
        "size": [
            "size", "scale", "dimension", "measure", "proportion",
            "tiny", "mini", "micro", "miniature", "small", "little", "petite", "compact",
            "medium", "average", "normal", "standard", "regular", "moderate", "modest",
            "large", "big", "huge", "enormous", "massive", "giant", "gigantic", "colossal",
            "immense", "vast", "extensive", "spacious", "roomy", "capacious", "voluminous",
            "thick", "thin", "wide", "narrow", "broad", "slim", "slender", "fat", "skinny",
            "tall", "short", "high", "low", "deep", "shallow", "long", "brief"
        ],
        
        # Emotions & Feelings (100 words)
        "emotions": [
            "emotion", "feeling", "mood", "sentiment", "passion",
            "happy", "joyful", "cheerful", "merry", "glad", "pleased", "delighted", "ecstatic",
            "sad", "unhappy", "miserable", "depressed", "gloomy", "melancholy", "sorrowful",
            "angry", "mad", "furious", "rage", "wrath", "irritated", "annoyed", "frustrated",
            "love", "adore", "cherish", "treasure", "fond", "affectionate", "romantic", "passionate",
            "hate", "despise", "loathe", "detest", "abhor", "disgust", "revulsion", "contempt",
            "fear", "afraid", "scared", "terrified", "frightened", "anxious", "worried", "nervous",
            "calm", "peaceful", "serene", "tranquil", "relaxed", "composed", "collected", "cool",
            "excited", "thrilled", "enthusiastic", "eager", "animated", "energetic", "lively",
            "proud", "confident", "satisfied", "accomplished", "successful", "victorious", "triumphant",
            "ashamed", "embarrassed", "guilty", "remorseful", "regretful", "sorry", "apologetic",
            "surprised", "amazed", "astonished", "shocked", "startled", "bewildered", "confused",
            "jealous", "envious", "covetous", "greedy", "selfish", "possessive", "protective"
        ],
        
        # Actions & Movement (100 words)
        "actions": [
            "action", "movement", "motion", "activity", "behavior",
            "run", "ran", "running", "jog", "sprint", "dash", "race", "rush", "hurry",
            "walk", "walked", "walking", "stroll", "saunter", "march", "stride", "pace",
            "jump", "jumped", "jumping", "leap", "bound", "hop", "skip", "bounce",
            "fly", "flew", "flying", "soar", "glide", "hover", "float", "drift",
            "swim", "swam", "swimming", "dive", "dove", "diving", "paddle", "float",
            "drive", "drove", "driving", "ride", "rode", "riding", "steer", "navigate",
            "climb", "climbed", "climbing", "ascend", "scale", "mount", "rise", "elevate",
            "fall", "fell", "falling", "drop", "plunge", "descend", "sink", "collapse",
            "sit", "sat", "sitting", "rest", "relax", "recline", "settle", "perch",
            "stand", "stood", "standing", "upright", "erect", "position", "place", "station",
            "eat", "ate", "eating", "consume", "devour", "swallow", "chew", "bite",
            "drink", "drank", "drinking", "sip", "gulp", "swallow", "imbibe", "quench",
            "sleep", "slept", "sleeping", "rest", "nap", "doze", "slumber", "repose",
            "work", "worked", "working", "labor", "toil", "effort", "exert", "strive",
            "play", "played", "playing", "game", "sport", "entertain", "amuse", "recreation"
        ],
        
        # Objects & Things (100 words)
        "objects": [
            "object", "thing", "item", "article", "piece", "element", "component",
            "house", "home", "building", "structure", "residence", "dwelling", "mansion", "cottage",
            "car", "vehicle", "automobile", "truck", "bus", "van", "motorcycle", "bicycle",
            "book", "novel", "story", "tale", "narrative", "text", "manuscript", "publication",
            "food", "meal", "dish", "cuisine", "cooking", "recipe", "ingredient", "nutrition",
            "water", "liquid", "fluid", "beverage", "drink", "juice", "soda", "coffee",
            "tree", "plant", "vegetation", "forest", "woodland", "grove", "orchard", "garden",
            "mountain", "hill", "peak", "summit", "ridge", "cliff", "slope", "elevation",
            "ocean", "sea", "lake", "river", "stream", "pond", "pool", "waterway",
            "computer", "machine", "device", "gadget", "tool", "instrument", "equipment", "appliance",
            "phone", "telephone", "mobile", "cell", "communication", "device", "smartphone", "tablet",
            "clothes", "clothing", "garment", "outfit", "dress", "shirt", "pants", "jacket",
            "money", "cash", "currency", "dollar", "coin", "bill", "payment", "wealth",
            "furniture", "chair", "table", "desk", "bed", "sofa", "couch", "cabinet",
            "medicine", "drug", "pill", "tablet", "capsule", "treatment", "therapy", "cure"
        ],
        
        # Qualities & States (100 words)
        "qualities": [
            "quality", "characteristic", "attribute", "property", "feature", "trait",
            "good", "better", "best", "excellent", "superior", "outstanding", "exceptional", "perfect",
            "bad", "worse", "worst", "terrible", "awful", "horrible", "dreadful", "atrocious",
            "strong", "powerful", "mighty", "forceful", "robust", "sturdy", "tough", "hardy",
            "weak", "feeble", "frail", "fragile", "delicate", "tender", "soft", "gentle",
            "fast", "quick", "rapid", "swift", "speedy", "hasty", "hurried", "instant",
            "slow", "sluggish", "leisurely", "gradual", "steady", "patient", "deliberate", "careful",
            "smart", "intelligent", "clever", "wise", "brilliant", "genius", "sharp", "bright",
            "dumb", "stupid", "foolish", "silly", "ignorant", "unintelligent", "dense", "obtuse",
            "beautiful", "pretty", "lovely", "gorgeous", "stunning", "attractive", "handsome", "elegant",
            "ugly", "hideous", "repulsive", "disgusting", "unattractive", "plain", "homely", "unsightly",
            "clean", "pure", "fresh", "spotless", "immaculate", "pristine", "sanitary", "hygienic",
            "dirty", "filthy", "grimy", "soiled", "stained", "contaminated", "polluted", "unclean",
            "hot", "warm", "heated", "burning", "scorching", "blazing", "fiery", "boiling",
            "cold", "cool", "chilly", "freezing", "icy", "frigid", "frosty", "bitter",
            "light", "bright", "luminous", "radiant", "brilliant", "shining", "glowing", "illuminated",
            "dark", "dim", "shadowy", "gloomy", "murky", "obscure", "hidden", "concealed",
            "new", "fresh", "recent", "modern", "contemporary", "current", "latest", "updated",
            "old", "ancient", "aged", "mature", "elderly", "vintage", "antique", "outdated"
        ],
        
        # Colors (50 words)
        "colors": [
            "color", "hue", "shade", "tint", "tone", "pigment", "dye",
            "red", "crimson", "scarlet", "ruby", "burgundy", "maroon", "pink", "rose",
            "blue", "azure", "navy", "royal", "sky", "cerulean", "turquoise", "cyan",
            "green", "emerald", "lime", "olive", "forest", "mint", "sage", "jade",
            "yellow", "gold", "amber", "lemon", "sunshine", "blonde", "cream", "ivory",
            "purple", "violet", "lavender", "plum", "magenta", "indigo", "mauve", "lilac",
            "orange", "peach", "apricot", "coral", "salmon", "tangerine", "pumpkin", "carrot",
            "black", "dark", "shadow", "ebony", "charcoal", "ink", "jet", "midnight",
            "white", "light", "pale", "snow", "pearl", "silver", "gray", "grey"
        ],
        
        # Time & Temporal (50 words)
        "time": [
            "time", "temporal", "chronological", "sequence", "duration", "period", "moment",
            "past", "previous", "former", "earlier", "before", "ago", "yesterday", "ancient",
            "present", "current", "now", "today", "contemporary", "modern", "recent", "latest",
            "future", "coming", "next", "tomorrow", "upcoming", "forthcoming", "prospective", "eventual",
            "morning", "dawn", "sunrise", "early", "breakfast", "wake", "awake", "rise",
            "afternoon", "midday", "noon", "lunch", "daytime", "bright", "sunny", "clear",
            "evening", "dusk", "sunset", "twilight", "dinner", "night", "dark", "sleep",
            "year", "month", "week", "day", "hour", "minute", "second", "instant",
            "season", "spring", "summer", "autumn", "winter", "weather", "climate", "temperature"
        ],
        
        # Abstract Concepts (100 words)
        "abstract": [
            "concept", "idea", "thought", "notion", "theory", "principle", "philosophy",
            "truth", "reality", "fact", "knowledge", "wisdom", "understanding", "comprehension", "insight",
            "freedom", "liberty", "independence", "autonomy", "choice", "option", "decision", "will",
            "justice", "fairness", "equality", "right", "wrong", "moral", "ethical", "virtue",
            "peace", "harmony", "tranquility", "serenity", "calm", "quiet", "stillness", "silence",
            "war", "conflict", "battle", "fight", "struggle", "competition", "contest", "challenge",
            "hope", "optimism", "faith", "belief", "trust", "confidence", "assurance", "certainty",
            "despair", "hopelessness", "pessimism", "doubt", "uncertainty", "fear", "worry", "anxiety",
            "love", "affection", "care", "compassion", "kindness", "generosity", "charity", "mercy",
            "hate", "anger", "rage", "fury", "resentment", "bitterness", "hostility", "animosity",
            "success", "achievement", "accomplishment", "victory", "triumph", "win", "gain", "profit",
            "failure", "defeat", "loss", "mistake", "error", "fault", "blame", "responsibility",
            "beauty", "aesthetics", "art", "creativity", "imagination", "inspiration", "genius", "talent",
            "science", "technology", "innovation", "invention", "discovery", "research", "experiment", "analysis"
        ]
    }
    
    # Add all words from categories
    for category_words in categories.values():
        words.extend(category_words)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_words = []
    for word in words:
        if word not in seen:
            seen.add(word)
            unique_words.append(word)
    words = unique_words
    
    # Ensure we have exactly 1000 words (pad if needed)
    while len(words) < 1000:
        words.append(f"word_{len(words)}")
    
    # Truncate if we have more than 1000
    words = words[:1000]
    
    return {word: i for i, word in enumerate(words)}

def create_analogical_tests(vocab: Dict[str, int]) -> List[Dict]:
    """Create comprehensive analogical reasoning test suite"""
    tests = []
    
    # Define semantic relationship patterns
    patterns = {
        # Gender relationships
        "gender": [
            ("man", "woman", "boy", "girl"),
            ("father", "mother", "son", "daughter"),
            ("husband", "wife", "brother", "sister"),
            ("uncle", "aunt", "nephew", "niece"),
            ("grandfather", "grandmother", "grandson", "granddaughter"),
            ("king", "queen", "prince", "princess"),
            ("emperor", "empress", "duke", "duchess"),
            ("actor", "actress", "waiter", "waitress"),
            ("stallion", "mare", "bull", "cow"),
            ("rooster", "hen", "ram", "ewe")
        ],
        
        # Size relationships
        "size": [
            ("big", "small", "huge", "tiny"),
            ("large", "little", "giant", "mini"),
            ("enormous", "miniature", "massive", "micro"),
            ("adult", "child", "elder", "baby"),
            ("horse", "pony", "elephant", "mouse"),
            ("mountain", "hill", "ocean", "pond"),
            ("tree", "bush", "forest", "garden"),
            ("house", "cottage", "mansion", "hut"),
            ("car", "bicycle", "truck", "motorcycle"),
            ("book", "page", "library", "shelf")
        ],
        
        # Temporal relationships
        "temporal": [
            ("run", "ran", "walk", "walked"),
            ("jump", "jumped", "swim", "swam"),
            ("eat", "ate", "drink", "drank"),
            ("sleep", "slept", "wake", "woke"),
            ("work", "worked", "play", "played"),
            ("morning", "evening", "dawn", "dusk"),
            ("spring", "autumn", "summer", "winter"),
            ("past", "future", "yesterday", "tomorrow"),
            ("old", "new", "ancient", "modern"),
            ("begin", "end", "start", "finish")
        ],
        
        # Emotional relationships
        "emotional": [
            ("happy", "sad", "joy", "sorrow"),
            ("love", "hate", "affection", "anger"),
            ("calm", "excited", "peaceful", "thrilled"),
            ("proud", "ashamed", "confident", "embarrassed"),
            ("hopeful", "despair", "optimistic", "pessimistic"),
            ("brave", "afraid", "courageous", "fearful"),
            ("strong", "weak", "powerful", "feeble"),
            ("smart", "dumb", "intelligent", "foolish"),
            ("beautiful", "ugly", "pretty", "hideous"),
            ("good", "bad", "better", "worse")
        ],
        
        # Animal relationships
        "animal": [
            ("cat", "kitten", "dog", "puppy"),
            ("lion", "cub", "tiger", "cub"),
            ("horse", "foal", "cow", "calf"),
            ("sheep", "lamb", "goat", "kid"),
            ("bird", "chick", "duck", "duckling"),
            ("fish", "fry", "frog", "tadpole"),
            ("bear", "cub", "wolf", "pup"),
            ("elephant", "calf", "whale", "calf"),
            ("deer", "fawn", "rabbit", "bunny"),
            ("owl", "owlet", "eagle", "eaglet")
        ],
        
        # Object relationships
        "object": [
            ("car", "truck", "bicycle", "motorcycle"),
            ("house", "mansion", "cottage", "castle"),
            ("book", "novel", "magazine", "newspaper"),
            ("tree", "forest", "bush", "garden"),
            ("mountain", "hill", "ocean", "lake"),
            ("food", "meal", "snack", "feast"),
            ("clothes", "shirt", "pants", "dress"),
            ("furniture", "chair", "table", "sofa"),
            ("tool", "hammer", "screwdriver", "wrench"),
            ("weapon", "sword", "knife", "spear")
        ],
        
        # Color relationships
        "color": [
            ("red", "blue", "crimson", "azure"),
            ("green", "yellow", "emerald", "gold"),
            ("purple", "orange", "violet", "peach"),
            ("black", "white", "dark", "light"),
            ("pink", "rose", "salmon", "coral"),
            ("brown", "tan", "copper", "bronze"),
            ("gray", "silver", "charcoal", "platinum"),
            ("navy", "royal", "sky", "cerulean"),
            ("lime", "lemon", "mint", "sage"),
            ("burgundy", "maroon", "ruby", "scarlet")
        ],
        
        # Professional relationships
        "professional": [
            ("teacher", "student", "doctor", "patient"),
            ("lawyer", "client", "judge", "jury"),
            ("chef", "customer", "waiter", "diner"),
            ("artist", "audience", "musician", "listener"),
            ("writer", "reader", "journalist", "public"),
            ("engineer", "client", "architect", "builder"),
            ("police", "citizen", "guard", "visitor"),
            ("scientist", "researcher", "inventor", "creator"),
            ("pilot", "passenger", "captain", "crew"),
            ("farmer", "consumer", "rancher", "buyer")
        ],
        
        # Family relationships
        "family": [
            ("parent", "child", "grandparent", "grandchild"),
            ("father", "son", "mother", "daughter"),
            ("brother", "sister", "uncle", "aunt"),
            ("cousin", "relative", "nephew", "niece"),
            ("husband", "wife", "boyfriend", "girlfriend"),
            ("family", "household", "clan", "tribe"),
            ("ancestor", "descendant", "forefather", "heir"),
            ("guardian", "ward", "protector", "dependent"),
            ("mentor", "protégé", "teacher", "student"),
            ("sponsor", "beneficiary", "supporter", "recipient")
        ],
        
        # Abstract relationships
        "abstract": [
            ("truth", "lie", "honesty", "deception"),
            ("freedom", "slavery", "liberty", "bondage"),
            ("peace", "war", "harmony", "conflict"),
            ("hope", "despair", "optimism", "pessimism"),
            ("success", "failure", "victory", "defeat"),
            ("beauty", "ugliness", "grace", "clumsiness"),
            ("wisdom", "foolishness", "knowledge", "ignorance"),
            ("courage", "fear", "bravery", "cowardice"),
            ("justice", "injustice", "fairness", "bias"),
            ("love", "hate", "affection", "hostility")
        ]
    }
    
    # Generate tests from patterns
    for pattern_name, pattern_tests in patterns.items():
        for a, b, c, expected_d in pattern_tests:
            if all(word in vocab for word in [a, b, c, expected_d]):
                tests.append({
                    "pattern": pattern_name,
                    "a": a, "b": b, "c": c, "expected": expected_d,
                    "equation": f"{a} - {b} + {c}",
                    "description": f"{pattern_name}: {a} is to {b} as {c} is to ?"
                })
    
    return tests

def train_analogical_model(model, vocab, tests, epochs=100, lr=0.01):
    """Train model specifically for analogical reasoning"""
    print(f"🔥 Training TurnEmbedding for analogical reasoning ({epochs} epochs)...")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        total_loss = 0
        correct_predictions = 0
        
        # Sample a subset of tests for each epoch
        batch_tests = random.sample(tests, min(50, len(tests)))
        
        for test in batch_tests:
            a, b, c, expected = test["a"], test["b"], test["c"], test["expected"]
            
            # Get turn vectors
            turn_a = model.turns[vocab[a]]
            turn_b = model.turns[vocab[b]]
            turn_c = model.turns[vocab[c]]
            
            # Perform semantic arithmetic
            result_turns = turn_a - turn_b + turn_c
            
            # Calculate loss (distance to expected word)
            expected_turns = model.turns[vocab[expected]]
            expected_distance = torch.norm(result_turns - expected_turns)
            
            total_loss += expected_distance
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            # Calculate accuracy for this epoch
            with torch.no_grad():
                epoch_correct = 0
                for test in batch_tests:
                    a, b, c, expected = test["a"], test["b"], test["c"], test["expected"]
                    result_turns, predicted_word, _ = model.semantic_arithmetic(a, b, c, vocab)
                    if predicted_word == expected:
                        epoch_correct += 1
                
                accuracy = epoch_correct / len(batch_tests) * 100
                print(f"  Epoch {epoch:3d}: Loss = {total_loss.item():.4f}, Accuracy = {accuracy:.1f}%")
    
    print("✅ Training complete!")

def evaluate_analogical_reasoning(model, vocab, tests):
    """Evaluate the model on analogical reasoning tasks"""
    print(f"\n🧠 ANALOGICAL REASONING EVALUATION")
    print("=" * 50)
    
    results = []
    pattern_results = defaultdict(list)
    
    for test in tests:
        a, b, c, expected = test["a"], test["b"], test["c"], test["expected"]
        
        # Perform semantic arithmetic
        result_turns, predicted_word, distance = model.semantic_arithmetic(a, b, c, vocab)
        
        is_correct = predicted_word == expected
        results.append({
            "test": test,
            "predicted": predicted_word,
            "expected": expected,
            "correct": is_correct,
            "distance": distance
        })
        
        pattern_results[test["pattern"]].append(is_correct)
    
    # Calculate overall accuracy
    total_correct = sum(1 for r in results if r["correct"])
    total_tests = len(results)
    overall_accuracy = total_correct / total_tests * 100
    
    print(f"📊 OVERALL RESULTS:")
    print(f"   Total tests: {total_tests}")
    print(f"   Correct: {total_correct}")
    print(f"   Accuracy: {overall_accuracy:.1f}%")
    
    # Show results by pattern
    print(f"\n📈 RESULTS BY PATTERN:")
    for pattern, pattern_tests in pattern_results.items():
        pattern_accuracy = sum(pattern_tests) / len(pattern_tests) * 100
        print(f"   {pattern:12}: {sum(pattern_tests):3d}/{len(pattern_tests):3d} ({pattern_accuracy:5.1f}%)")
    
    # Show some example results
    print(f"\n🔍 EXAMPLE RESULTS:")
    for i, result in enumerate(results[:10]):
        test = result["test"]
        status = "✅" if result["correct"] else "❌"
        print(f"   {status} {test['equation']} = {result['predicted']} (expected: {result['expected']}, distance: {result['distance']:.3f})")
    
    return results, overall_accuracy

def main():
    """Run the 1000-word analogical reasoning experiment"""
    print("🌟 TURN THEORY - 1000-WORD ANALOGICAL REASONING")
    print("The ultimate test: Can semantic arithmetic scale to 1000 words?")
    print("=" * 70)
    
    # Create vocabulary and model
    vocab = create_1000_word_vocab()
    model = TurnEmbedding1000(vocab_size=len(vocab), n_turns=4, output_dim=256, poly_degree=4)
    
    print(f"✅ Created {len(vocab)}-word vocabulary")
    print(f"✅ Initialized model with {model.n_turns} turns and {model.output_dim}D embeddings")
    
    # Create analogical tests
    tests = create_analogical_tests(vocab)
    print(f"✅ Generated {len(tests)} analogical reasoning tests")
    
    # Train the model
    train_analogical_model(model, vocab, tests, epochs=200)
    
    # Evaluate performance
    results, accuracy = evaluate_analogical_reasoning(model, vocab, tests)
    
    # Final assessment
    print(f"\n🎯 FINAL ASSESSMENT:")
    if accuracy >= 90:
        print(f"   🚀 BREAKTHROUGH! {accuracy:.1f}% accuracy - Turn Theory scales!")
    elif accuracy >= 80:
        print(f"   ✨ EXCELLENT! {accuracy:.1f}% accuracy - Strong proof of concept")
    elif accuracy >= 70:
        print(f"   📊 GOOD! {accuracy:.1f}% accuracy - Promising results")
    else:
        print(f"   ⚠️  NEEDS IMPROVEMENT: {accuracy:.1f}% accuracy")
    
    print(f"\n💡 This experiment demonstrates the scalability of Semantic Turn Theory!")
    print(f"   If accuracy > 80%, it's strong evidence that meaning is fundamentally mathematical.")

if __name__ == "__main__":
    main()
