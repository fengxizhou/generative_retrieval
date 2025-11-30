import torch
import numpy as np

# 1. The Prefix Trie
# This stores our valid item "paths" to prevent hallucinations.
class PrefixTrie:
    def __init__(self):
        self.root = {}

    def insert(self, sequence):
        # sequence is a tuple like (12, 45, 9)
        node = self.root
        for token in sequence:
            if token not in node:
                node[token] = {}
            node = node[token]
        # Mark the end of a valid sequence
        node['__END__'] = True

    def get_valid_next_tokens(self, prefix_sequence):
        # Given a prefix like (12, 45), returns valid next tokens e.g., [9]
        node = self.root
        for token in prefix_sequence:
            if token not in node:
                return [] # Prefix doesn't exist
            node = node[token]
        
        # Return all keys that are not the end marker
        return [k for k in node.keys() if k != '__END__']

# 2. Mock Model (The "Transformer")
# In reality, this would be a loaded PyTorch model.
class MockGenerativeModel:
    def predict_next_token_scores(self, user_history, current_generated_sequence):
        # Returns random logits for vocab size 100
        # In a real scenario, this uses the model to get logits for the next step
        return torch.randn(100) 

# 3. The Constrained Beam Search Function
def constrained_beam_search(model, trie, user_history, beam_width=3, max_steps=3):
    # Each beam is a tuple: (score, current_sequence)
    # Start with one empty beam with score 0
    beams = [(0.0, [])] 
    
    for step in range(max_steps):
        candidates = []
        
        for score, seq in beams:
            # A. Get Valid Tokens from Trie
            # If seq is empty, get all valid starting tokens.
            # If seq is [12], get valid tokens that follow 12.
            valid_tokens = trie.get_valid_next_tokens(seq)
            
            if not valid_tokens:
                # Dead end or finished sequence, keep as is
                candidates.append((score, seq))
                continue

            # B. Get Model Predictions
            logits = model.predict_next_token_scores(user_history, seq)
            
            # C. Apply Constraints (Masking)
            # We set logits of INVALID tokens to negative infinity
            mask = torch.ones_like(logits) * float('-inf')
            mask[valid_tokens] = 0
            masked_logits = logits + mask
            
            # Convert to log probabilities
            log_probs = torch.nn.functional.log_softmax(masked_logits, dim=-1)
            
            # D. Expand Beam
            # Get top K tokens for this specific beam path
            top_k_log_probs, top_k_indices = torch.topk(log_probs, k=beam_width)
            
            for log_prob, token_idx in zip(top_k_log_probs, top_k_indices):
                token = token_idx.item()
                new_score = score + log_prob.item()
                new_seq = seq + [token]
                candidates.append((new_score, new_seq))
        
        # E. Prune Beams
        # Sort all candidates by score and keep top 'beam_width'
        ordered = sorted(candidates, key=lambda x: x[0], reverse=True)
        beams = ordered[:beam_width]
        
    return beams

# --- Usage Example ---

# A. Build the Catalog (Trie)
catalog_items = [
    (10, 2, 5),  # Matrix
    (10, 2, 9),  # Inception
    (10, 8, 1),  # Interstellar
    (12, 45, 9)  # Nike Shoes
]

trie = PrefixTrie()
for item in catalog_items:
    trie.insert(item)

# B. Run Inference
model = MockGenerativeModel()
user_history = [10, 2, 5] # User watched Matrix

# "Generate the next item ID!"
final_beams = constrained_beam_search(model, trie, user_history, beam_width=2, max_steps=3)

print("Top Generated Semantic IDs:")
for score, seq in final_beams:
    print(f"Sequence: {seq} | Log Probability: {score:.2f}")