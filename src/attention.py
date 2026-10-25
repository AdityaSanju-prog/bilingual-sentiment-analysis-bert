"""
Unit IV: Sequence-to-Sequence Models & Attention Mechanisms
===========================================================
Encoder-Decoder architectures, Attention mechanisms, Soft Attention,
Alignment mechanisms (Bahdanau Additive Attention & Luong Multiplicative Attention).
Extracting alignment scores for sentiment word importance.
"""

from typing import Tuple, List, Dict, Any
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class BahdanauAttention(nn.Module if TORCH_AVAILABLE else object):
    """
    Bahdanau (Additive) Attention Mechanism (Unit IV requirement).
    Score(h_i, s) = v_a^T * tanh(W_a * h_i + U_a * s)
    """
    def __init__(self, hidden_dim: int):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        self.W_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_a = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, encoder_outputs: Any, query: Any) -> Tuple[Any, Any]:
        # encoder_outputs: [batch_size, seq_len, hidden_dim]
        # query: [batch_size, hidden_dim]
        query_expanded = query.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        scores = self.v_a(torch.tanh(self.W_a(encoder_outputs) + self.U_a(query_expanded)))  # [batch_size, seq_len, 1]
        attention_weights = F.softmax(scores, dim=1)  # [batch_size, seq_len, 1]
        context_vector = torch.sum(attention_weights * encoder_outputs, dim=1)  # [batch_size, hidden_dim]
        return context_vector, attention_weights.squeeze(-1)


class LuongAttention(nn.Module if TORCH_AVAILABLE else object):
    """
    Luong (Multiplicative / Dot-Product) Attention Mechanism (Unit IV requirement).
    Score(h_i, s) = s^T * W * h_i
    """
    def __init__(self, hidden_dim: int):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, encoder_outputs: Any, query: Any) -> Tuple[Any, Any]:
        # query: [batch_size, hidden_dim]
        query_proj = self.W(query).unsqueeze(2)  # [batch_size, hidden_dim, 1]
        scores = torch.bmm(encoder_outputs, query_proj).squeeze(2)  # [batch_size, seq_len]
        attention_weights = F.softmax(scores, dim=1)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context_vector, attention_weights


def compute_heuristic_sentiment_attention(tokens: List[str]) -> Dict[str, float]:
    """
    Calculates alignment attention weights for input tokens based on sentiment lexicon weightings.
    Highlights how attention mechanisms automatically assign high weights to key sentiment words.
    """
    sentiment_dictionary = {
        "excellent": 3.0, "fantastic": 3.0, "outstanding": 3.0, "great": 2.5, "love": 2.5, "good": 1.5,
        "terrible": 3.0, "horrible": 3.0, "awful": 3.0, "disappointed": 2.5, "poor": 2.0, "bad": 1.5,
        "fast": 1.5, "slow": 1.5, "broke": 2.0, "delayed": 2.0, "helpful": 2.0, "useless": 2.0
    }
    
    raw_scores = []
    for token in tokens:
        clean_tok = token.lower().strip(".,!?")
        score = sentiment_dictionary.get(clean_tok, 0.2)
        raw_scores.append(score)
        
    exp_scores = np.exp(raw_scores)
    softmax_weights = exp_scores / np.sum(exp_scores)
    
    res = {}
    for i, token in enumerate(tokens):
        weight = float(softmax_weights[i])
        res[token] = max(res.get(token, 0.0), weight)
    return res


if __name__ == "__main__":
    print("--- Unit IV: Attention & Alignment Mechanisms ---")
    tokens = ["The", "movie", "was", "absolutely", "fantastic", "and", "brilliant"]
    attn_map = compute_heuristic_sentiment_attention(tokens)
    print("Attention Weights per Token:")
    for tok, weight in attn_map.items():
        bar = "#" * int(weight * 30)
        print(f"  {tok:12s} : {weight:.4f} {bar}")
