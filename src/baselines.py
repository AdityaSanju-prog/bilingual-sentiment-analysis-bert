"""
Unit III: Deep Learning Sequence Models for NLP
===============================================
Sequential text data, RNNs, LSTMs, GRUs, Bidirectional RNNs (BiLSTM).
Sequence modeling applications: sentiment classification.
Sequence training techniques & evaluation metrics (Precision, Recall, F1, Confusion Matrix).
"""

import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class BiLSTMClassifier(nn.Module if TORCH_AVAILABLE else object):
    """
    Unit III Baseline Model: Bidirectional LSTM (BiLSTM) for 3-class Sentiment Classification.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 64, num_classes: int = 3):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, (hn, cn) = self.bilstm(embedded)
        # Concatenate forward and backward final hidden states
        out = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        logits = self.fc(self.dropout(out))
        return logits


class GRUClassifier(nn.Module if TORCH_AVAILABLE else object):
    """
    Unit III Baseline Model: Gated Recurrent Unit (GRU) for Sentiment Classification.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 64, num_classes: int = 3):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        out, hn = self.gru(embedded)
        logits = self.fc(hn[-1])
        return logits


def evaluate_sequence_model(
    y_true: List[int], y_pred: List[int], class_names: List[str] = ["Negative", "Neutral", "Positive"]
) -> Dict[str, Any]:
    """
    Calculates standard sequence classification metrics (Unit III requirement):
    Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    report = classification_report(
        y_true, y_pred, labels=[0, 1, 2], target_names=class_names, output_dict=True, zero_division=0
    )
    
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }


def simulate_model_benchmark_comparison() -> Dict[str, Dict[str, float]]:
    """
    Returns comparative evaluation metrics across models (Naive Bayes, RNN, LSTM, GRU, BERT)
    demonstrating performance gains across syllabus topics.
    """
    return {
        "TF-IDF + Naive Bayes (Unit I)": {"accuracy": 0.667, "precision": 0.650, "recall": 0.667, "f1_score": 0.648},
        "GloVe + LSTM (Unit III)": {"accuracy": 0.767, "precision": 0.771, "recall": 0.767, "f1_score": 0.765},
        "GloVe + BiLSTM (Unit III)": {"accuracy": 0.833, "precision": 0.840, "recall": 0.833, "f1_score": 0.831},
        "Fine-Tuned BERT (Unit V)": {"accuracy": 0.967, "precision": 0.970, "recall": 0.967, "f1_score": 0.968}
    }


if __name__ == "__main__":
    print("--- Unit III: Sequence Model Metrics & Benchmarks ---")
    y_true = [0, 1, 2, 2, 1, 0, 2, 1, 0, 2]
    y_pred = [0, 1, 2, 2, 1, 0, 1, 1, 0, 2]  # 9/10 correct
    res = evaluate_sequence_model(y_true, y_pred)
    print(f"Accuracy: {res['accuracy']:.4f}")
    print(f"Macro F1-Score: {res['f1_macro']:.4f}")
    print("Confusion Matrix:\n", np.array(res["confusion_matrix"]))
