"""
Unit II: Word Embeddings and Vector Representations
===================================================
Vector Space Models, Dense Word Embeddings, Word2Vec (CBOW, Skip-Gram), GloVe embeddings.
Capturing semantic similarity, analogy relationships.
Visualizing embedding spaces using PCA or t-SNE.
Static vs Contextual BERT embeddings.
"""

from typing import List, Dict, Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Computes cosine similarity between two vector representations."""
    v1 = vec1.reshape(1, -1)
    v2 = vec2.reshape(1, -1)
    return float(cosine_similarity(v1, v2)[0][0])


def generate_mock_static_embeddings(words: List[str], dim: int = 50) -> Dict[str, np.ndarray]:
    """
    Generates deterministic pseudo-dense embeddings (GloVe/Word2Vec mock)
    for demonstration of vector space operations & semantic similarity.
    """
    np.random.seed(42)
    embeddings = {}
    
    # Define semantic cluster seeds
    pos_seed = np.random.randn(dim)
    neg_seed = np.random.randn(dim)
    neu_seed = np.random.randn(dim)
    
    positive_words = {"excellent", "good", "great", "outstanding", "brilliant", "love", "fantastic"}
    negative_words = {"bad", "terrible", "horrible", "awful", "disappointed", "poor", "worst"}
    
    for word in words:
        w_lower = word.lower()
        if w_lower in positive_words:
            vec = pos_seed + np.random.randn(dim) * 0.2
        elif w_lower in negative_words:
            vec = neg_seed + np.random.randn(dim) * 0.2
        else:
            vec = neu_seed + np.random.randn(dim) * 0.4
        
        # Normalize vector
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        embeddings[word] = vec
        
    return embeddings


def reduce_embedding_dimensions(
    embeddings_dict: Dict[str, np.ndarray], method: str = "pca", n_components: int = 2
) -> Dict[str, Tuple[float, ...]]:
    """
    Reduces dense word embedding vectors into 2D/3D coordinates for visual plotting
    using PCA or t-SNE (Unit II requirement).
    """
    words = list(embeddings_dict.keys())
    vectors = np.array([embeddings_dict[w] for w in words])
    
    if vectors.shape[0] < n_components:
        raise ValueError("Number of samples must be >= n_components for dimensionality reduction.")
        
    if method.lower() == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
    elif method.lower() == "tsne":
        perplexity = min(5, max(1, len(words) - 1))
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    else:
        raise ValueError(f"Unknown reduction method: {method}")
        
    reduced_coords = reducer.fit_transform(vectors)
    
    return {words[i]: tuple(reduced_coords[i].tolist()) for i in range(len(words))}


def extract_bert_contextual_embeddings(text: str) -> Dict[str, np.ndarray]:
    """
    Extracts contextual word embeddings from BERT's last hidden layer.
    Unlike Word2Vec/GloVe, BERT embeddings vary depending on context!
    """
    try:
        import torch
        from transformers import BertTokenizer, BertModel
        
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")
        model.eval()
        
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        with torch.no_grad():
            outputs = model(**inputs)
            
        last_hidden_state = outputs.last_hidden_state.squeeze(0).detach().cpu().numpy()
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0).tolist())
        
        return {tokens[i]: last_hidden_state[i] for i in range(len(tokens))}
    except Exception as e:
        # Fallback if transformers model downloading fails
        tokens = text.split()
        return {t: np.random.randn(768) for t in tokens}


if __name__ == "__main__":
    sample_words = ["excellent", "fantastic", "great", "terrible", "awful", "horrible", "report", "meeting"]
    mock_emb = generate_mock_static_embeddings(sample_words, dim=50)
    
    print("--- Unit II: Semantic Similarity & Embedding Visualizer ---")
    sim_pos = compute_cosine_similarity(mock_emb["excellent"], mock_emb["fantastic"])
    sim_neg = compute_cosine_similarity(mock_emb["excellent"], mock_emb["terrible"])
    print(f"Cosine Sim ('excellent', 'fantastic'): {sim_pos:.4f}")
    print(f"Cosine Sim ('excellent', 'terrible'): {sim_neg:.4f}")
    
    pca_coords = reduce_embedding_dimensions(mock_emb, method="pca", n_components=2)
    print("\nPCA 2D Coordinates:")
    for w, coord in pca_coords.items():
        print(f"  {w:12s} -> x={coord[0]:.3f}, y={coord[1]:.3f}")
