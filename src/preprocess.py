"""
Unit I: Foundations of NLP & Text Processing
=============================================
Origin of NLP, linguistic essentials, morphology, syntax, semantics.
Tokenization, stemming, lemmatization, stop-word removal, punctuation handling,
handling out-of-vocabulary (OOV) words, normalization, Bag-of-Words (BoW), n-grams, TF-IDF.
"""

import re
import string
from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Standard English stopwords fallback list
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself",
    "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most",
    "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our",
    "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "whom", "why", "with", "you", "your", "yours"
}

# Hindi Devanagari stopwords
HINDI_STOPWORDS = {
    "का", "के", "की", "को", "में", "से", "पर", "और", "यह", "वह", "इस", "उस", "ने",
    "तो", "ही", "भी", "या", "था", "थी", "थे", "है", "हैं", "हूँ", "हो", "हुआ",
    "हुई", "हुए", "कर", "किया", "किए", "गया", "गई", "गए", "दिया", "लिए", "अपने",
    "अपनी", "अपना", "सब", "कल", "आज", "तक", "रहे", "रही", "रहा", "सकते", "सकता"
}

# Hinglish (Romanized Hindi) stopwords & structural markers
HINGLISH_STOPWORDS = {
    "hai", "hain", "hoon", "ho", "ka", "ke", "ki", "ko", "mein", "se", "par",
    "pe", "aur", "yeh", "woh", "is", "us", "ne", "toh", "hi", "bhi", "ya",
    "tha", "thi", "the", "kar", "karo", "diya", "gaya", "gayi", "gaye", "hua",
    "hue", "raha", "rahi", "rahe", "hoga", "hogi", "bhai", "yaar", "apne", "apna"
}

# Common Hinglish lexical markers for heuristic language detection
HINGLISH_MARKERS = {
    "hai", "hain", "nahi", "nhi", "aur", "bhai", "yaar", "kya", "tha", "thi", "the",
    "mein", "ke", "ki", "ko", "se", "ho", "gaya", "gayi", "gaye", "bahut", "bohot",
    "kaafi", "par", "pe", "chala", "raha", "rahi", "rahe", "karo", "kar", "hoga",
    "hogi", "kamaal", "badhiya", "shuru", "milenge", "aaya", "aaye", "saaf", "theek",
    "ekdum", "mast", "sab", "phir", "pehle", "bataya", "aagaya", "aagaye", "dekh",
    "lekin", "magar", "kyun", "kaisa", "kaisi", "waale", "waali", "karna"
}


def detect_language(text: str) -> str:
    """
    Bilingual Feature: Detects language of input text:
    - 'hi': Hindi in Devanagari script (Unicode range \u0900-\u097F)
    - 'hinglish': Hindi in Roman script (Code-mixed English-Hindi)
    - 'en': Standard English
    """
    if not text:
        return "en"
        
    # Check for Devanagari characters
    devanagari_count = len(re.findall(r"[\u0900-\u097F]", text))
    if devanagari_count > 0:
        return "hi"
        
    # Check for Hinglish markers in Roman text
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return "en"
        
    hinglish_hits = sum(1 for w in words if w in HINGLISH_MARKERS)
    if hinglish_hits >= 2 or (len(words) <= 5 and hinglish_hits >= 1):
        return "hinglish"
        
    return "en"


def clean_text(text: str, remove_punct: bool = True, lowercase: bool = True) -> str:
    """
    Normalizes text by lowercasing and optionally removing punctuation.
    Preserves Devanagari script characters alongside Latin alphabet.
    """
    if lowercase:
        text = text.lower()
    if remove_punct:
        # Standard punctuation plus Devanagari danda and double danda
        punct_pattern = r"[" + re.escape(string.punctuation) + r"।॥]"
        text = re.sub(punct_pattern, " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_basic(text: str) -> List[str]:
    """
    Multilingual word tokenization supporting English, Hindi (Devanagari), and Hinglish.
    Preserves Devanagari words (\u0900-\u097F) and Latin alphanumeric words.
    """
    cleaned = clean_text(text, remove_punct=True, lowercase=True)
    # Match both Devanagari token sequences and Latin token sequences
    tokens = re.findall(r"[\u0900-\u097F]+|[a-zA-Z0-9']+", cleaned)
    return tokens if tokens else cleaned.split()


def remove_stopwords(tokens: List[str], lang: str = "en") -> List[str]:
    """Filters out stop words based on detected language ('en', 'hi', or 'hinglish')."""
    if lang == "hi":
        return [t for t in tokens if t not in HINDI_STOPWORDS]
    elif lang == "hinglish":
        return [t for t in tokens if t not in HINGLISH_STOPWORDS and t not in STOPWORDS]
    return [token for token in tokens if token not in STOPWORDS]


def simple_stemmer(word: str) -> str:
    """Demonstration rule-based stemmer (suffix removal for English)."""
    suffixes = ["ing", "ly", "ed", "es", "s", "ment", "ness", "ful"]
    word_lower = word.lower()
    for suffix in suffixes:
        if word_lower.endswith(suffix) and len(word_lower) - len(suffix) > 2:
            return word_lower[:-len(suffix)]
    return word_lower


def compute_bow_and_tfidf(corpus: List[str]) -> Dict[str, np.ndarray]:
    """
    Computes Bag-of-Words (CountVectorizer) and TF-IDF matrices for a given corpus.
    Demonstrates N-gram extraction (unigrams + bigrams).
    """
    cleaned_corpus = [clean_text(doc) for doc in corpus]
    
    # 1. Bag-of-Words (Unigrams + Bigrams)
    bow_vectorizer = CountVectorizer(ngram_range=(1, 2))
    bow_matrix = bow_vectorizer.fit_transform(cleaned_corpus)
    
    # 2. TF-IDF Matrix
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_corpus)
    
    return {
        "bow_matrix": bow_matrix.toarray(),
        "bow_feature_names": bow_vectorizer.get_feature_names_out(),
        "tfidf_matrix": tfidf_matrix.toarray(),
        "tfidf_feature_names": tfidf_vectorizer.get_feature_names_out(),
        "tfidf_vectorizer": tfidf_vectorizer
    }


def compare_tokenization_with_bert(sample_text: str):
    """
    Compares traditional Word Tokenization with BERT WordPiece Tokenization.
    Demonstrates handling of Out-Of-Vocabulary (OOV) words like 'unbelievably' or 'antigravity'.
    """
    traditional_tokens = tokenize_basic(sample_text)
    
    try:
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_tokens = tokenizer.tokenize(sample_text)
        bert_ids = tokenizer.encode(sample_text)
    except Exception as e:
        bert_tokens = ["[Error loading HuggingFace BERT Tokenizer]"]
        bert_ids = []

    return {
        "text": sample_text,
        "traditional_tokens": traditional_tokens,
        "bert_wordpiece_tokens": bert_tokens,
        "bert_input_ids": bert_ids
    }


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass
            
    sample_texts = [
        ("en", "The customer service was unbelievably helpful and lightning-fast!"),
        ("hi", "नए हॉस्टल के कमरे साफ और बड़े हैं, बहुत पसंद आए!"),
        ("hinglish", "Exam ki date phir se change ho gayi, sirf ek din pehle bataya, bahut frustrating hai.")
    ]
    print("--- Unit I: Multilingual Preprocessing & Tokenization Demo ---")
    for expected_lang, text in sample_texts:
        detected = detect_language(text)
        tokens = tokenize_basic(text)
        filtered = remove_stopwords(tokens, lang=detected)
        print(f"\n[{detected.upper()}] Text: {text}")
        print(f"Detected Lang: {detected} (Expected: {expected_lang})")
        print("Tokens:", tokens[:6], "...")
        print("After Stop-word Removal:", filtered[:6], "...")


