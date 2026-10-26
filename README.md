# NLP Assignment: Sentiment Analysis Using BERT (Mapped to Syllabus Units I–VI)

**Course**: Natural Language Processing (NLP)  
**Topic**: 3-Class Sentiment Classification (Positive, Negative, Neutral) using BERT & Transformers  
**Frameworks**: PyTorch, HuggingFace `transformers`, Scikit-Learn, Pandas, Matplotlib, HTML5/CSS3/JavaScript  
**Live Demo**: [![Deployed on Vercel](https://img.shields.io/badge/Deployed_on-Vercel-black?style=for-the-badge&logo=vercel)](https://bilingual-sentiment-analysis-bert.vercel.app) • [https://bilingual-sentiment-analysis-bert.vercel.app](https://bilingual-sentiment-analysis-bert.vercel.app)

---

## 📌 Executive Summary

This project implements a complete, submission-ready Natural Language Processing (NLP) assignment for **3-Class Text Sentiment Analysis using BERT**. The project is explicitly structured and mapped across all **6 Units** of the academic NLP syllabus. 

It contains both:
1. **Python Engine & Jupyter Notebook (`nlp_sentiment_assignment.ipynb`)**: Executable data pipelines, baseline deep learning sequence models (BiLSTM/GRU), Bahdanau/Luong attention mechanisms, fine-tuned BERT transformer classifier, and LLM rationale explainer.
2. **Interactive Web Dashboard Studio (`web_app/index.html`)**: Real-time sentiment predictor, tokenization & OOV inspector, 2D PCA vector space plot, model accuracy benchmark arena, multi-head attention heatmap, and syllabus alignment viewer.

---

## 🗺️ Syllabus Mapping Matrix

| Unit | Unit Title | Syllabus Topics Covered | Implementation Location |
|:---|:---|:---|:---|
| **Unit I** | Foundations of NLP & Text Processing | Tokenization, Stemming, Lemmatization, Stop-word removal, Normalization, Bag-of-Words, TF-IDF, Out-of-Vocabulary (OOV) handling | [`src/preprocess.py`](file:///c:/Aditya/Sentimental%20Analysis%20using%20bert%20NLP/src/preprocess.py) & Notebook Section 1 |
| **Unit II** | Word Embeddings & Vector Representations | Vector Space Models, GloVe/Word2Vec, Cosine similarity, Analogy relationships, PCA & t-SNE 2D visualization | [`src/embeddings.py`](file:///c:/Aditya/Sentimental%20Analysis%20using%20bert%20NLP/src/embeddings.py) & Notebook Section 2 |
| **Unit III** | Deep Learning Sequence Models | Sequential text data, RNNs, LSTMs, GRUs, BiLSTM, Sequence classification, Accuracy, F1-Score, Confusion Matrix | [`src/baselines.py`](file:///c:/Aditya/Sentimental%20Analysis%20using%20bert%20NLP/src/baselines.py) & Notebook Section 3 |
| **Unit IV** | Seq2Seq & Attention Mechanisms | Encoder-Decoder models, Soft Attention, Alignment mechanisms (Bahdanau Additive & Luong Multiplicative Attention), BLEU/ROUGE | [`src/attention.py`](file:///c:/Aditya/Sentimental%20Analysis%20using%20bert%20NLP/src/attention.py) & Notebook Section 4 |
| **Unit V** | Transformers & Pretrained LMs | Self-attention, Multi-head attention, Positional encoding, Pretrained BERT, Transfer Learning & Fine-tuning BERT | [`src/bert_classifier.py`](file:///c:/Aditya/Sentimental%20Analysis%20using%20bert%20NLP/src/bert_classifier.py) & Notebook Section 5 |
| **Unit VI** | Generative NLP & LLMs | Text decoding strategies (Greedy, Beam, Top-k, Nucleus sampling), Perplexity, Explainability, Hallucination mitigation | [`src/llm_explainer.py`](file:///c:/Aditya/Sentimental%20Analysis%20using%20bert%20NLP/src/llm_explainer.py) & Notebook Section 6 |

---

## 📂 Project Architecture

```
c:\Aditya\Sentimental Analysis using bert NLP\
├── dataset/
│   ├── sentiment_dataset.csv       # Standard 3-Class Sentiment Dataset
│   └── unilyf_sentiment.csv       # 5,400-row Bilingual & Code-Mixed Campus Dataset
├── generate_unilyf_data.py         # Synthetic Bilingual Data Generator (EN, HI, Hinglish)
├── src/
│   ├── __init__.py
│   ├── preprocess.py               # Unit I: Preprocessing, Bilingual Language Detection, TF-IDF
│   ├── embeddings.py               # Unit II: Embeddings, Cosine Similarity, PCA & t-SNE reduction
│   ├── baselines.py                # Unit III: BiLSTM & GRU baselines, evaluation metrics
│   ├── attention.py                # Unit IV: Bahdanau & Luong attention mechanisms
│   ├── bert_classifier.py          # Unit V: Multilingual BERT Classifier & Campus Topic Detection
│   └── llm_explainer.py            # Unit VI: Bilingual LLM rationale generation & explainability
├── web_app/
│   ├── index.html                  # Bilingual Interactive Dashboard UI (EN • HI • Hinglish)
│   ├── styles.css                  # Dark mode glassmorphism styles with Devanagari support
│   └── app.js                      # Real-time multilingual inference, language & topic detection
├── serve.py                        # Multi-Threaded HTTP Local Web Server
├── nlp_sentiment_assignment.ipynb  # Executable Jupyter Notebook mapped to Units I-VI
├── requirements.txt                # Dependency specifications
└── README.md                       # Comprehensive Assignment Documentation
```

---

## 🚀 How to Run the Project

### 1. Run Python Modules & Verification
Install dependencies:
```bash
pip install -r requirements.txt
```

Run test suite across syllabus modules:
```bash
python src/preprocess.py
python src/embeddings.py
python src/baselines.py
python src/attention.py
python src/bert_classifier.py
python src/llm_explainer.py
```

### 2. Launch Jupyter Notebook for Submission
Launch Jupyter Notebook to view or execute the complete assignment step-by-step:
```bash
jupyter notebook nlp_sentiment_assignment.ipynb
```

### 3. Launch Interactive Web Dashboard
- **Live Vercel Production Site**: [https://bilingual-sentiment-analysis-bert.vercel.app](https://bilingual-sentiment-analysis-bert.vercel.app)
- **Local Multi-Threaded Server**:
```bash
python serve.py
```
Then navigate to `http://localhost:8000` in your web browser.

### 4. Generate or Regenerate Bilingual Campus Dataset
Generate synthetic bilingual and code-mixed student feedback across 8 campus topics (Exams, Hostel, Canteen, WiFi, Events, Placements, Fees, Library):
```bash
python generate_unilyf_data.py --per_cell 600 --out dataset/unilyf_sentiment.csv
```
This generates 5,400 balanced samples across English (`en`), Hindi Devanagari (`hi`), and Romanized Hinglish (`hinglish`).

---

## 📊 Empirical Model Benchmark Results

| Model / Architecture | Unit Reference | Accuracy | Macro F1-Score | Key Advantage |
|:---|:---|:---|:---|:---|
| **TF-IDF + Naive Bayes** | Unit I | 66.7% | 0.648 | Fast baseline, zero deep learning overhead |
| **GloVe + LSTM** | Unit III | 76.7% | 0.765 | Captures sequential context over time |
| **GloVe + BiLSTM** | Unit III | 83.3% | 0.831 | Processes text in both forward and backward directions |
| **Fine-Tuned BERT** | Unit V | **96.7%** | **0.968** | Contextual self-attention representations & transfer learning |

---

## 🎓 Unit-by-Unit Theoretical Deep Dive

### Unit I: Text Preprocessing & WordPiece OOV Handling
Traditional word tokenization fails when encountering unseen words. BERT uses **WordPiece subword tokenization** to split rare tokens into sub-tokens (e.g. `unbelievably` $\rightarrow$ `un` + `##believ` + `##ably`), preserving vocabulary coverage without out-of-vocabulary loss.

### Unit II: Semantic Vector Space & Cosine Similarity
Static word vectors (Word2Vec/GloVe) represent words as fixed dense vectors. Contextual embeddings (BERT) adjust vector representations dynamically based on surrounding text context. Dimensionality reduction algorithms like **PCA** and **t-SNE** project these high-dimensional spaces into 2D plane coordinates.

### Unit III: Sequence Modeling with BiLSTM
Bidirectional LSTMs preserve sequence dependency by propagating hidden states from left-to-right ($\vec{h}_t$) and right-to-left ($\overleftarrow{h}_t$). Concatenating these vectors allows capturing past and future context simultaneously.

### Unit IV: Soft Attention Alignment Mechanisms
Attention mechanisms compute alignment scores between target query states $s_t$ and encoder outputs $h_i$:
- **Bahdanau Additive Attention**: $\text{score}(h_i, s_t) = v_a^T \tanh(W_a h_i + U_a s_t)$
- **Luong Multiplicative Attention**: $\text{score}(h_i, s_t) = s_t^T W_a h_i$

### Unit V: BERT Transformer Fine-Tuning
BERT (Bidirectional Encoder Representations from Transformers) leverages Multi-Head Self-Attention:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
Transfer learning allows fine-tuning pre-trained weights for specific downstream tasks like 3-class sentiment classification.

### Unit VI: LLM Explainability & Generation Decoding
Explainability ensures transparency in transformer predictions. Decoding strategies dictate how language models generate rationale:
- **Greedy Search**: Selects $\arg\max P(w_t | w_{<t})$.
- **Beam Search**: Keeps top-$B$ candidates.
- **Nucleus (Top-p) Sampling**: Restricts candidate pool to cumulative probability threshold $p$.
