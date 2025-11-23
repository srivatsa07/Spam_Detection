# Spam Detection – Comparative Analysis

This project compares multiple machine learning and deep learning algorithms for spam email classification using the Enron Spam Dataset from HuggingFace. The goal is to evaluate how different models, feature extraction methods, and preprocessing steps affect spam detection performance.

## Dataset
- **Enron Spam Dataset** from HuggingFace  
- Balanced: ~50.8% spam / 49.2% ham  
- Used fields: `label`, `text` (subject + message)  
- Custom split for training, validation, and testing

## Preprocessing
Experiments conducted **with and without**:
- Lowercasing  
- Removal of punctuation, numbers, special characters  
- Tokenization  
- Stopword removal  
- Lemmatization  

Note: Preprocessing improved ML and BiLSTM performance, minimal effect on ALBERT.

## Feature Extraction
### For Machine Learning Models
- **TF-IDF**  
  - `max_features=12000`, `ngram_range=(1,2)`  
  - Hyperparameters optimized using Optuna  
- **Word2Vec**  
  - `vector_size=300`, `window=7`, `min_count=3`, `sg=1`  

### For Deep Learning Models
- **ALBERT tokenizer** (max length 128)  
- **GloVe embeddings** for BiLSTM (tested 100d/200d/300d → best: **300d**)

## Models
- **SVM**  
  - Tuned using Optuna (C, kernel, degree)  
  - StandardScaler applied to Word2Vec vectors  
- **Random Forest**  
  - Optimized with Optuna (n_estimators, max_depth, min_samples_split)  
- **ALBERT (albert-base-v2)**  
  - Trained 6 epochs with AdamW, LR scheduling, gradient clipping  
- **BiLSTM**  
  - 2-layer BiLSTM, 128 hidden units  
  - Dropout, early stopping, AdamW optimizer  

All deep learning models trained with GPU acceleration.

## Results (Summary)
- **Best Performing Model:**  
  **SVM + TF-IDF**  
  - Accuracy: **99.35%**  
  - F1-Score: **99.36%**

- **Best Deep Learning Model:**  
  **BiLSTM + GloVe (300d)**  
  - F1-Score: **99.06%**  
  - Very high recall (99.80%)

- **ALBERT:** Competitive but slower; minimal benefit from preprocessing  
- **Random Forest:** Fastest to train; moderate performance compared to others
