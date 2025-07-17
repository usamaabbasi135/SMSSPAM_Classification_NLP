# SMS Spam Classifier with Three NLP Approaches

A comprehensive pipeline for detecting SMS spam messages using three different natural language processing techniques and classifiers.

---

## Overview

This project demonstrates how to build and compare three text‐classification approaches:

- **Bag-of-Words with Gaussian Naive Bayes**
- **TF-IDF with Multinomial Naive Bayes**
- **Average Word2Vec Embeddings with Logistic Regression**

Each approach follows a complete workflow from raw SMS text to evaluated model performance.

---

## Dataset

- A labeled collection of SMS messages, with each message marked as “spam” or “ham” (legitimate).
- The typical format includes:
  - One column for the message text
  - One column for the label
- Any similar SMS classification dataset can be substituted by matching those two columns.

---

## Project Structure

- `SPAM_NLP_PROJECT.ipynb` – Jupyter notebook containing all steps, from data loading through model evaluation.
- `README.md` – This documentation file.

---

## Installation

1. Ensure a Python 3.7+ environment is available.

2. Install the required libraries:
   - pandas
   - numpy
   - nltk
   - gensim
   - scikit-learn
   - matplotlib
   - seaborn

3. Download NLTK resources for stopwords and lemmatization:
   - `stopwords`
   - `wordnet`

---

## Data Preprocessing

The preprocessing pipeline includes:

- Removing all non-alphabetic characters (numbers, punctuation, special symbols)
- Lowercasing the entire message text
- Tokenizing sentences into words
- Removing stopwords (common English words that do not carry much meaning)
- Applying lemmatization (converting words to their base/root form)

**Outputs:**
- A list of cleaned strings (for use in vectorizer approaches like CountVectorizer and TF-IDF)
- A parallel list of tokenized word lists (used for Word2Vec training)

---

## Approach 1: Bag-of-Words + Gaussian Naive Bayes

- Converts each SMS message into binary word-presence vectors using CountVectorizer
- Splits data into training and testing subsets
- Trains a **Gaussian Naive Bayes** classifier on the resulting feature matrix
- Evaluates model performance using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrix

---

## Approach 2: TF-IDF + Multinomial Naive Bayes

- Transforms messages into weighted word vectors using TF-IDF to emphasize informative terms
- Uses consistent train/test splits for reliable comparison
- Trains a **Multinomial Naive Bayes** classifier
- Evaluates performance with similar metrics: accuracy, precision, recall, F1-score, and confusion matrix

---

## Approach 3: Average Word2Vec Embeddings + Logistic Regression

- Trains a **Word2Vec model** on the tokenized word list to learn dense vector representations of words
- Generates a fixed-length vector for each message by averaging its word vectors
- Ensures the number of feature vectors matches the number of labels (handling mismatches)
- Trains a **Logistic Regression** model
- Measures the same evaluation metrics as the other approaches

---

## Results Summary

| Approach                              | Accuracy | Notes                                                |
|---------------------------------------|----------|------------------------------------------------------|
| Bag-of-Words + Gaussian Naive Bayes   | 0.8753   | Quick to train; ignores word importance              |
| TF-IDF + Multinomial Naive Bayes      | 0.8700   | Captures term importance; uses sparse representation |
| Avg-Word2Vec + Logistic Regression    | 0.8689   | Dense semantic embeddings; slightly lower accuracy   |

---

## Future Improvements

- Incorporate **pre-trained word embeddings** like GloVe or FastText for better semantic understanding
- Experiment with **deep learning models** such as LSTM, GRU, or transformers
- Apply **hyperparameter tuning** and **cross-validation** for better model optimization
- Address **class imbalance** with augmentation or resampling techniques
- Add **additional features**, e.g., message length, special character count, or keyword flags

---

## Author

**Usama Abbasi**  
AI Engineer and Data Analyst (Afiniti)  
📍 Islamabad, Pakistan

---

## 📄 License

This project is licensed under the **MIT License**.  
You are free to fork, adapt, and contribute to this repository.
