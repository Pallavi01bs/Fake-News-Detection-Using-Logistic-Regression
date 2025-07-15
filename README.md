# 🧠 Fake News Detection Using Logistic Regression

A Natural Language Processing (NLP) and Machine Learning project designed to classify news articles as **real** or **fake** using Logistic Regression and TF-IDF vectorization.

## 📌 Overview

With the rapid rise of digital content and social media platforms, misinformation has become a pressing global issue. This project presents an end-to-end machine learning pipeline that detects fake news articles. It utilizes a supervised learning algorithm — **Logistic Regression** — paired with text preprocessing and feature extraction using **TF-IDF** to identify whether a piece of news is credible or not.

## 🗂️ Dataset

- **File**: `fake_news_dataset_expanded.csv`
- **Features**:
  - `text`: News article content
  - `label`: Target label (`0` for Real, `1` for Fake)

## 🔧 Technologies & Tools

- **Language**: Python 3.7+
- **Libraries**:
  - `pandas`, `numpy` – Data manipulation
  - `scikit-learn` – ML algorithms and evaluation
  - `nltk` – Text preprocessing (stopwords, stemming)
  - `pickle` – Model serialization

## ⚙️ Workflow

1. **Text Preprocessing**
   - Convert to lowercase
   - Remove punctuation, digits, and special characters
   - Remove stopwords
   - Apply stemming

2. **Feature Extraction**
   - Use **TF-IDF Vectorizer** to convert processed text into numerical feature vectors.

3. **Model Training**
   - Split data into training and testing sets (80/20 split)
   - Train a **Logistic Regression** model on the TF-IDF features.

4. **Model Evaluation**
   - Evaluate using **Accuracy**, **Confusion Matrix**, and **Classification Report** (Precision, Recall, F1-Score)

5. **Model Persistence**
   - Save the trained model and vectorizer as `.pkl` files for reuse.

## 📊 Results

- **Accuracy Achieved**: 97.21%
- **Evaluation Metrics**:
  - High precision and recall for both real and fake news classes.
  - Confusion matrix confirms low misclassification rate.

## 🛠 Usage Instructions

### 🔹 Setup

1. Ensure the dataset `fake_news_dataset_expanded.csv` is available.
2. Install required dependencies:

```bash
pip install pandas numpy scikit-learn nltk
```

Run the script:

```bash
python fake_news_detection.py
```
#### 🔹 Example Prediction
```python
example = "Pope Francis used his annual Christmas Day message..."
print(predict_news(example))  # Output: Fake News or Real News
```

## 📁 Project Structure
```bash
📦 Fake-News-Detection/
├── fake_news_detection.py         # Main source code
├── fake_news_dataset_expanded.csv # Dataset
├── logistic_model.pkl             # Trained ML model
├── tfidf_vectorizer.pkl           # Saved TF-IDF vectorizer
└── README.md                      # Project documentation
```

## ✅ Key Features
 - High accuracy fake news detection
 - Clean and reusable ML pipeline
 - Portable and customizable
 - Fully documented with clear code structure

## 🧠 Concepts Used
 - Natural Language Processing (NLP)
 - Supervised Learning
 - Logistic Regression
 - TF-IDF Vectorization
 - Model Evaluation (Accuracy, F1, Confusion Matrix)
 - Serialization using Pickle

## 📚 References
 - Scikit-learn Documentation
 - NLTK Documentation
 - TF-IDF Explained
 - Logistic Regression in ML
 - Project Literature
