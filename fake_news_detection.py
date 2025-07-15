import pandas as pd
import numpy as np
import string
import re
import pickle
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords if not already
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("D:/Desktop/Project/py tutorials/fake_news_dataset_expanded.csv")

# Preprocessing function
stemmer = PorterStemmer()
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    filtered = [stemmer.stem(word) for word in tokens if word not in stopwords.words('english')]
    return " ".join(filtered)

# Apply preprocessing
df['content'] = df['text'].apply(preprocess_text)

# Features and labels
X = df['content']
y = df['label']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, stratify=y, random_state=2)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy Score:", acc)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer
with open('logistic_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Prediction function
def predict_news(news):
    news = preprocess_text(news)
    news_vector = vectorizer.transform([news])
    prediction = model.predict(news_vector)
    return "Fake News" if prediction[0] == 1 else "Real News"

# Example prediction
example = "Pope Francis used his annual Christmas Day message..."
print("Prediction for sample input:", predict_news(example))
