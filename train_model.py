import os
import re
import pickle
import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords
nltk.download('stopwords')

# -----------------------------
# LOAD DATASET
# -----------------------------

columns = [
    'target',
    'id',
    'date',
    'flag',
    'user',
    'text'
]

print("Loading dataset...")

df = pd.read_csv(
    'data/sentiment.csv',
    encoding='latin-1',
    names=columns
)

# Keep only required columns
df = df[['target', 'text']]

# -----------------------------
# LABEL CONVERSION
# 0 = Negative
# 4 = Positive
# -----------------------------

df = df[df['target'].isin([0, 4])]

df['target'] = df['target'].map({
    0: 0,
    4: 1
})

print(df['target'].value_counts())

# -----------------------------
# TEXT CLEANING
# -----------------------------

stemmer = PorterStemmer()

stop_words = set(stopwords.words('english'))

def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtags
    text = re.sub(r"#", "", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenization
    words = text.split()

    # Remove stopwords + stemming
    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

print("Cleaning text...")

df['clean_text'] = df['text'].apply(clean_text)

# -----------------------------
# FEATURES & LABELS
# -----------------------------

X = df['clean_text']

y = df['target']

# -----------------------------
# TF-IDF VECTORIZATION
# -----------------------------

print("Creating TF-IDF vectors...")

vectorizer = TfidfVectorizer(max_features=10000)

X = vectorizer.fit_transform(X)

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# MODEL TRAINING
# -----------------------------

print("Training Logistic Regression model...")

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# -----------------------------
# MODEL EVALUATION
# -----------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# -----------------------------
# SAVE MODEL
# -----------------------------

os.makedirs('models', exist_ok=True)

with open('models/logistic_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('models/tfidf_model.pkl', 'wb') as vector_file:
    pickle.dump(vectorizer, vector_file)

print("\nModel Saved Successfully!")