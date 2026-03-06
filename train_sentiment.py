# -----------------------------
# train_sentiment.py
# -----------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib  # to save model and vectorizer

# ---------- STEP 1: Load Data ----------
print("Loading training and testing data...")
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(f"Training rows: {len(train_df)}, Testing rows: {len(test_df)}")

# ---------- STEP 2: TF-IDF VECTORIZATION ----------
print("Converting text to numerical features using TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000,  # max 5000 words
                             ngram_range=(1,2),  # unigrams + bigrams
                             stop_words='english')

X_train = vectorizer.fit_transform(train_df['text'])
y_train = train_df['sentiment']

X_test = vectorizer.transform(test_df['text'])
y_test = test_df['sentiment']

# ---------- STEP 3: TRAIN MODEL ----------
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=500)  # increase max_iter if needed
model.fit(X_train, y_train)

# ---------- STEP 4: EVALUATE MODEL ----------
print("Evaluating model on test data...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy on test data: {accuracy*100:.2f}%\n")
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# ---------- STEP 5: SAVE MODEL AND VECTORIZER ----------
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("\n✅ Model and vectorizer saved as 'sentiment_model.pkl' and 'tfidf_vectorizer.pkl'")

# ---------- STEP 6: TEST PREDICTION ----------
sample_texts = [
    "The product is excellent and I loved it!",
    "Delivery was late and the quality is bad."
]

sample_features = vectorizer.transform(sample_texts)
sample_pred = model.predict(sample_features)

for text, pred in zip(sample_texts, sample_pred):
    print(f"Text: {text}\nPredicted Sentiment: {pred}\n")