import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load & clean
df = pd.read_csv("emails.csv", encoding="latin-1")
df = df.rename(columns={"Message_body": "text", "Label": "label"})
df = df.dropna(subset=["text", "label"])

X = df["text"]
y = df["label"]

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Vectorize — character n-grams instead of words
vectorizer = TfidfVectorizer(
    analyzer="char_wb",    # character-level instead of word-level
    ngram_range=(3, 5),    # sequences of 3 to 5 characters
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# 4. Train — lower alpha for less smoothing
model = MultinomialNB(alpha=0.1)
model.fit(X_train_vec, y_train)

# 5. Evaluate
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print()
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import joblib

joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")