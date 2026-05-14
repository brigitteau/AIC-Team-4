import pandas as pd
import re
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("emails.csv", encoding="latin-1")
df = df.rename(columns={'Message_body': 'text', 'Label': 'label'})
df = df.dropna(subset=['text', 'label'])

# Drop the row index column — it's not a feature
X = df["text"]
y = df["label"]  # "Spam" or "Non-Spam"

# Split — stratified because dataset is imbalanced (61% Spam / 39% Non-Spam)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 100 train, 25 test
    random_state=42,
    stratify=y          # preserves 61/39 ratio in both splits
)

print(f"Train: {len(X_train)} samples")
print(f"Test:  {len(X_test)} samples")
print(f"\nTrain class distribution:\n{y_train.value_counts()}")
print(f"\nTest class distribution:\n{y_test.value_counts()}")


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Convert text to numbers
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

# Results
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, pos_label="Spam")
cm = confusion_matrix(y_test, y_pred, labels=["Non-Spam", "Spam"])

print("=" * 50)
print("       RANDOM FOREST - SPAM DETECTION")
print("=" * 50)
print(f"\nAccuracy:  {acc:.4f}  ({acc*100:.1f}%)")
print(f"F1 Score:  {f1:.4f}")
print(f"\n--- CONFUSION MATRIX ---")
print(f"                Predicted")
print(f"                Non-Spam  Spam")
print(f"Actual Non-Spam    {cm[0][0]}        {cm[0][1]}")
print(f"Actual Spam        {cm[1][0]}        {cm[1][1]}")
print(f"\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred, target_names=['Non-Spam', 'Spam']))