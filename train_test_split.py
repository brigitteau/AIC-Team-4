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