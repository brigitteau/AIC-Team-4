import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Step 1: Load the data ─────────────────────────────────────
df = pd.read_csv('emails.csv', encoding='latin-1')
df = df.rename(columns={'Message_body': 'text', 'Label': 'label'})
df = df.dropna(subset=['text', 'label'])

# ── Step 2: Clean the text ────────────────────────────────────
def clean_text(text):
    text = text.lower()                        # lowercase everything
    text = re.sub(r'[^a-z\s]', '', text)       # remove punctuation and numbers
    text = re.sub(r'\s+', ' ', text).strip()   # remove extra spaces
    return text

df['text'] = df['text'].apply(clean_text)

# ── Step 3: Remove stop words and vectorize ───────────────────
stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                  'to', 'for', 'of', 'with', 'is', 'it', 'this', 'that',
                  'you', 'your', 'we', 'i', 'my', 'me', 'be', 'are', 'was',
                  'have', 'has', 'not', 'from', 'will', 'can', 'if', 'as'])

vectorizer = TfidfVectorizer(stop_words=list(stop_words), max_features=5000)
X = vectorizer.fit_transform(df['text'])

# ── Step 4: Encode the labels ─────────────────────────────────
encoder = LabelEncoder()
y = encoder.fit_transform(df['label'])   # Spam=1, Non-Spam=0

print("Label mapping:", dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))

# ── Step 5: Split into train and test sets ────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 80% train, 20% test
    random_state=42,    # makes the split the same every time you run it
    stratify=y          # keeps spam/non-spam ratio balanced in both sets
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")
print("Preprocessing done!")


# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ── Step 6: Train the model ───────────────────────────────────
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ── Step 7: Evaluate the model ────────────────────────────────
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# ── Step 8: Save the model ────────────────────────────────────
joblib.dump(model, 'logisticregressionmodel.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(encoder, 'encoder.pkl')

print("Model saved!")

