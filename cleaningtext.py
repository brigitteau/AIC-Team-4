import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('emails.csv', encoding='latin-1')
df = df.rename(columns={'Message_body': 'text', 'Label': 'label'})
df = df.dropna(subset=['text', 'label'])

def clean_text(text):
    text = text.lower()                        # lowercase everything
    text = re.sub(r'[^a-z\s]', '', text)       # remove punctuation and numbers
    text = re.sub(r'\s+', ' ', text).strip()   # remove extra spaces
    return text

df['text'] = df['text'].apply(clean_text)

print(df.head())
print(df.shape)
