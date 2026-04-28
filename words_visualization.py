import pandas as pd
import plotly.express as px
from collections import Counter
import re

df = pd.read_csv('emails.csv', encoding='latin-1')
spam = df[df['Label'] == 'Spam']['Message_body']


counts = df['Label'].value_counts().reset_index()
counts.columns = ['Label', 'Count']


# Flatten all words, lowercase, remove punctuation
words = re.findall(r'\b[a-z]{3,}\b', spam.str.lower().str.cat(sep=' '))

#Count and show top 20
common = Counter(words).most_common(20)
for word, count in common:
    print(f"{word}: {count}")

df_words = pd.DataFrame(common, columns=['word', 'count']).sort_values('count')
fig = px.bar(df_words, x='count', y='word', orientation='h', title='Most common spam words')
fig.show()