import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('emails.csv', encoding='latin-1')

# Add length column
df['length'] = df['Message_body'].str.len()

# Filter spam only
spam = df[df['Label'] == 'Spam']

# Plot
plt.figure(figsize=(10, 5))
spam['length'].hist(bins=50, color='red', edgecolor='black', alpha=0.7)
plt.xlabel('Email Length (characters)')
plt.ylabel('Count')
plt.title('Distribution of Spam Email Lengths')
plt.tight_layout()
plt.show()