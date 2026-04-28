import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('emails.csv', encoding='latin-1')

counts = df['Label'].value_counts()

colors = ['#022851', '#FFBF00']

fig, ax = plt.subplots(figsize=(6, 6))

ax.pie(
    counts.values,
    labels=counts.index,
    colors=colors,
    autopct='%1.1f%%',
    startangle=90
)
ax.set_title('Spam vs Non-Spam Distribution')

plt.tight_layout()
plt.savefig('spam_chart.png', dpi=150)
plt.show()