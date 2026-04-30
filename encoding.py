import pandas as pd

df = pd.read_csv("emails.csv", encoding = "latin-1")

df["Label_encoded"] = df["Label"].map({"Spam": 1, "Not Spam": 0})

print(df[["Label", "Label_encoded"]].head())