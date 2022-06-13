import pandas as pd


df = pd.read_csv('./staticFiles/uploads/PREDICT.csv')
df.dropna(inplace=True)
cols = list(df)[1:6]

print(df[-10:])
