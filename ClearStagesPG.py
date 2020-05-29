'''
The purpose of this file is to clear any events such as Movement, LM, etc.
Anything that isn't needed for the classification of REM or NREM.
'''


import pandas as pd

df = pd.read_csv("CSV/01523Manual.csv")

columns = ['Event','Duration','Start Time','End Time']
df2 = pd.DataFrame(columns=columns)

for index,row in df.iterrows():
    if row['Duration'] == 30:
        if row['Event'] != "REM":
            row['Event'] = "NREM"
        df2 = df2.append(row,ignore_index=True)

df2.to_csv(r'ClearedStagesPediatric2.csv')
