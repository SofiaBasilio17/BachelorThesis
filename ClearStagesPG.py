'''
The purpose of this file is to clear any events such as Movement, LM, etc.
Anything that isn't needed for the classification of REM or NREM.
'''


import pandas as pd

df = pd.read_csv("StagesToClear.csv")

columns = ['Event','Duration','Start Time','End Time','Start Epoch','End Epoch']
df2 = pd.DataFrame(columns=columns)

for index, row in df.iterrows():
    if row['Duration'] == 30:
        df2 = df2.append(row,ignore_index=True)

df2.to_csv(r'ClearedStages.csv')
