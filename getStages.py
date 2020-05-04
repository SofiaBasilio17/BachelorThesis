import pandas as pd

df = pd.read_csv("CSV_Sofia/ClearedStages.csv")


columns = ['Event','Duration','Start Time','End Time','Start Epoch','End Epoch']
df2 = pd.DataFrame(columns=columns)

for index, row in df.iterrows():
    if row['Event'] != 'REM':
        row['Event'] = 'NREM'
    df2 = df2.append(row,ignore_index=True)


df2.to_csv(r'sofiaStaged2.csv')
