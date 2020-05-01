import pandas as pd

df = pd.read_csv("sofiaeventsheet.csv")

stages = ['N1','N2','N3','REM']
columns = ['Event','Duration','Start Time','End Time','Start Epoch','End Epoch']
df2 = pd.DataFrame(columns=columns)

for index, row in df.iterrows():
    if row['Event'] in stages:
        if row['Event'] != 'REM':
            row['Event'] = 'NREM'
        print(row)
        df2 = df2.append(row,ignore_index=True)


df2.to_csv(r'sofiaStaged.csv')
