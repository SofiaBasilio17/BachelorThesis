import json
import pandas as pd

with open('pediatric_edf/01523.scoring.json') as data_file:
    data = json.load(data_file)
    print(type(data))
    counter=0
    for k, v in data.items():
        stages = v

    df = pd.DataFrame(stages, columns = ['label' , 'start', 'onset' , 'duration'])

    df.to_csv(r'01523.csv')
