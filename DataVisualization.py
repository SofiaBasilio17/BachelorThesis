import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sleep_data = pd.read_csv("CSV_Sofia/sofiaReadySVM2.csv")
sleep_data = sleep_data.drop(['index'], axis=1)


sns.set_style("whitegrid")
sns.FacetGrid(sleep_data, hue="Class", size=4) \
   .map(plt.scatter, "RIP Phase", "C3-M2") \
   .add_legend()
plt.show()
