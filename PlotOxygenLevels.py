import pyedflib
from pyedflib import EdfReader
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

reader = EdfReader("../../Sofia_recordings/Rec1.edf")

labels = reader.getSignalLabels()
#print("labels:", labels)

#### Getting Spo2 levels
spo2_index = labels.index("SpO2")
spo2_data = reader.readSignal(spo2_index)
spo2_data = list(filter(lambda x: x > 1, spo2_data))

plt.hist(spo2_data, facecolor='blue', alpha=0.5)
plt.show()
