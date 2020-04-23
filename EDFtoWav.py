import pyedflib
from pyedflib import EdfReader
import numpy as np
from scipy import signal
from scipy.io.wavfile import write
import matplotlib.pyplot as plt


reader = EdfReader("../../Sofia_recordings/Rec1.edf")

labels = reader.getSignalLabels()
a_index = labels.index("Audio")
a_data = reader.readSignal(a_index)
print(a_data)
a_header = reader.getSignalHeader(a_index)
a_sf = a_header.get('sample_rate')



write("audiorec1.wav", a_sf, a_data)


# print("labels:", labels)
# print("Duration: ", reader.file_duration)

# E1_index = labels.index("E1")
# E1Header = reader.getSignalHeader(E1_index)
# print(E1Header.get('sample_rate'))
# E1_data = reader.readSignal(E1_index)
# plot_spectogram(200,E1_data)
