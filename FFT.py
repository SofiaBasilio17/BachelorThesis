import pyedflib
from pyedflib import EdfReader
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy import signal
from scipy.fftpack import fft

def get_signal_segment(signal, sf, onset=0, duration=10):
    duration = duration * sf
    onset = onset * sf
    return signal[onset:(onset+duration)]

def get_sample_frequency(index):
    header = reader.getSignalHeader(index)
    print("HEADER:",header)
    return header.get('sample_rate')

reader = EdfReader("../../Sofia_recordings/Rec1.edf")

labels = reader.getSignalLabels()
# print("labels:", labels)
# print("Duration: ", reader.file_duration)

# E1_index = labels.index("E1")
# E1Header = reader.getSignalHeader(E1_index)
# print(E1Header.get('sample_rate'))
# E1_data = reader.readSignal(E1_index)

# FTen = E1_data[:2000]
# time = list(range(0,2000))
# FFTften = signal.fftconvolve(FTen,time)

sf = 200
samples1 = []
#### sampling the first 30 seconds
for l in labels:
    tempIndex = labels.index(l)
    print(tempIndex)
    tempFrequency = get_sample_frequency(tempIndex)
    print('sampling frequency of ', l,' is ', tempFrequency)
    tempData = reader.readSignal(tempIndex)
    samples1.append(get_signal_segment(tempData, tempFrequency, 0, 30))
    print(samples1[0])
    


#### 10 second samples
# sample1 = get_signal_segment(E1_data, sf, 0, 1000)
# sample2 = get_signal_segment(E1_data, sf, 10, 10)
# sample3 = get_signal_segment(E1_data, sf, 100, 10)
# sample4 = get_signal_segment(E1_data, sf, 1000, 10)
#
# plt.subplot(221)
# plt.specgram(fft(sample1),Fs=200)
# plt.subplot(222)
# plt.specgram(fft(sample2),Fs=200)
# plt.subplot(223)
# plt.specgram(fft(sample3),Fs=200)
# plt.subplot(224)
# plt.specgram(fft(sample4),Fs=200)
#
# plt.plot(samples1[0])
# plt.show()
