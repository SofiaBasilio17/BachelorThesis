import pyedflib
from pyedflib import EdfReader
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
from scipy import signal
from scipy.fftpack import fft, rfft, fftfreq
from operator import add


def hours_to_secs(hours):
    return hours * 3600

def get_signal_segment(signal, sf, onset=0, duration=10):
    # Gets a sample of a signal
    duration = duration * sf
    onset = onset * sf
    return signal[onset:(onset+duration)]

def get_sample_frequency(index):
    # Gets the sampling frequency of a signal
    header = reader.getSignalHeader(index)
    return header.get('sample_rate')

def plot_spectogram(label, samplingFrequency, signalData):
    # Plot rfft and spectrogram of a signal

    testx = np.arange(0, 5, 1/samplingFrequency)
    testy = np.sin(testx * 2 * 3.14 * 30)
    testy = list(map(add, testy, np.sin(testx * 2 * 3.14 * 80)*0.5))

    plt.subplot(221)
    plt.title('FFT and spectrogram of ' + label)
    # Plotting the Normal Signal
    plt.plot(signalData)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    # using rfft instead of fft due to problems with complex numbers
    plt.subplot(222)
    y = rfft(signalData)
    # y = np.abs(y)
    yy = np.abs(y)
    sampfreq = fftfreq(len(testy), d=1/samplingFrequency)
    # power = np.abs(y)
    plt.plot(y)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.subplot(223)
    plt.specgram(signalData,Fs=samplingFrequency)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()

def get_label_info(label,labels):
    # Returns sampling frequency and data of a label
    index = labels.index(label)
    frequency = get_sample_frequency(index)
    data = reader.readSignal(index)
    return frequency, data



reader = EdfReader("../../Sofia_recordings/Rec1.edf")

labels = reader.getSignalLabels()
max_time = reader.file_duration
# print("labels:", labels)
# print("Duration: ", reader.file_duration)

# E1_index = labels.index("Audio")
# E1Header = reader.getSignalHeader(E1_index)
# sf = E1Header.get('sample_rate')
# E1_data = reader.readSignal(E1_index)
#
# plot_spectogram('Audio',sf,E1_data)

w_labels = ['E1','E2','E3','E4']
#,'Right Leg','Left Leg','SpO2'
samples1 = {}

for i in range(0,max_time,30):
    for w in w_labels:
        frequency, data = get_label_info(w,labels)
        samples1[w] = get_signal_segment(data, frequency, i, 30)
    fig, ax = plt.subplots(4)
    ax[0].plot(samples1['E1'],label='E1')
    ax[1].plot(samples1['E2'],label='E2')
    ax[2].plot(samples1['E3'],label='E3')
    ax[3].plot(samples1['E4'],label='E4')
    offset = str(i + 30)
    fig.suptitle(str(i) + ' - ' + offset + 'Seconds, EEGs')
    plt.subplots_adjust(hspace=0.39)
    plt.savefig('Epochs/REC1/'+ str(i) + '_' + offset +'.png')
    plt.close()



# for signal in samples1:
#     if signal == 'SpO2':
#         break
#     ax.plot(time1,samples1[signal],label=signal)
#

# plt.plot(sample1)
# plt.show()
# plot_spectogram('E1',frequency,data)

# sf = 200
# samples1 = []
# #### sampling the first 30 seconds
# for l in labels:
#     tempIndex = labels.index(l)
#     print(tempIndex)
#     tempFrequency = get_sample_frequency(tempIndex)
#     print('sampling frequency of ', l,' is ', tempFrequency)
#     tempData = reader.readSignal(tempIndex)
#     samples1.append(get_signal_segment(tempData, tempFrequency, 0, 30))
#     print(samples1[0])
