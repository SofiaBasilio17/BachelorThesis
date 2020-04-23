import pyedflib
from pyedflib import EdfReader
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
'''
The annotations contain the duration of each stage for example Stage W duration = 1560 Seconds
I have to split the data into 30 second epochs and get the corresponding stage for that,
EG: the first 1560/30 epochs are of stage W
1. Split 2 recordings from the data into 30 second Epochs
2. Split 1 recording from the data into 30 second Epochs
3. Train on 1
4. Test and compare on 2 for accuracy and confusion matrix
'''


'''
TODO 1/04
Make a CSV file with all the data from the samples
Add to the dataframe the column of class with each stage equivalent to the sample
'''
# def split_staged_segments(start, end, duration=30, stage):
#     epochs = {}

def get_signal_segment(signal, sf, onset=0, duration=30):
    # Gets a sample of a signal
    duration = duration * sf
    onset = onset * sf
    return signal[onset:(onset+duration)]

def get_sample_frequency(rec1,index):
    # Gets the sampling frequency of a signal
    header = rec1.getSignalHeader(index)
    return header.get('sample_rate')

def get_label_info(rec1,label,labels):
    # Returns sampling frequency and data of a label
    index = labels.index(label)
    frequency = get_sample_frequency(rec1,index)
    data = rec1.readSignal(index)
    return frequency, data


def create_dataframe(columns):
    # creating the pandas dataframe to hold all the values
    dataframe = pd.DataFrame(columns=columns)
    return dataframe

def add_to_dataframe(data,dataframe,stage):
    # adding new row of samples to the dataframe
    data['Class'] = stage
    tempdf = pd.DataFrame.from_dict([data])
    dataframe = dataframe.append(tempdf, ignore_index=True)
    return dataframe

def write_to_csv(dataframe):
    dataframe.to_csv(r'test.csv')



rec1 = EdfReader("sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4001E0-PSG.edf")
staged1 = EdfReader("sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4001EC-Hypnogram.edf")
labels = rec1.getSignalLabels()


stage_counter = 0
staging = staged1.readAnnotations()
max_time = rec1.file_duration
samples = {}
dataframe = create_dataframe(labels)
# need to make a counter to tell which stage it is
for i in range(0,max_time,30):
    # iterating through the time as epochs of 30 seconds
    if staging[0][stage_counter+1] < i:
        #going through the stages, staging[x+1] is the time the stage ends
        stage_counter += 1
    for l in labels:
        frequency, data = get_label_info(rec1,l,labels)
        temp = get_signal_segment(data, frequency, i, 30)
        samples[l] = np.average(np.array(temp))
        print(samples[l])
    dataframe = add_to_dataframe(samples,dataframe,staging[2][stage_counter])
write_to_csv(dataframe)
