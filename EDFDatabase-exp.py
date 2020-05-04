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
Make a CSV file with all the data from the samples
Add to the dataframe the column of class with each stage equivalent to the sample
'''



def get_signal_segment(signal, sf, onset=0, duration=30):
    # Gets a sample of a signal using its sample rate (sf), start at onset and duration of 30 seconds
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
    # writing to CSV file
    dataframe.to_csv(r'test.csv')


# reading from recording
rec1 = EdfReader("sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4001E0-PSG.edf")

# reading from scoring format
staged1 = EdfReader("sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4001EC-Hypnogram.edf")

# getting signal labels
labels = rec1.getSignalLabels()

# counter to tell which stage it is
stage_counter = 0

# annotations from the scoring, includes 2 arrays, one with duration
# and the second with the respective stage
staging = staged1.readAnnotations()

# full duration of the recording
max_time = rec1.file_duration

# dictionary to hold the samples of signals of 30 seconds
samples = {}

# initialization of pandas dataframe
dataframe = create_dataframe(labels)


'''From the beginning of the recording until its end I iterated through the data every
30 seconds,extracting the sample rate for each signal and the data in those 30 seconds.'''
# iterating through the time as epochs of 30 seconds
for i in range(0,max_time,30):
    # if we are done with the duration of the current stage
    if staging[0][stage_counter+1] < i:
        # going through the stages, staging[0][x+1] is the time the stage ends
        # increasing counter for the next stage if we have reached the previous' end
        stage_counter += 1

    # going through each signal
    for l in labels:

        # extracting the sample rate (frequency) and data for the signal
        frequency, data = get_label_info(rec1,l,labels)

        # accordingly to the sample rate and the time (30 seconds) extracting the signal segment
        temp = get_signal_segment(data, frequency, i, 30)

        # creating the epoch for the corresponding signal, the average represents the 30 seconds of data
        samples[l] = np.average(np.array(temp))

    # adding the sample (containing all signals of the course of 30 seconds) and the respective stage
    # to the dataframe
    dataframe = add_to_dataframe(samples,dataframe,staging[2][stage_counter])

# finilizing the collection of all samples, writing dataframe to a CSV file, to be used later by the SVM
write_to_csv(dataframe)
