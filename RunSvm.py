'''
Small data preprocessing.
Runs an SVM with a test size of 40%, prints confusion, accuracy and general score.
'''
import numpy as np
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def count_Null(df):
    # checks for any NaN values that need to be taken care of
    return df.isnull().sum()

def fill_Nan(df):
    # filling all Nan values with 0
    return df.fillna(0)

sleep_data = pd.read_csv("CSV_Sofia/sofiaReadySVM2.csv")

#Resetting indexes
sleep_data = sleep_data.reset_index()

# Filling Nan Values
sleep_data = fill_Nan(sleep_data)
#print(count_Null(sleep_data))


# dropping useless index
sleep_data = sleep_data.drop(['index'], axis=1)
# some error was accurring with a non existing column called level_0, fixed
sleep_data = sleep_data.drop(['level_0'], axis=1)

# check all the types
#print(sleep_data.dtypes)


#### divide the data into attributes and labels
X = sleep_data.drop('Class', axis=1)
y = sleep_data['Class']

#### divide data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40)

#### train the algorithm on the training data
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

#### predicting
y_pred = svclassifier.predict(X_test)



### evaluating the algorithm
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print(svclassifier.score(X_test, y_test))
