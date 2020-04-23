
import numpy as np
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


sleep_data = pd.read_csv("test.csv")
#### divide the data into attributes and labels
X = sleep_data.drop('Class', axis=1)
y = sleep_data['Class']

#### divide data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

#### train the algorithm on the training data
svclassifier = SVC(kernel='sigmoid')
svclassifier.fit(X_train, y_train)

#### predicting
y_pred = svclassifier.predict(X_test)



#### evaluating the algorithm
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
