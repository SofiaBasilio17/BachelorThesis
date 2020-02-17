import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
bankdata = pd.read_csv("bill_authentication.csv")

print(bankdata.shape)

#### divide the data into attributes and labels
X = bankdata.drop('Class', axis=1)
y = bankdata['Class']

#### divide data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

#### train the algorithm on the training data
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

#### predicting
y_pred = svclassifier.predict(X_test)


print(y_pred)
#### evaluating the algorithm
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
