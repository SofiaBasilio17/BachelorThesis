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
from sklearn.decomposition import PCA
import pylab as pl

def count_Null(df):
    # checks for any NaN values that need to be taken care of
    return df.isnull().sum()

def fill_Nan(df):
    # filling all Nan values with 0
    return df.fillna(0)


sleep_data = pd.read_csv("CSV/sofiaReadySVM.csv")
#sleep_data = pd.read_csv("CSV/pediatric1ReadySVM.csv")

#Resetting indexes
sleep_data = sleep_data.reset_index()

# Filling Nan Values
sleep_data = fill_Nan(sleep_data)


# dropping useless index
sleep_data = sleep_data.drop(['index'], axis=1)


#### divide the data into attributes and labels
X = sleep_data.drop('Class', axis=1)
y = sleep_data['Class']

#### divide data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40)

#### train the algorithm on the training data
svclassifier = SVC(kernel='poly',degree=6,coef0=7)


#### Uncomment this block to visualize PCA
#### Visualization block: start
# y_train = y_train.reset_index()
# y_train = y_train.drop(['index'],axis=1)
# pca = PCA(n_components=2).fit(X_train)
# pca_2d =  pca.transform(X_train)
#
# for i in range(0, pca_2d.shape[0]):
#     if y_train.values[i] == 'REM':
#         c1 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='r',marker='+')
#     elif y_train.values[i] == 'NREM':
#         c2 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='g',marker='o')
# pl.xlabel('Principal Component 1')
# pl.ylabel('Principal Component 2')
# pl.legend([c1,c2],['REM','NREM'])
# pl.title('PSG Training dataset')
# pl.show()
#### Visualization block: end

#### training the svm
svclassifier.fit(X_train, y_train)

#### predicting the svm
y_pred = svclassifier.predict(X_test)



### evaluating the algorithm
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print(svclassifier.score(X_test, y_test))
