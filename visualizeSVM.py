from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
def count_Null(df):
    # checks for any NaN values that need to be taken care of
    return df.isnull().sum()

def fill_Nan(df):
    # filling all Nan values with 0
    return df.fillna(0)


#sleep_data = pd.read_csv("CSV/pediatric1ReadySVM.csv")
sleep_data = pd.read_csv("CSV/sofiaReadySVM2.csv")

#Resetting indexes
sleep_data = sleep_data.reset_index()

# Filling Nan Values
sleep_data = fill_Nan(sleep_data)
#print(count_Null(sleep_data))


# dropping useless index
sleep_data = sleep_data.drop(['index'], axis=1)
# some error was accurring with a non existing column called level_0, fixed
#sleep_data = sleep_data.drop(['level_0'], axis=1)

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

ax = plt.gca()
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, s=50, cmap='autumn')
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])

ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.show()
