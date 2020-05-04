# Kivy Imports
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import  Button
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout

# EDF Imports
import pyedflib
from pyedflib import EdfReader


# SVM Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np



''' TODO:
Preview Labels,
Train,
Test,
Show Confusion Matrix,
Accuracy,
'''


class pythonGUI(App):
    def build(self):
        # Initialize Widgets
        self.box = BoxLayout(orientation='vertical')
        # widget.size_hint = (width_percent, height_percent)
        self.loadBtn = Button(text='Load .edf File', on_press=self.loadFile, size_hint=(0.3,0.2))
        self.loadStatus = Label(size_hint=(0.4,0.2))
        self.titlecf = Label(size_hint=(0.4,0.2))
        self.cf = Label(size_hint=(0.4,0.4))
        self.titleclassi = Label(size_hint=(0.4,0.2))
        self.classi = Label(size_hint=(0.4,0.7))
        self.testinglabels = Label(size_hint=(0.4,0.4))
        self.testinglabels.text ="TESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTEST\nTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTEST"

        # Add Widgets to the Box Layout
        self.box.add_widget(self.loadBtn)
        self.box.add_widget(self.loadStatus)
        self.box.add_widget(self.titlecf)
        self.box.add_widget(self.cf)
        self.box.add_widget(self.titleclassi)
        self.box.add_widget(self.classi)
        self.box.add_widget(self.testinglabels)





        return self.box

    def trainAndTest(self):
        # Train and Test with Sigmoid Kernel
        svclassifier = SVC(kernel='poly', degree=8)
        svclassifier.fit(self.X_train, self.y_train)
        self.y_pred = svclassifier.predict(self.X_test)
        self.confusionMatrix = confusion_matrix(self.y_test, self.y_pred)
        self.titlecf.text = "Confusion Matrix: "
        self.cf.text = str(self.confusionMatrix)
        self.classification = classification_report(self.y_test, self.y_pred)
        self.titleclassi.text = "Classification Report: "
        self.classi.text = str(self.classification)
    # def confusionMatrix(self):
    #     # Update Confusion Matrix Label


    def split(self):
        # Splits data into train and test
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        # Assign colum names to the dataset
        colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
        # Read dataset to pandas dataframe
        self.irisdata = pd.read_csv(url, names=colnames)
        X = self.irisdata.drop('Class', axis=1)
        y = self.irisdata['Class']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.20)
        self.trainAndTest()

    def tryLoad(self):
        # Trying to load EDF file
        try:
            self.reader = EdfReader("../../Sofia_recordings/Rec1.edf")
            self.loadBtn.disabled = True
            return True
        except:
            return False

    def loadFile(self,instance):
        if self.tryLoad():
            self.loadStatus.text = "File Loaded successfully"
            self.split()
        else:
            self.loadStatus.text = "File didn't load properly"


if __name__ == '__main__':
    # Running GUI
    pythonGUI().run()
