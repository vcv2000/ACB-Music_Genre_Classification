import matplotlib
import pandas as pd
import sklearn
from sklearn import discriminant_analysis as da
from scipy import stats
import random
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
import pickle as pl



def yaltering(Y_train,Y_test,string):
    A = Y_test.to_numpy().transpose()[0]
    B = Y_train.to_numpy().transpose()[0]
    test = []
    train = []
    for i in A:
        if i == string:
            test.append(1)
        else:
            test.append(0)
    for j in B:
        if j == string:
            train.append(1)
        else:
            train.append(0)
    training = np.array(train)
    testing = np.array(test)
    return training,testing

def modelbuilding(X,Y):
    model = da.LinearDiscriminantAnalysis(n_components=1,shrinkage=0.1,solver="lsqr",tol=0.0001)
    model.fit(X, Y)
    return model




Y_test = pd.read_csv("Y_test.csv")
Y_train = pd.read_csv("Y_train.csv")
X_train = pd.read_csv("X_train_u.csv").iloc[:, 1:]
X_test = pd.read_csv("X_test_u.csv").iloc[:, 1:]
classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
values = []
dict = {}
for i in classes:
    Y_train_rock,Y_test_rock = yaltering(Y_train,Y_test,i)
    rockmodel =modelbuilding(X_train,Y_train_rock)
    test_y_prediction = rockmodel.predict(X_test)
    rho = sklearn.metrics.balanced_accuracy_score(Y_test_rock, test_y_prediction)
    values.append([i,rho])
    dict[i] = [Y_test_rock,test_y_prediction]
    pl.dump(rockmodel, open(f"{i}model.sav", 'wb'))

print(values)