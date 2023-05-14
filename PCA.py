import pandas as pd
import sklearn.metrics
from scipy import stats
from sklearn import discriminant_analysis as da
from sklearn.decomposition import PCA
import random

random.seed(455664)


def modelbuilding(X,Y,X_test,Y_test):
    model = da.LinearDiscriminantAnalysis()
    model.fit(X, Y)
    test_y_prediction = model.predict(X_test)
    rho= sklearn.metrics.accuracy_score(Y_test, test_y_prediction)
    return rho


def KRUSTY(Data_train,Data_test,Hs,Percentagem):
    train_x = Data_train.copy()
    test_x = Data_test.copy()

    length = len(Hs)
    i = 0
    loss = []
    while (i/length)*100 < Percentagem:
        loss.append(Hs[i])
        i+=1

    for i in loss:
        train_x.pop(i)
        test_x.pop(i)
    return train_x,test_x


X_test = pd.read_csv("X_test.csv")
X_train = pd.read_csv("X_train.csv")
Y_test = pd.read_csv("Y_test.csv")
Y_train = pd.read_csv("Y_train.csv")
Hs = open("Hs.txt","r")
H = []
for i in Hs.read().split("\n")[1:-1]:
    H.append(i.split(" ")[1].split("'")[1])

train, test = KRUSTY(X_train,X_test,H,31)
NOPCA = modelbuilding(train,Y_train,test,Y_test)

pca = PCA(n_components="mle")
train = pca.fit_transform(train)
test = pca.transform(test)
WITHPCA =modelbuilding(train,Y_train,test,Y_test)

print(NOPCA)
#NoPca 0.78
print(WITHPCA)
#WithPca 0.78


