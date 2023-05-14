import pandas as pd
from scipy import stats
from sklearn import discriminant_analysis as da
from sklearn.decomposition import PCA
import random

random.seed(455664)




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

Hs = open("Hs.txt","r")
H = []
for i in Hs.read().split("\n")[1:-1]:
    H.append(i.split(" ")[1].split("'")[1])

X_test = pd.read_csv("X_test.csv")
X_train = pd.read_csv("X_train.csv")
Y_test = pd.read_csv("Y_test.csv")
Y_train = pd.read_csv("Y_train.csv")
train, test = KRUSTY(X_train,X_test,H,31)
pca = PCA(n_components="mle")
train = pd.DataFrame(pca.fit_transform(train))
test = pd.DataFrame(pca.transform(test))
train.to_csv("X_train_u.csv")
test.to_csv("X_test_u.csv")

