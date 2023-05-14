import sklearn.metrics
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd





class MDC():
    def __init__(self):
        self.class_list = {}
        self.centroids = {}

    def fit(self, X, y):
        self.class_list = np.unique(y, axis=0)

        self.centroids = np.zeros((len(self.class_list), X.shape[1]))  # each row is a centroid
        for i in range(len(self.class_list)):  # for each class, we evaluate its centroid
            temp = np.where(y == self.class_list[i])[0]
            self.centroids[i, :] = np.mean(X[temp], axis=0)

    def predict(self, X):
        temp = np.argmin(
            cdist(X, self.centroids),  # distance between each pair of the two collections of inputs
            axis=1
        )
        y_pred = np.array([self.class_list[i] for i in temp])

        return y_pred


Dict ={'blues':0, 'classical':1, 'country':2, 'disco':3, 'hiphop':4, 'jazz':5, 'metal':6, 'pop':7, 'reggae':8, 'rock':9}


X_test = pd.read_csv("X_test_u.csv")
X_train = pd.read_csv("X_train_u.csv")
Y_test = pd.read_csv("Y_test.csv")
Y_train = pd.read_csv("Y_train.csv")
C = Y_train.copy().to_numpy()
D = Y_test.copy().to_numpy()
for i in range(len(C)):
    C[i,0] = int(Dict[C[i,0]])
for i in range(len(D)):
    D[i, 0] = int(Dict[D[i, 0]])
scaler = MinMaxScaler()
X_tr_scaled = scaler.fit_transform(X_train)
X_ts_scaled = scaler.transform(X_test)
mdc = MDC()
mdc.fit(X_tr_scaled, C.transpose().tolist()[0])

prediction = mdc.predict(X_ts_scaled)
y_test = D.transpose().tolist()[0]


file = open("Minimumdistance_u.txt","w")
p = sklearn.metrics.accuracy_score(y_test,prediction)
rho2 = sklearn.metrics.f1_score(y_test, prediction,average='macro')
rho3 = sklearn.metrics.precision_score(y_test, prediction,average='macro')
file.write("Accuracy Score-" + str(p)+"\nF1 Score-" + str(rho2) + "\nPrecision Score-" + str(rho3))
