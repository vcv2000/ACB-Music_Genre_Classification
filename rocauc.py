from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
import numpy as np

file = pd.read_csv("dados.csv")
Dict ={'blues':0, 'classical':1, 'country':2, 'disco':3, 'hiphop':4, 'jazz':5, 'metal':6, 'pop':7, 'reggae':8, 'rock':9}

X = file.iloc[:,1:-1]
Y = file.iloc[:,-1].to_numpy()
for i in range(len(Y)):
    Y[i] = int(Dict[Y[i]])

Y = pd.DataFrame(Y)
roc = []
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


print(X_test.shape)
print(y_test.shape)

for i in X_train.columns:
    clf = DecisionTreeClassifier()
    clf.fit(X_train[i].fillna(0).to_frame(),y_train.astype('int'))
    score = clf.predict_proba(X_test[i].to_frame())
    roc.append(roc_auc_score(y_test.astype('int'),pd.DataFrame(score[:,1]), multi_class='ovr'))

print(roc)