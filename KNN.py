from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import sklearn
from joblib import dump, load

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
y_train = pd.DataFrame(C)
y_test = pd.DataFrame(D)
params = {
        'n_neighbors':	 [1, 2, 3],
        'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size':[10,20,30,50,100],
        'weights':['uniform', 'distance']
        }
y_train = y_train.astype(float)
X_train = X_train.astype(float)
xgb = KNeighborsClassifier()
random_search = RandomizedSearchCV(xgb, param_distributions=params, scoring='roc_auc', n_jobs=4,
                                    verbose=3, random_state=1001 )

random_search.fit(X_train.iloc[100:],y_train.iloc[100:])
results = pd.DataFrame(random_search.cv_results_)

results.to_csv('KNN-random-grid-search-results-01.csv', index=False)
ax = random_search.best_estimator_
ax.fit(X_train,y_train)

dump(ax, 'KNN.joblib')
prediction = ax.predict(X_test)
y_test = y_test.to_numpy().transpose().tolist()[0]



rho = sklearn.metrics.accuracy_score(y_test, prediction)

file = open("KNN.txt","w")

rho2 = sklearn.metrics.f1_score(y_test, prediction,average='macro')
rho3 = sklearn.metrics.precision_score(y_test, prediction,average='macro')
file.write(str(random_search.best_params_) + ":\n" + "Accuracy Score-" + str(rho)+"\nF1 Score-" + str(rho2) + "\nPrecision Score-" + str(rho3))
