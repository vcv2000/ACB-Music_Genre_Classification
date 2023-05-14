import numpy as np
import pandas as pd
import KrustyWalace as KW
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

file = pd.read_csv("Dados.csv")
filex = file.iloc[:,1:-1]
filey = file.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(filex, filey, test_size = 0.2, random_state = 0)

names = X_train.columns
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = pd.DataFrame(X_train)
X_train.columns = names
X_test = pd.DataFrame(X_test)
X_test.columns = names


X_train.to_csv("X_train.csv",index= False)
y_train.to_csv("Y_train.csv",index= False)
X_test.to_csv("X_test.csv",index= False)
y_test.to_csv("Y_test.csv",index= False)
