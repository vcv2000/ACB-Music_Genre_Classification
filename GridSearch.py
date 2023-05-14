# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 19:00:46 2023

@author: vitor
"""

import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline




X_train = pd.read_csv("X_train_u.csv")
y_train = pd.read_csv("Y_train.csv")



pipeline = Pipeline([
    ('selector', SelectKBest(spearmanr)),
    ('classifier', LinearDiscriminantAnalysis())
])

param_grid = {
    'feature_selector': [5, 10, 15, 20],
    'LinearDiscriminantAnalysis': [1, 3, 5, 7]
}


grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)


print("Best hyperparameters: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)