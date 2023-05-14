import pandas as pd
from scipy import stats
from sklearn import discriminant_analysis as da
from scipy import stats
import random



Y_test = pd.read_csv("Y_test.csv").to_numpy()
Y_train = pd.read_csv("Y_train.csv").to_numpy()

