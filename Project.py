import numpy as np
import pandas as pd
import KrustyWalace as KW
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



random.seed(200)

def sorting(a):
    return a["Value"]


#[[[500.0, 'blues', 100],
# [500.0, 'classical', 100],
# [500.0, 'country', 100],
# [500.0, 'disco', 100],
# [500.0, 'hiphop', 100],
# [500.0, 'jazz', 99],
# [500.0, 'metal', 100],
# [500.0, 'pop', 100],
# [500.0, 'reggae', 100],
# [500.0, 'rock', 100],
# [500.0, 'ALL', 999]], 'chroma_stft_max']
file = pd.read_csv("dados.csv")

X = file.iloc[:,1:-1]
Y = file.iloc[:,-1]
sc = StandardScaler()

X = sc.fit_transform(X)

variables = np.array(file.columns[1:-1])
p = Y.to_numpy().transpose()
Rs = []
Hs = []
c = pd.DataFrame(X).to_numpy().transpose()
for i in range(len(c)-1):
    print(f"Percentage:{(i/(len(c)-1))*100}%")
    R,H= KW.Krustywallece(c[i],p)
    Rs.append([R,variables[i]])
    Hs.append([H,variables[i]])

data = []
columns = []
rows = []
for i in Rs[0][0]:
    columns.append(i[1])
for i in range(len(Rs)):
    rows.append(Rs[i][-1])
    arr = []
    for j in range(len(Rs[i][0])):
        arr.append(Rs[i][0][j][0])
    data.append(arr)


datase = pd.DataFrame(data,index=rows)
datase.columns = columns

datase.to_csv("Data.csv")
fiile = open("Hs.txt","w")
fiile.write(f"Hs for the dataset :\n")
L = []
for i in Hs:
    L.append({"Value" : i[0], "Descriptor": i[1]})

HS = sorted(L, key=sorting)
for i in range(len(HS)):
    fiile.write(f"{HS[i]['Value'],HS[i]['Descriptor']}\n")

foole = open("Rs.txt","w")
foole.write(f"Rs for the dataset :\n")
for i in Rs:
    foole.write(f"{i}\n")
