import pickle
import pandas as pd


def Genres(data, guess = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']):
    if type(guess) == list:
        for i in guess:
            loaded_model = pickle.load(open(f"{i}model.sav", 'rb'))
            A = loaded_model.predict(data)
        return A
    else:
        return "Not a list"
X_test = pd.read_csv("X_test_u.csv").iloc[:, 1:]
print(Genres(X_test,guess = ["blues"]))