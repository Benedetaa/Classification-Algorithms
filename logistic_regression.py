import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap
from matplotlib.colors import ListedColormap
from scipy.interpolate import make_interp_spline

dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)



# # print(y_train)
# # print(X_test)
# # print(y_test)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# # print(X_test)


classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train) #Train the Machine / Algorithm

# print(classifier.predict(sc.transform([[45,30000]])))

y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


cm = confusion_matrix(y_test, y_pred)
print(cm)
ac_sc = accuracy_score(y_test, y_pred)
print(ac_sc)



