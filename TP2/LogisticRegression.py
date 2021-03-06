# encoding:utf-8

import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("SocialNet.csv")
start = 2
# dataset["Gender"].replace(to_replace={"Male":1, "Female":0}, inplace=True)
# start = 1

X = dataset.iloc[:, start:-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
ax.matshow(cm, cmap=plt.cm.Greens)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i, s=cm[i, j])
plt.xlabel("Classes réelles")
plt.ylabel("Classes prédites")
plt.show()
