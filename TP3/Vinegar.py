# encoding:utf-8
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("vinegar_quality.csv")
print(dataset)
X = dataset.iloc[:,:11].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

from sklearn.tree import DecisionTreeRegressor
classifier = DecisionTreeRegressor(random_state=0)
classifier.fit(X_train, y_train)

from sklearn import tree
tree.plot_tree(classifier)
plt.show()
