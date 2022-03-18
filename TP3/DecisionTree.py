import numpy as np 
import matplotlib.pyplot as plt
# import pandas as pd 

from sklearn.datasets import load_iris 
dataset = load_iris()
print(dataset)

X = dataset.data[:, 2:4] 
y = dataset.target

plt.scatter(X[:, 0][y == 0], X[:, 1][y == 0], color='g', label='setosa') 
plt.scatter(X[:, 0][y == 1], X[:, 1][y == 1], color='r', label='virginica') 
plt.scatter(X[:, 0][y == 2], X[:, 1][y == 2], color='b', label='versicolor') 
plt.legend() 
# plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=0)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='gini', random_state=0)
classifier.fit(X_train, y_train)

from sklearn import tree

X_names = np.array(dataset.feature_names) [2:4]
tree.plot_tree(classifier, feature_names= X_names)
plt.show()
