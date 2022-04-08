# encoding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Churn_Modelling.csv")

X = dataset.iloc[:, 3:-1]
y = dataset.iloc[:, -1].values
colnames = list(X)

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
preprocess = make_column_transformer( (OneHotEncoder(), ['Geography', 'Gender']),
                                      (StandardScaler(), ['CreditScore', 'Age', 'Tenure', 'Balance',
                                                          'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                                                          'EstimatedSalary'])   )
X = preprocess.fit_transform(X)
X = np.delete(X, [0, 3], 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0) 
