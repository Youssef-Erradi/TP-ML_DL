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
preprocess = make_column_transformer((OneHotEncoder(), ['Geography', 'Gender']),
                                      (StandardScaler(), ['CreditScore', 'Age', 'Tenure', 'Balance',
                                                          'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                                                          'EstimatedSalary']))
X = preprocess.fit_transform(X)
X = np.delete(X, [0, 3], 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0) 

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
classifier.fit(X_train, y_train, batch_size=10, epochs=100)
y_pred = classifier.predict(X_test) 
y_pred = np.where(y_pred < 0.5, 0, 1)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) 
print(cm)

# ============
df = pd.DataFrame([[600, "France", "Male", 40, 3, 60000, 2, 1, 1, 50000]], columns=colnames)

predict = preprocess.fit_transform(df)
predict = np.delete(predict, [0, 3], 1) 

prediction = classifier.predict(predict)
if(prediction > 0.5):
    print("Le client ne va pas quitter")
else:
    print("Le client va quitter")
