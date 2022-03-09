# encoding:utf-8
# page 1
import numpy as np
import pandas as pd

# page 2
dataset = pd.read_csv("Data.csv")

# page 3
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 3].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# page 4
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
labelencoder_y = LabelEncoder() 
y = labelencoder_y.fit_transform(y)

# page 5
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
transform_column = ColumnTransformer([("Catgory", OneHotEncoder(), [0])], remainder="passthrough")
x = transform_column.fit_transform(x)
x = x[:, 1:]

# page 6
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=0) 

# page 8
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
x_train = standard_scaler.fit_transform(x_train) 
x_test = standard_scaler.transform(x_test)
print(x_train)
