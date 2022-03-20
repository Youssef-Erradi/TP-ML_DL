# encoding:utf-8
import pandas as pd

dataset = pd.read_csv('vinegar_quality.csv')
X = dataset.iloc[:,:11].values
y = dataset.iloc[:,-1].values
