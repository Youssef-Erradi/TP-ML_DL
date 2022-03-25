# encoding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=300, centers=4, cluster_std=.6, random_state=0)

plt.scatter(X[:,0], X[:,1])
plt.show()
