# -*- coding: utf-8 -*-
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import pandas as pd
import numpy as np

pca = PCA(n_components=0.95, whiten=True)#n_components=0.95,
digit = pd.read_csv('D:\\Users\\pengj\\OneDrive\\program\\JUNK\\mnist_train.csv')
test = pd.read_csv('D:\\Users\\pengj\\OneDrive\\program\\JUNK\\mnist_test.csv')
label = digit.values[:, 0].astype(int)
train = digit.values[:, 1:].astype(int)
test_data = test.values[:, 1:].astype(int)

pca.fit(train)
train_data = pca.transform(train)

svc = SVC()
svc.fit(train_data, label)

test_data = pca.transform(test_data)
ans = svc.predict(test_data)

a = []
for i in range(len(ans)):
    a.append(i+1)

np.savetxt('PCA_0.95_SVC.csv', np.c_[a, ans],
    delimiter=',', header='ImageId,Label', comments='', fmt='%d')
