from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

pca = PCA(whiten=True)
digit = pd.read_csv('D:\\Users\\pengj\\OneDrive\\program\\JUNK\\mnist_train.csv')
train = digit.values[1:, :].astype(int)
pca.fit(train)
exr = pca.explained_variance_ratio_

x = []
y = []
line85 = []
line98 = []
for i in range(len(exr)) :
    x.append(i)
    line85.append(0.85)
    line98.append(0.98)
    if i == 0:
        y.append(exr[0])
    else:
        y.append(exr[i]+y[i-1])

plt.plot(x, y)
plt.plot(x, line85)
plt.plot(x, line98)
plt.show()
