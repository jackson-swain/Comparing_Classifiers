import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

##Load the data
data = pd.read_csv("marriage.csv", header=None)
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

##Split data into train/test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

##Scalar features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

##Define the classifiers
classifiers = {
    "Naive Bayes": GaussianNB(var_smoothing=1e-3),
    "Logistic Regression":LogisticRegression(random_state=44),
    "KNN":KNeighborsClassifier(n_neighbors=3)
}

##Print the accuracy of each method
for name, clf in classifiers.items():
    clf.fit(x_train_scaled, y_train)
    accuracy = accuracy_score(y_test, clf.predict(x_test_scaled))
    print(f"{name}: {accuracy}")

##Run PCA to reduce to 2D
pca = PCA(n_components=2)
x_train_pca = pca.fit_transform(x_train_scaled)
x_test_pca = pca.transform(x_test_scaled)

##Plot the results
fig, axes = plt.subplots(1, 3, figsize=(15,5))
for i, (name, clf) in enumerate(classifiers.items()):
    ##Train on 2D data
    clf.fit(x_train_pca, y_train)
    accuracy = accuracy_score(y_test, clf.predict(x_test_pca))
    print(f"{name}: {accuracy}")

    ##Plot the decision boundary
    h = 0.1
    x_min, x_max = x_train_pca[:,0].min() - 1, x_train_pca[:,0].max() + 1
    y_min, y_max = x_train_pca[:,1].min() - 1, x_train_pca[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    axes[i].contourf(xx, yy, z, alpha=0.5, cmap="RdBu")
    axes[i].scatter(x_train_pca[:,0], x_train_pca[:,1], c=y_train, cmap="RdBu", edgecolors='k')
    axes[i].scatter(x_test_pca[:,0], x_test_pca[:,1], c=y_test, cmap="RdBu", marker="^", alpha=0.6)

plt.tight_layout()
plt.show()