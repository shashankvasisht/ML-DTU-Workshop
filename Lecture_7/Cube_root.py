import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import datasets


#from scratch of PCA
#from sklearn import datasets
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash','Alcalinity of ash', 'Magnesium', 'Total phenols','Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins','Color intensity', 'Hue','OD280/OD315 of diluted wines', 'Proline']
X,y = df_wine.iloc[:,1:].values,df_wine.iloc[:,0].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 0)
#print(df_wine.head())


sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

#Eigendecomposition of the covariance matrix.
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

#print('\nEigenvalues \n%s' % eigen_vals)
#print('\nEigenvalues \n%s' % eigen_vecs)
#total and explained variance
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
#print('\variance explained \n%s' % cum_var_exp)


plt.bar(range(1, 14), var_exp, alpha=0.5, align='center',label='individual explained variance')
plt.step(range(1, 14), cum_var_exp, where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
#plt.show()

#Feature transformation
# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda k: k[0], reverse=True)
#print(eigen_pairs)
# arrays (i.e., the sorting algorithm will only regard the
# first element of the tuples, now)

w = np.hstack((eigen_pairs[0][1][:, np.newaxis],eigen_pairs[1][1][:, np.newaxis]))
#print('Matrix W:\n', w)


X_train_pca = X_train_std.dot(w)
#print(X_train_pca)



## Example on iris data set
iris = datasets.load_iris()
#taking the 3D dataset
X = iris["data"][:, :3]
print("3D data")
print(np.shape(X))

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X2D_using_sklearn = pca.fit_transform(X)
print('shape by X2D_using_sklearn')
print(np.shape(X2D_using_sklearn))

pca_components_sklearn = pca.components_
print(pca_components_sklearn,"components from sklearn")

pca_explained_variance_ratio__using_sklearn = pca.explained_variance_ratio_
print(pca_explained_variance_ratio__using_sklearn,"pca_explained_variance_ratio__using_sklearn")

#By projecting down to 2D, we lost about 1.1% of the variance:
print(1 - pca.explained_variance_ratio_.sum())


#another example of PCA on MNIST
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
from sklearn.cross_validation import train_test_split

X = mnist["data"]
y = mnist["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y)
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_train)
print(pca.n_components_) 
#154
print(np.sum(pca.explained_variance_ratio_))
#0.9503623084769207
#both are same approach
pca = PCA(n_components = 154)
X_reduced = pca.fit_transform(X_train)
X_recovered = pca.inverse_transform(X_reduced)


