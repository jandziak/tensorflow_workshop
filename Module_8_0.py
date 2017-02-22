# Tensorflow workshop with Jan Idziak
#-------------------------------------
#
#script is a part from sklean workshop  
#
# Implementing Recurent Neural Network
#---------------------------------------
#
###Scikit Learn Supervised Learning

# Classification

#Load libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn import svm
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import tree
from sklearn import metrics
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import decomposition


iris = datasets.load_iris()
X,y = iris.data, iris.target
# digits = datasets.load_digits()
# X,y = digits.data, digits.target


# Train test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=25)

### Step 1: Load the classifer

# K Nearest Neighbors Classificaiton
clf_knn = neighbors.KNeighborsClassifier()
clf_knn = neighbors.KNeighborsClassifier(n_neighbors=3,weights='distance',
    algorithm='kd_tree')

# Support Vector Machine
clf_svm = svm.SVC()
clf_svm = svm.SVC(kernel='rbf',degree=3)
# Linear Clasifier
clf_lm = linear_model.SGDClassifier()
# Naive Bayes
clf_nb = naive_bayes.GaussianNB()
# Tree model
clf_tree = tree.DecisionTreeClassifier()

### Step 2: Models Training

clf_knn.fit(X_train,y_train)
clf_svm.fit(X_train,y_train)
clf_lm.fit(X_train,y_train)
clf_nb.fit(X_train,y_train)
clf_tree.fit(X_train,y_train)


### Step 3_1: Testing

# K Nearest Neighbors Classificaiton
print(clf_knn.predict(X_test)[:20])
print(y_test[:20])
score = clf_knn.score(X_test,y_test)
print(score)
# Support Vector Machine
print(clf_svm.predict(X_test)[:20])
print(y_test[:20])
score = clf_svm.score(X_test,y_test)
print(score)
# Linear Clasifier
print(clf_lm.predict(X_test)[:20])
print(y_test[:20])
score = clf_lm.score(X_test,y_test)
print(score)
# Naive Bayes
print(clf_nb.predict(X_test)[:20])
print(y_test[:20])
score = clf_nb.score(X_test,y_test)
print(score)
# Tree model
print(clf_tree.predict(X_test)[:20])
print(y_test[:20])
score = clf_tree.score(X_test,y_test)
print(score)

### Step 3_2: Measure the Performance

predicted = clf_nb.predict(X_test)
expected = y_test
matches = (predicted == expected)

score = matches.sum()/len(matches)
print("Score = ", score)
print(metrics.classification_report(expected, predicted))


### Regression


from sklearn import linear_model
boston = datasets.load_boston()
X,y = boston.data, boston.target

print(boston.data.shape)
print(boston.feature_names)
print(boston.target.shape)

lm = linear_model.LinearRegression()
lm.fit(X,y)
print(sum((y-lm.predict(X))**2)/len(y))
 
plt.scatter(y,lm.predict(X))
plt.xlabel('Price')
plt.ylabel('Predict Price')
plt.show()

### Skleanr Unsupervised Learning

# Clustering

## K-Means Clustering
iris = datasets.load_iris()
X,y = iris.data,iris.target
c = cluster.KMeans(n_clusters=2)
c.fit(X) 

print(c.labels_[::10])
print(y[::10])

## Pricipal Component Analysis

digits = datasets.load_digits()
X,y = digits.data, digits.target

pca = decomposition.PCA()
pca.fit(X)
pca.n_components = 3 

X_reduced = pca.fit_transform(X)
plt.scatter(X_reduced[:,0],X_reduced[:,1],c=y)
plt.show()
print(pca.explained_variance_)

### Intro to Neural Networks

iris = datasets.load_iris()
X,y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

from sklearn import neural_network
clf = neural_network.MLPClassifier(20,'identity','sgd')

clf.fit(X_train,y_train)

print(clf.predict(X_test))
print(y_test)
print(X)

