#------------------------------------------------------------------------------
# Student performance analysis
#------------------------------------------------------------------------------
# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Sklearn libraries
from sklearn.linear_model    import LinearRegression
from sklearn.preprocessing   import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics         import confusion_matrix
from sklearn.model_selection import GridSearchCV

# Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
import keras.models as km

#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------
# Building ANN regressor
def build_regressor():
	model = Sequential()
	model.add(Dense(units = 13, kernel_initializer='normal', activation='relu', input_dim = 12))
	model.add(Dense(units =  6, kernel_initializer='normal', activation='relu'))
	model.add(Dense(units =  1, kernel_initializer='normal'))
	model.compile(optimizer='adam', loss='mean_squared_error',  metrics = ['accuracy'] )
	return model

# Building ANN classifier
def build_classifier(optimizer, neurons, n_layer):
    model = Sequential()
    model.add(Dense(units = neurons, kernel_initializer = 'uniform', activation = 'relu', input_dim = 12))
    if n_layer > 1: 
        for i in range(n_layer - 1):
            model.add(Dense(units = neurons, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

#------------------------------------------------------------------------------
# Preprocessing
#------------------------------------------------------------------------------
# loading data
dataset = pd.read_csv('StudentsPerformance.csv')
X = dataset.iloc[:, 0:5].values

# math score
y = dataset.iloc[:, 5].values 
# reading score
#y = dataset.iloc[:, 6].values

# categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:, 0])
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:, 2])
labelencoder_X_3 = LabelEncoder()
X[:,3] = labelencoder_X_3.fit_transform(X[:, 3])
labelencoder_X_4 = LabelEncoder()
X[:,4] = labelencoder_X_4.fit_transform(X[:, 4])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[: , 1:]
onehotencoder_2 = OneHotEncoder(categorical_features = [5])
X = onehotencoder_2.fit_transform(X).toarray()
X = X[: , 1:]

#------------------------------------------------------------------------------
# Comparison of regressions 
#------------------------------------------------------------------------------
"""
# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)

# Random forest regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)
prediction = regressor.predict(X_test)
error = np.sum(prediction - y_test) / 150

# Multiple linear regression
from sklearn.linear_model import LinearRegression
regressor_lin = LinearRegression()
regressor_lin.fit(X_train, y_train)
prediction_lin = regressor_lin.predict(X_test)
error_lin = np.sum(prediction_lin - y_test) / 150

# Artificial neuronal networks
regressor_ann = build_model()
regressor_ann.fit(X_train, y_train, batch_size = 8, epochs = 100)
prediction_ann = regressor_ann.predict(X_test)

# Score visialization: Score vs Kid
for i in range(len(X_values)): X_values[i] = i
#plt.plot(X_values, y_test, color = 'black', label = 'Real values' )
plt.scatter(X_values, (prediction_ann[0,:] - y_test), color = 'blue', label = 'ANN' )
plt.scatter(X_values, (prediction_lin - y_test), color = 'red', label = 'linear')
plt.scatter(X_values, (prediction - y_test), color = 'green', label = 'Random forest')
#plt.legend()
plt.xlabel('Kid')
plt.ylabel('Math score deviation')
#plt.ylim(0, 100)
plt.show()
"""
#------------------------------------------------------------------------------
# Score over average or not?
#------------------------------------------------------------------------------
"""
# define mean score
mean_score = np.mean(y)
for i in range(len(y)):
    if y[i] > mean_score:
        y[i] = 1
    else:
        y[i] = 0
        
# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

# ANN classifier
classifier = build_classifier('adam', 7, 3)
classifier.fit(X_train, y_train, batch_size = 32, epochs = 200)
prediction = classifier.predict(X_test)
prediction = (prediction > 0.5)
cm = confusion_matrix(y_test, prediction)

# Grid search
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [32],
              'epochs': [200],
              'optimizer': ['adam'],
              'neurons': [7,8],
              'n_layer': [3,4]}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 4)
grid_search     = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy   = grid_search.best_score_

#------------------------------------------------------------------------------
# Save and/or load model
#------------------------------------------------------------------------------
classifier.save('students.h5', overwrite=True)
km.load_model('students.h5')
"""

#------------------------------------------------------------------------------
# Scores vs. parents education
#------------------------------------------------------------------------------
test0, test1, test2, test3, test4, test5 = [], [], [], [], [], []
X_math = dataset.iloc[:, [2,5]].values

for i in range(len(X_math)):
    if X_math[i,0] == 'some high school' : 
        X_math[i,0] = 0
        test0.append(X_math[i,1])
    elif X_math[i,0] == 'high school' : 
        X_math[i,0] = 1   
        test1.append(X_math[i,1])
    elif X_math[i,0] == "associate's degree" : 
        X_math[i,0] = 2
        test2.append(X_math[i,1])
    elif X_math[i,0] == "some college" : 
        X_math[i,0] = 3
        test3.append(X_math[i,1])
    elif X_math[i,0] == "bachelor's degree" : 
        X_math[i,0] = 4
        test4.append(X_math[i,1])
    elif X_math[i,0] == "master's degree" : 
        X_math[i,0] = 5
        test5.append(X_math[i,1])

X_1 = dataset.iloc[:, [5,6]].values
X_1[:,0] = X_math[:,0]

X_new = X_1[0:6, 0:6]
X_new[0,1] = np.average(test0)
X_new[1,1] = np.average(test1)
X_new[2,1] = np.average(test2)
X_new[3,1] = np.average(test3)
X_new[4,1] = np.average(test4)
X_new[5,1] = np.average(test5)

for i in range(len(X_new)):
    X_new[i,0] = i

# Score visialization: Score vs. parents
#plt.plot(X_values, y_test, color = 'black', label = 'Real values' )
plt.scatter(X_new[:,0], X_new[:,1], color = 'blue' )
#plt.legend()
plt.xlabel('Parents education level')
plt.ylabel('average math score')
#plt.ylim(0, 100)
plt.show()


#------------------------------------------------------------------------------
# K-means clustering 
#------------------------------------------------------------------------------
"""
# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X_1)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
X = X_1

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
#plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
#plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
#plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of students')
plt.xlabel('Score 1')
plt.ylabel('Score 2')
plt.legend()
plt.show()
"""

