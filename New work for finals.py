#installing libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split



from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score

import warnings
warnings.filterwarnings('ignore')
#importing dataset
data = pd.read_csv('suicide.csv')
data = data.sort_values(['year'], ascending = True)
print(data.shape)
data.head()


# correlation plot
f, ax = plt.subplots(figsize = (4, 3))
corr = data.corr()
sns.heatmap(corr, mask = np.zeros_like(corr, dtype = np.bool), 
            cmap = sns.diverging_palette(3, 3, as_cmap = True), square = True, ax = ax)

# renaming the columns

data.rename({'sex' : 'gender', 'suicides_no' : 'suicides'}, inplace = True, axis = 1)
data.columns
data.info()
data.describe()
data.isnull().sum()
data['country'].value_counts().count()

# visualising the different countries distribution in the dataset

data['country'].value_counts(normalize = True)
data['country'].value_counts(dropna = False).plot.bar(color = 'cyan', figsize = (24, 8))
plt.title('Distribution of 141 coutries in suicides')
plt.xlabel('country name')
plt.ylabel('count')
plt.show()
data['year'].value_counts().count()

# visualising the different year distribution in the dataset

data['year'].value_counts(normalize = True)
data['year'].value_counts(dropna = False,).plot.bar(color = 'magenta', figsize = (8, 6))
plt.title('Distribution of suicides from the year 1985 to 2016')
plt.xlabel('year')
plt.ylabel('count')
plt.show()

# label encoding for gender

from sklearn.preprocessing import LabelEncoder
# creating an encoder

le = LabelEncoder()
data['gender'] = le.fit_transform(data['gender'])
data['gender'].value_counts()
# replacing categorical values in the age column

data['age'] = data['age'].replace('5-14 years', 0)
data['age'] = data['age'].replace('15-24 years', 1)
data['age'] = data['age'].replace('25-34 years', 2)
data['age'] = data['age'].replace('35-54 years', 3)
data['age'] = data['age'].replace('55-74 years', 4)
data['age'] = data['age'].replace('75+ years', 5)
#data['age'].value_counts()

# suicides in different age groups

x1 = data[data['age'] == 0]['suicides'].sum()
x2 = data[data['age'] == 1]['suicides'].sum()
x3 = data[data['age'] == 2]['suicides'].sum()
x4 = data[data['age'] == 3]['suicides'].sum()
x5 = data[data['age'] == 4]['suicides'].sum()
x6 = data[data['age'] == 5]['suicides'].sum()
x = pd.DataFrame([x1, x2, x3, x4, x5, x6])
x.index = ['5-14', '15-24', '25-34', '35-54', '55-74', '75+']
x.plot(kind = 'bar', color = 'grey')
plt.title('suicides in different age groups')
plt.xlabel('Age Group')
plt.ylabel('count')
plt.show()

# visualising the gender distribution in the dataset

data['gender'].value_counts(normalize = True)
data['gender'].value_counts(dropna = False).plot.bar(color = 'black', figsize = (4, 3))
plt.title('Distribution of 141 coutries in suicides')
plt.xlabel('gender')
plt.ylabel('count')
plt.show()

# total population of 141 countres over which the suicides survey is committed

data['population'].sum()
# Average population

Avg_pop = data['population'].mean()
print(Avg_pop)
# total number of suicides committed in the 141 countries from 1985 to 2016

data['suicides'].sum()
# Average suicide in the world

Avg_sui = data['suicides'].mean()
print(Avg_sui)

# Imputing the NaN values from the population column

data['population'] = data['population'].fillna(data['population'].median())
data['population'].isnull().any()

# Imputing the values suicides no column
 
data['suicides'] = data['suicides'].fillna(0)
data['suicides'].isnull().any()
# rearranging the columns 
data = data[['country', 'year', 'gender', 'age', 'population', 'suicides']]
data.head(0)

# Removing the country Column

data = data.drop(['country'], axis = 1)
data.head(0)

#splitting the data into dependent and independent variables

x = data.iloc[:,:-1]
y = data.iloc[:,-1]
print(x.shape)
print(y.shape)

# splitting the dataset into training and testing sets

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 45)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# min max scaling

# importing the min max scaler

from sklearn.preprocessing import MinMaxScaler

# creating a scaler

mm = MinMaxScaler()
# scaling the independent variables
x_train = mm.fit_transform(x_train)
x_test = mm.transform(x_test)
# using principal component analysis
from sklearn.decomposition import PCA
# creating a principal component analysis model
#pca = PCA(n_components = None)
# feeding the independent variables to the PCA model
#x_train = pca.fit_transform(x_train)
#x_test = pca.transform(x_test)
# visualising the principal components that will explain the highest share of variance
#explained_variance = pca.explained_variance_ratio_
#print(explained_variance)
# creating a principal component analysis model
#pca = PCA(n_components = 1)
# feeding the independent variables to the PCA model
#x_train = pca.fit_transform(x_train)
#x_test = pca.transform(x_test)
# applying k means clustering
# selecting the best choice for no. of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
  km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
  km.fit(x_train)
  wcss.append(km.inertia_)
  
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('no. of clusters')
plt.ylabel('WCSS')
plt.show()

# applying kmeans with 4 clusters

km = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_means = km.fit_predict(x_train)

# visualising the clusters

plt.scatter(x_train[y_means == 0, 0], x_train[y_means == 0, 1], s = 100, c = 'pink', label = 'cluster 1')
plt.scatter(x_train[y_means == 1, 0], x_train[y_means == 1, 1], s = 100, c = 'cyan', label = 'cluster 2')
plt.scatter(x_train[y_means == 2, 0], x_train[y_means == 2, 1], s = 100, c = 'magenta', label = 'cluster 3')
plt.scatter(x_train[y_means == 3, 0], x_train[y_means == 3, 1], s = 100, c = 'violet', label = 'cluster 4')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], s = 100, c = 'red', label = 'centroids')
plt.title('Cluster of Clients')
plt.xlabel('cc')
plt.show()
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# creating the model

model = LinearRegression()
# feeding the training data into the model

model.fit(x_train, y_train)
# predicting the test set results

y_pred = model.predict(x_test)

# calculating the mean squared error

mse = np.mean((y_test - y_pred)**2)
print("MSE :", mse)
# calculating the root mean squared error

rmse = np.sqrt(mse)
print("RMSE :", rmse)
#calculating the r2 score

r2 = r2_score(y_test, y_pred)
print("r2_score :", r2)
from sklearn.svm import SVR
# creating the model

model = SVR()
# feeding the training data into the model

model.fit(x_train, y_train)
# predicting the test set results

y_pred = model.predict(x_test)
# calculating the mean squared error

mse = np.mean((y_test - y_pred)**2)
print("MSE :", mse)
# calculating the root mean squared error

rmse = np.sqrt(mse)
print("RMSE :", rmse)
#calculating the r2 score

r2 = r2_score(y_test, y_pred)
print("r2_score :", r2)
from sklearn.ensemble import RandomForestRegressor
# creating the model

model = RandomForestRegressor()

# feeding the training data into the model

model.fit(x_train, y_train)

# predicting the test set results

y_pred = model.predict(x_test)

# calculating the mean squared error

mse = np.mean((y_test - y_pred)**2)
print("MSE :", mse)

# calculating the root mean squared error

rmse = np.sqrt(mse)
print("RMSE :", rmse)

#calculating the r2 score

r2 = r2_score(y_test, y_pred)
print("r2_score :", r2)
from sklearn.tree import DecisionTreeRegressor

# creating the model

model = DecisionTreeRegressor()

# feeding the training data into the model

model.fit(x_train, y_train)

# predicting the test set results

y_pred = model.predict(x_test)

# calculating the mean squared error

mse = np.mean((y_test - y_pred)**2)
print("MSE :", mse)

# calculating the root mean squared error

rmse = np.sqrt(mse)
print("RMSE :", rmse)

#calculating the r2 score

r2 = r2_score(y_test, y_pred)
print("r2_score :", r2)
from sklearn.ensemble import AdaBoostRegressor

# creating the model

model = AdaBoostRegressor()

# feeding the training data into the model

model.fit(x_train, y_train)

# predicting the test set results

y_pred = model.predict(x_test)

# calculating the mean squared error

mse = np.mean((y_test - y_pred)**2)
print("MSE :", mse)

# calculating the root mean squared error

rmse = np.sqrt(mse)
print("RMSE :", rmse)

#calculating the r2 score

r2 = r2_score(y_test, y_pred)
print("r2_score :", r2)
from sklearn.neural_network import MLPClassifier
# creating the model
model = MLPClassifier(hidden_layer_sizes = 100, max_iter = 50 )
# feeding the training data into the model
model.fit(x_train, y_train)

# predicting the test set results
y_pred = model.predict(x_test)

# calculating the mean squared error
mse = np.mean((y_test - y_pred)**2)
print("MSE :", mse)

# calculating the root mean squared error

rmse = np.sqrt(mse)
print("RMSE :", rmse)
#calculating the r2 score
r2 = r2_score(y_test, y_pred)
print("r2_score :", r2)

#logisitic regression

x = os_data_X[final_col]
x_test = X_test[final_col]
x = pd.concat([X,X_test])

y = os_data_y['y']
y = pd.concat([y,y_test])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

logreg = LogisticRegression()
logreg.fit(x, y)

y_pred = logreg.predict(X_test)
logreg_accy_train = round(logreg.score(x_train, y_train),2)
logreg_accy = round(logreg.score(x_test, y_test),2)

print('Accuracy of logistic regression classifier on training set: {:.2f}'.format(logreg_accy_train))
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg_accy))

#knn
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import mean_absolute_error, accuracy_score

## choosing the best n_neighbors
nn_scores = []

best_prediction = [-1,-1]

for i in range(1,100):
    knn = KNeighborsClassifier(n_neighbors=i, weights='distance', metric='minkowski', p =2)
    knn.fit(x_train,y_train)
    score = accuracy_score(y_test, knn.predict(x_test))
    
    #print i, score
    if score > best_prediction[1]:
        best_prediction = [i, score]
    nn_scores.append(score)
    

knn_best = KNeighborsClassifier(n_neighbors= best_prediction[0], weights='distance', metric='minkowski', p =2)
knn_best.fit(x_train,y_train)
knn_accy_train = round(accuracy_score(y_test, knn.predict(x_test)),2)
knn_accy = round(best_prediction[1],2)    

                                 
plt.plot(range(1,100),nn_scores)
print ('The best number of neighbors is: {:.0f}'.format(best_prediction[0]))
print ('Accuracy of KNN classifier on training set: {:.2f}'.format(knn_accy_train))
print ('Accuracy of KNN classifier on test set: {:.2f}'.format(knn_accy))

n_neighbors=[1,2,3,4,5,6,7,8,9,10]
weights=['uniform','distance']
param = {'n_neighbors':n_neighbors, 
         'weights':weights}
grid2 = GridSearchCV(knn, 
                     param,
                     verbose=False, 
                     cv=StratifiedKFold(n_splits=5, random_state=15, shuffle=True)
                    )
grid2.fit(x_train, y_train)

knn_grid = grid2.best_estimator_
knn_grid_accy_train = round(grid2.best_score_,2)
knn_grid_accy = round(knn_grid.score(x_test, y_test),2)

print (grid2.best_params_)
print ('Accuracy of KNN classifier GridSearch CV on training set: {:.2f}'.format(knn_grid_accy_train))
print ('Accuracy of KNN classifier GridSearch CV on test set: {:.2f}'.format(knn_grid_accy ))

#svm

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)
from sklearn.svm import SVC
svclassifier = SVC(kernel='non-linear')
svclassifier.fit(x_train, y_train)
y_pred = svclassifier.predict(x_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
