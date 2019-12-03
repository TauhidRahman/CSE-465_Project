# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

"""**Importing the Dataset**"""

data = pd.read_csv('dataset on suicide rates.csv')
data = data.sort_values(['year'], ascending = True)

print(data.shape)

data.head()

"""**Data Visualization**"""

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

#splitting the data into dependent and independent variables

x = data.iloc[:,:-1]
y = data.iloc[:,-1]

print(x.shape)
print(y.shape)

# plotter

x = np.array(data.loc[:,'year']).reshape(-1,1)
y = np.array(data.loc[:,'suicides']).reshape(-1,1)
#Scatter Plot
plt.figure(figsize = [10,10])
plt.scatter(x=x,y=y,)
plt.xlabel('Year')
plt.ylabel('Suicides number')
plt.show()

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

# Lineer Regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
predict_space = np.linspace(min(x), max(x)).reshape(-1,1)  # Prediction Space
#print(predict_space)
lis = ['female', 'male']
lis2 = ['5-14 years', '15-24 years', '25-34 years', '35-54 years', '55-74 years', '75+ years']
for i in lis:
    for k in lis2:
        data_1 = data[data['gender'] == i]
        data_sex = data_1[data_1['age'] == k ]
        x_sex = np.array(data_sex.loc[:,'year']).reshape(-1,1)
        y_sex = np.array(data_sex.loc[:,'suicides']).reshape(-1,1)
        reg.fit(x_sex,y_sex)                                               # Fit
        predicted = reg.predict(predict_space)                     # Prediction
        print( i, k, 'R^2 Score: ', reg.score(x_sex,y_sex))                       # R^2 calculation
        # print(i)
        #plt.figure(figsize = [9,6])
        #print(i,k)
        plt.plot(predict_space, predicted, color = 'black', linewidth = 2)
        plt.scatter(x_sex,y_sex)
        plt.title('Scatter Plot')
        plt.xlabel('Year')
        plt.ylabel('Suicides number')
        plt.show()







