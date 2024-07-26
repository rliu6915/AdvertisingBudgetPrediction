# https://youtu.be/bluclMxiUkA
"""
Multiple Linear Regression uses several explanatory variables to predict the outcome of a response variable.
There are a lot of variables and multiple linear regression is designed to create a model 
based on all these variables. 

#Advertising data
The effect that the independent variables TV, radio, and newspaper have on the dependent variable sales.
have on the dependent variable sales.


"""

import pandas as pd
import seaborn as sns
import numpy as np

df = pd.read_csv('files_for_training_model/Advertising.csv')
print(df.head())

df = df.drop("Unnamed: 0", axis=1)
#A few plots in Seaborn to understand the data

sns.lmplot(x='TV', y='sales', data=df)  
sns.lmplot(x='radio', y='sales', data=df)  
sns.lmplot(x='newspaper', y='sales', data=df)  


x_df = df.drop('sales', axis=1)
y_df = df['sales']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=42)

from sklearn import linear_model

#Create Linear Regression object
model = linear_model.LinearRegression()

#Now let us call fit method to train the model using independent variables.
#And the value that needs to be predicted (Images_Analyzed)

model.fit(X_train, y_train) #Indep variables, dep. variable to be predicted
print(model.score(X_train, y_train))  #Prints the R^2 value, a measure of how well


prediction_test = model.predict(X_test)    
print(y_test, prediction_test)
print("Mean sq. errror between y_test and predicted =", np.mean(prediction_test-y_test)**2)

import pickle
pickle.dump(model, open('models/model.pkl','wb'))

model = pickle.load(open('models/model.pkl','rb'))
print(model.predict([[230, 37, 69]]))


#Model is ready. Let us check the coefficients, stored as reg.coef_.
#These are a, b, and c from our equation. 
#Intercept is stored as reg.intercept_
#print(model.coef_, model.intercept_)

#All set to predict the number of images someone would analyze at a given time
#print(model.predict([[13, 2, 23]]))
