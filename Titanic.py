########### TITANIC KAGGLE COMPETITION ###########

import pandas as pd
import numpy as np
import matplotlib as plt
import math
import sklearn

#read in train & test files
Original_train = pd.read_csv("train.csv")
Original_test = pd.read_csv("test.csv")

#modify test data
train = Original_train.copy()
test = Original_test.copy()


##################### DATA CLEANSING ####################

#delete irrelevant columns from both train and test data sets
del train['PassengerId']
del test['PassengerId']
del train['Name']
del test['Name']
del train['Ticket']
del test['Ticket']
del train['Cabin']
del test['Cabin']

#Have a look at the data and check for missing values
train.info()
print("############################")
test.info()
print("############################")

#create a combined data set and find the means & modes of the attributes with missing values.
#These will be used for imputation
combined_data = pd.concat((train.iloc[:,1:8],test))
combined_data.info()
print("############################")
age_mean = combined_data['Age'].mean()
fare_mean = combined_data['Fare'].mean()
embarked_mode = combined_data['Embarked'].mode()[0]

#manual imputation for training set
for i in range(len(train)):
    if (pd.isnull(train.loc[i,'Age'])):
        train.loc[i,'Age'] = age_mean
    if (pd.isnull(train.loc[i,'Embarked'])):
        train.loc[i,'Embarked'] = embarked_mode
        
#manual imputation for test set
for i in range(len(test)):
    if (pd.isnull(test.loc[i,'Age'])):
        test.loc[i,'Age'] = age_mean
    if (pd.isnull(test.loc[i,'Fare'])):
        test.loc[i,'Fare'] = fare_mean


##################### DATA VISUALISATION ####################
train['Survived'].hist()
train[train['Survived']==1]['Age'].hist()
#train[train['Survived']==0]['Age'].hist()


##################### FEATURE ENGINEERING ####################
def FeatureEngineering(data, model='KNN'):
    if (model=='KNN'):
        return
    else :
        return

        
        
##################### PREDICTIVE MODEL ####################





