########### TITANIC KAGGLE COMPETITION ###########

import pandas as pd
import numpy as np
import matplotlib as plt
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#read in train & test files
Original_train = pd.read_csv("input/train.csv")
Original_test = pd.read_csv("input/test.csv")

#modify test data
train = Original_train.copy()
test = Original_test.copy()


##################### DATA CLEANSING ####################

#delete irrelevant columns from both train and test data sets
test_ids = test['PassengerId']
del train['PassengerId']
del test['PassengerId']
del train['Name']
del test['Name']
del train['Ticket']
del test['Ticket']
del train['Cabin']
del test['Cabin']

#Have a look at the data and check for missing values
print('Training data...')
train.info()
print("############################")
print('Testing data...')
test.info()
print("############################")

#create a combined data set and find the means & modes of the attributes with missing values.
#These will be used for imputation
combined_data = pd.concat((train.iloc[:,1:8],test))
print('Combined data...')
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
    if (train.loc[i,'Sex'] == 'male'):
        train.loc[i,'Sex'] = 1
    if (train.loc[i,'Sex'] == 'female'):
        train.loc[i,'Sex'] = 2
    if (train.loc[i,'Embarked'] == 'C'):
        train.loc[i,'Embarked'] = 1
    if (train.loc[i,'Embarked'] == 'S'):
        train.loc[i,'Embarked'] = 2
    if (train.loc[i,'Embarked'] == 'Q'):
        train.loc[i,'Embarked'] = 3
        
#manual imputation for test set
for i in range(len(test)):
    if (pd.isnull(test.loc[i,'Age'])):
        test.loc[i,'Age'] = age_mean
    if (pd.isnull(test.loc[i,'Fare'])):
        test.loc[i,'Fare'] = fare_mean
    if (test.loc[i,'Sex'] == 'male'):
        test.loc[i,'Sex'] = 1
    if (test.loc[i,'Sex'] == 'female'):
        test.loc[i,'Sex'] = 2
    if (test.loc[i,'Embarked'] == 'C'):
        test.loc[i,'Embarked'] = 1
    if (test.loc[i,'Embarked'] == 'S'):
        test.loc[i,'Embarked'] = 2
    if (test.loc[i,'Embarked'] == 'Q'):
        test.loc[i,'Embarked'] = 3
        
#split up attributes and target in training data
X_train = train.iloc[:,1:8]
t_train = pd.DataFrame(train.iloc[:,0])

#encode the 3 categorical variables
encoder = OneHotEncoder(categorical_features=[0,1,6],sparse = False)
X_train = pd.DataFrame(encoder.fit_transform(X_train))
X_train = X_train.rename(columns = {0:'Pclass1',1:'Pclass2',2:'Pclass3',3:'Male',4:'Female',5:'EmbarkedC',6:'EmbarkedS',7:'EmbarkedQ',8:'Age',9:'SibSp',10:'Parch',11:'Fare'})
test = pd.DataFrame(encoder.fit_transform(test))
test = test.rename(columns = {0:'Pclass1',1:'Pclass2',2:'Pclass3',3:'Male',4:'Female',5:'EmbarkedC',6:'EmbarkedS',7:'EmbarkedQ',8:'Age',9:'SibSp',10:'Parch',11:'Fare'})



##################### DATA VISUALISATION ####################
#train['Survived'].hist()
#train[train['Survived']==1]['Age'].hist()
#train[train['Survived']==0]['Age'].hist()


##################### FEATURE ENGINEERING ####################
#function to apply feature engineering to the training and testing data
def FeatureEngineering(train, test, model='Random Forest'):
    if (model == 'Random Forest'):
        return
    else:
        return

        
        
##################### PREDICTIVE MODEL ####################
#function to train a model on the training data CV, and predict on the testing data.
def predict(X, t, test, model='Random Forest'):
    #split data into test and train
    X_train, X_val, t_train, t_val = train_test_split(X,t,test_size=0.2)
    print('Training on {} data points, validating on {} data points'.format(len(X_train),len(X_val)))
    if (model == 'Random Forest'):
        #initialise RF model and fit it
        model1 = RandomForestClassifier(n_estimators=50,
                                        max_features=0.8,
                                        max_depth=8,
                                        random_state=2)
        model1.fit(X_train,t_train.values.ravel())
        #predict on the training data and print predictive accuracy
        t_val_predicted = model1.predict(X_val)
        val_accuracy = accuracy_score(t_val,t_val_predicted)
        print("Accuracy of Random Forest on validation set:",val_accuracy)
        #Train again on whole dataset and predict on the test data
        model1.fit(X,t.values.ravel())
        t_test = model1.predict(test)
        return t_test
    else:
        return

##################### PREDICT #############################
t_test = pd.DataFrame(predict(X_train, t_train, test))
#re-attach passenger ids to predictions
t_test = pd.concat([test_ids,t_test],axis=1)
t_test.columns = ['PassengerId','Survived']
#save predictions to csv file
t_test.to_csv('output/test_predictions.csv',index=False)





