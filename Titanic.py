########### TITANIC KAGGLE COMPETITION ###########

import pandas as pd

#read in train & test files
Original_train = pd.read_csv("train.csv")
Original_test = pd.read_csv("test.csv")

#modify test data
train = Original_train.copy()
test = Original_test.copy()

#delete irrelevant columns from both train and test data sets
del train['PassengerId']
del test['PassengerId']
del train['Name']
del test['Name']
del train['Ticket']
del test['Ticket']
del train['Cabin']
del test['Cabin']

print hi