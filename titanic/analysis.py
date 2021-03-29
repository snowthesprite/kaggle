import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sys

df = pd.read_csv('/home/runner/kaggle/dataset.csv')

keep_cols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']

df = df[keep_cols]

def change_sex_to_int (sex) :
    if sex == 'male' :
        return 0
    elif sex == 'female' :
        return 1

df['Sex'] = df['Sex'].apply(change_sex_to_int)


age_nan = df['Age'].apply(lambda entry : np.isnan(entry))
age_not_nan = df['Age'].apply(lambda entry : not np.isnan(entry))

mean_age = df['Age'][age_not_nan].mean()

df['Age'][age_nan] = mean_age


def id_greater_than_zero (element) :
    if element > 0 :
        return 1
    else :
        return 0

df['SibSp>0'] = df['SibSp'].apply(id_greater_than_zero)


df['Parch>0'] = df['Parch'].apply(id_greater_than_zero)
del df['Parch']


df['Cabin'] = df['Cabin'].fillna('None')
def get_cabin_type (cabin) :
    if cabin != 'None' :
        return cabin[0]
    else :
        return cabin

df['CabinType'] = df['Cabin'].apply(get_cabin_type)

for cabin_type in df['CabinType'].unique() :
    dummy_name = 'CabinType=' + cabin_type
    dummy_vals = df['CabinType'].apply(lambda entry : int(entry == cabin_type))
    df[dummy_name] = dummy_vals

del df['CabinType']


df['Embarked'] = df['Embarked'].fillna('None')

for port in df['Embarked'].unique() :
    dummy_name = 'Embarked=' + port
    dummy_vals = df['Embarked'].apply(lambda entry : int(entry == port))
    df[dummy_name] = dummy_vals


del df['Cabin']
del df['Embarked']


#''''
used_features = [
    'Sex',
    'Pclass',
    'Fare',
    'Age',
    'SibSp', 'SibSp>0',
    'Parch>0',
    'Embarked=C', 'Embarked=None', 'Embarked=Q', 'Embarked=S', 
    'CabinType=A', 'CabinType=B', 'CabinType=C', 'CabinType=D', 'CabinType=E', 'CabinType=F', 'CabinType=G', 'CabinType=None', 'CabinType=T']

cols_needed = ['Survived'] + used_features
df = df[cols_needed]
#'''



df_train = df[:500]
df_test = df[500:]

arr_train = np.array(df_train)
arr_test = np.array(df_test)

y_train = arr_train[:,0]
y_test = arr_test[:,0]

x_train = arr_train[:,1:]
x_test = arr_test[:,1:]

regressor = LinearRegression()
regressor.fit(x_train, y_train)

coeff_dict = {'Constant' : regressor.intercept_}
feature_cols = df_train.columns[1:]
feature_coeffs = regressor.coef_

for i in range(len(feature_cols)) :
    col = feature_cols[i]
    coeff = feature_coeffs[i]
    coeff_dict[col] = coeff

y_test_predictions = regressor.predict(x_test)
y_train_predictions = regressor.predict(x_train)

def from_regress_out_to_survival_predict (output) :
    if output < 0.5 :
        return 0 
    else :
        return 1

y_test_predictions = [from_regress_out_to_survival_predict(output) for output in y_test_predictions]
y_train_predictions = [from_regress_out_to_survival_predict(output) for output in y_train_predictions]

def find_accuracy (predict, actual) :
    num_correct = 0
    num_incorrect = 0
    for i in range(len(predict)) :
        if predict[i] == actual[i] :
            num_correct += 1
        else :
            num_incorrect+=1 
    return num_correct/(num_correct + num_incorrect)



print("\n",'features:', used_features)
print('train:', find_accuracy(y_train_predictions, y_train))
print('test:', find_accuracy(y_test_predictions, y_test), "\n")

print(coeff_dict)