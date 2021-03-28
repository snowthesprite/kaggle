import pandas as pd
import numpy as np

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


'''
def get_cabin_type (cabin) :
    if np.isnan(cabin) :
        return ''
    else :
        return cabin[0]

print(df['Cabin'].apply(get_cabin_type))
#'''

del df['Cabin']
del df['Embarked']

train_df = df[:500]
test_df = [500:]