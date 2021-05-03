import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

deBug = False

df = pd.read_csv('/home/runner/kaggle/titanic/dataset.csv')

keep_cols = ['Survived', 'Sex', 'Pclass', 'Fare', 'Age', 'SibSp', 'Parch', 'Embarked', 'Cabin']

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


df['Embarked'] = df['Embarked'].fillna('None')

for port in df['Embarked'].unique() :
    dummy_name = 'Embarked=' + port
    dummy_vals = df['Embarked'].apply(lambda entry : int(entry == port))
    df[dummy_name] = dummy_vals


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
del df['Cabin']
del df['Embarked']

terms = list(df.columns)[1:]

#make interaction terms
for col_1 in terms :
    for col_2 in terms :
        interaction = col_1+' * '+col_2
        no_inter = col_1[:-1] not in col_2 and col_2[:-1] not in col_1
        not_in = interaction not in list(df.columns) and col_2+' * '+col_1 not in list(df.columns)
        if no_inter and not_in :
            df[interaction] = df[col_1]*df[col_2]

feature_list = ['Sex',
                 'Pclass', 'Fare', 'Age', 'SibSp', 'SibSp>0', 'Parch>0', 'Embarked=C', 'Embarked=None', 'Embarked=Q', 'Embarked=S', 'CabinType=A', 'CabinType=B', 'CabinType=C', 'CabinType=D', 'CabinType=E', 'CabinType=F', 'CabinType=G', 'CabinType=None', 'CabinType=T', 'Sex * Pclass', 'Sex * Fare', 'Sex * Age', 'Sex * SibSp', 'Sex * SibSp>0', 'Sex * Parch>0', 'Sex * Embarked=C', 'Sex * Embarked=None', 'Sex * Embarked=Q', 'Sex * Embarked=S', 'Sex * CabinType=A', 'Sex * CabinType=B', 'Sex * CabinType=C', 'Sex * CabinType=D', 'Sex * CabinType=E', 'Sex * CabinType=F', 'Sex * CabinType=G', 'Sex * CabinType=None', 'Sex * CabinType=T', 'Pclass * Fare', 'Pclass * Age', 'Pclass * SibSp', 'Pclass * SibSp>0', 'Pclass * Parch>0', 'Pclass * Embarked=C', 'Pclass * Embarked=None', 'Pclass * Embarked=Q', 'Pclass * Embarked=S', 'Pclass * CabinType=A', 'Pclass * CabinType=B', 'Pclass * CabinType=C', 'Pclass * CabinType=D', 'Pclass * CabinType=E', 'Pclass * CabinType=F', 'Pclass * CabinType=G', 'Pclass * CabinType=None', 'Pclass * CabinType=T', 'Fare * Age', 'Fare * SibSp', 'Fare * SibSp>0', 'Fare * Parch>0', 'Fare * Embarked=C', 'Fare * Embarked=None', 'Fare * Embarked=Q', 'Fare * Embarked=S', 'Fare * CabinType=A', 'Fare * CabinType=B', 'Fare * CabinType=C', 'Fare * CabinType=D', 'Fare * CabinType=E', 'Fare * CabinType=F', 'Fare * CabinType=G', 'Fare * CabinType=None', 'Fare * CabinType=T', 'Age * SibSp', 'Age * SibSp>0', 'Age * Parch>0', 'Age * Embarked=C', 'Age * Embarked=None', 'Age * Embarked=Q', 'Age * Embarked=S', 'Age * CabinType=A', 'Age * CabinType=B', 'Age * CabinType=C', 'Age * CabinType=D', 'Age * CabinType=E', 'Age * CabinType=F', 'Age * CabinType=G', 'Age * CabinType=None', 'Age * CabinType=T', 'SibSp * Parch>0', 'SibSp * Embarked=C', 'SibSp * Embarked=None', 'SibSp * Embarked=Q', 'SibSp * Embarked=S', 'SibSp * CabinType=A', 'SibSp * CabinType=B', 'SibSp * CabinType=C', 'SibSp * CabinType=D', 'SibSp * CabinType=E', 'SibSp * CabinType=F', 'SibSp * CabinType=G', 'SibSp * CabinType=None', 'SibSp * CabinType=T', 'SibSp>0 * Parch>0', 'SibSp>0 * Embarked=C', 'SibSp>0 * Embarked=None', 'SibSp>0 * Embarked=Q', 'SibSp>0 * Embarked=S', 'SibSp>0 * CabinType=A', 'SibSp>0 * CabinType=B', 'SibSp>0 * CabinType=C', 'SibSp>0 * CabinType=D', 'SibSp>0 * CabinType=E', 'SibSp>0 * CabinType=F', 'SibSp>0 * CabinType=G', 'SibSp>0 * CabinType=None', 'SibSp>0 * CabinType=T', 'Parch>0 * Embarked=C', 'Parch>0 * Embarked=None', 'Parch>0 * Embarked=Q', 'Parch>0 * Embarked=S', 'Parch>0 * CabinType=A', 'Parch>0 * CabinType=B', 'Parch>0 * CabinType=C', 'Parch>0 * CabinType=D', 'Parch>0 * CabinType=E', 'Parch>0 * CabinType=F', 'Parch>0 * CabinType=G', 'Parch>0 * CabinType=None', 'Parch>0 * CabinType=T', 'Embarked=C * CabinType=A', 'Embarked=C * CabinType=B', 'Embarked=C * CabinType=C', 'Embarked=C * CabinType=D', 'Embarked=C * CabinType=E', 'Embarked=C * CabinType=F', 'Embarked=C * CabinType=G', 'Embarked=C * CabinType=None', 'Embarked=C * CabinType=T', 'Embarked=None * CabinType=A', 'Embarked=None * CabinType=B', 'Embarked=None * CabinType=C', 'Embarked=None * CabinType=D', 'Embarked=None * CabinType=E', 'Embarked=None * CabinType=F', 'Embarked=None * CabinType=G', 'Embarked=None * CabinType=None', 'Embarked=None * CabinType=T', 'Embarked=Q * CabinType=A', 'Embarked=Q * CabinType=B', 'Embarked=Q * CabinType=C', 'Embarked=Q * CabinType=D', 'Embarked=Q * CabinType=E', 'Embarked=Q * CabinType=F', 'Embarked=Q * CabinType=G', 'Embarked=Q * CabinType=None', 'Embarked=Q * CabinType=T', 'Embarked=S * CabinType=A', 'Embarked=S * CabinType=B', 'Embarked=S * CabinType=C', 'Embarked=S * CabinType=D', 'Embarked=S * CabinType=E', 'Embarked=S * CabinType=F', 'Embarked=S * CabinType=G', 'Embarked=S * CabinType=None', 'Embarked=S * CabinType=T']
removed_feat = []
perma_test_acc = 0.7877237851662404

#''''
def from_regress_out_to_survival_predict (output) :
    if output < 0.5 :
        return 0 
    return 1

def find_accuracy (predict, actual) :
    num_correct = 0
    num_incorrect = 0
    for i in range(len(predict)) :
        if predict[i] == actual[i] :
            num_correct += 1
        else :
            num_incorrect+=1 
    return num_correct/(num_correct + num_incorrect)

index = 0
for key in feature_list :
    temp_terms = feature_list.copy()
    temp_terms.remove(key)
    temp_df = df[['Survived'] + temp_terms]

    df_train = temp_df[:500]
    df_test = temp_df[500:]

    arr_train = np.array(df_train)
    arr_test = np.array(df_test)

    y_train = arr_train[:,0]
    y_test = arr_test[:,0]

    x_train = arr_train[:,1:]
    x_test = arr_test[:,1:]

    regressor = LogisticRegression(max_iter=100, random_state=0)
    regressor.fit(x_train, y_train)

    y_test_predictions = regressor.predict(x_test)
    y_test_predictions = [from_regress_out_to_survival_predict(output) for output in y_test_predictions]
    y_test_acc = find_accuracy(y_test_predictions, y_test)

    
    if y_test_acc >= perma_test_acc :
        feature_list.remove(key)
        perma_test_acc = y_test_acc
        keep = 'remove'
        removed_feat.append(index)
        
    else :
        keep = 'kept'

    if deBug :
        print('Possible Remove:', key, '(index {})'.format(index))
        print('Testing:', y_test_acc)
        print('Current:', perma_test_acc)
        print(keep)
        print('Removed ids:', removed_feat, '\n')
    index += 1

print('\n\n')
print(feature_list)
print(removed_feat)
print('Testing:', perma_test_acc)
#'''