import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import sys

df = pd.read_csv('/home/runner/kaggle/titanic/dataset.csv')

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

terms = list(df.columns)[1:]

#make interaction terms
for col_1 in terms :
    for col_2 in terms :
        interaction = col_1+'*'+col_2
        no_inter = col_1[:-1] not in col_2 and col_2[:-1] not in col_1
        not_in = interaction not in list(df.columns) and col_2+'*'+col_1 not in list(df.columns)
        if no_inter and not_in :
            df[interaction] = df[col_1] * df[col_2]

all_terms = list(df.columns)

perma_terms = ['Survived']
perma_test_acc = 0
perma_train_acc = 0

#print(all_terms['Sex'])
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

while True :
    temp_test_acc = 0
    temp_train_acc = 0
    for key in all_terms :
        if key in perma_terms :
            continue
    
        temp_df = df.copy()[perma_terms + [key]]

        df_train = temp_df[:500]
        df_test = temp_df[500:]

        arr_train = np.array(df_train)
        arr_test = np.array(df_test)

        y_train = arr_train[:,0]
        y_test = arr_test[:,0]

        x_train = arr_train[:,1:]
        x_test = arr_test[:,1:]

        #print('\n\n',x_train,'\n\n', x_test, '\n\n')

        regressor = LogisticRegression(max_iter=10)
        regressor.fit(x_train, y_train)

        y_test_predictions = regressor.predict(x_test)
        y_train_predictions = regressor.predict(x_train)
        y_test_predictions = [from_regress_out_to_survival_predict(output) for output in y_test_predictions]
        y_train_predictions = [from_regress_out_to_survival_predict(output) for output in y_train_predictions]

        y_train_acc = find_accuracy(y_train_predictions, y_train)
        y_test_acc = find_accuracy(y_test_predictions, y_test)

        if y_train_acc > temp_train_acc or y_test_acc > temp_test_acc :
            possible_term = key
            temp_train_acc = y_train_acc
            temp_test_acc = y_test_acc
    if temp_train_acc < perma_train_acc or temp_test_acc < perma_test_acc :
        break
    perma_terms.append(possible_term)
    perma_train_acc = temp_train_acc
    perma_test_acc = temp_test_acc 

print('\n\n\n')
print(perma_terms)
print('Training:', perma_train_acc)
print('Testing:', perma_test_acc)
#'''

''''
features_used = list(df.columns)[1:]
df = df[['Survived']+features_used]

df_train = df[:500]
df_test = df[500:]

arr_train = np.array(df_train)
arr_test = np.array(df_test)

y_train = arr_train[:,0]
y_test = arr_test[:,0]

x_train = arr_train[:,1:]
x_test = arr_test[:,1:]

regressor = LogisticRegression(max_iter=1000)
regressor.fit(x_train, y_train)

coeff_dict = {'Constant' : round(regressor.intercept_[0],4)}
feature_cols = df_train.columns[1:]
feature_coeffs = regressor.coef_[0]

for i in range(len(feature_cols)) :
    col = feature_cols[i]
    coeff = feature_coeffs[i]
    coeff_dict[col] = round(coeff,4)

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


print('\n')
#print("\n",'features:', features_used)
print('train:', round(find_accuracy(y_train_predictions, y_train),3))
print('test:', round(find_accuracy(y_test_predictions, y_test),3), "\n")

#print(coeff_dict)
#'''