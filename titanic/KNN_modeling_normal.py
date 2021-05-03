import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from timeit import default_timer as timer
start_time = timer()
print(timer())
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

survived = df['Survived']

def find_accuracy (predict, actual) :
    num_correct = 0
    num_incorrect = 0
    for i in range(len(predict)) :
        if predict[i] == actual[i] :
            num_correct += 1
        else :
            num_incorrect+=1 
    return num_correct/(num_correct + num_incorrect)

#k_size = [1,3,5,10,15,20,30,40,50,75,100,150,200,300,400,600,800]

##Unscaled
#'''
acc_un = []

predict_un = {k : [] for k in k_size}
for row_id in range(survived.size) :
    mod_df = df.drop(row_id)
    mod_survived = survived.drop(row_id)
    for neigh_numb in k_size :
        KNN = KNeighborsClassifier(n_neighbors=neigh_numb)
        KNN.fit(mod_df, mod_survived)
        predict_un[neigh_numb].append(KNN.predict([df.iloc[row_id]]))

for predictions in predict_un.values() :
    acc_un.append(find_accuracy(predictions,survived))
#'''
print('Fin Un')
##Simple scaling
'''
df_simple = df.copy()
df_simple['Pclass'] = df['Pclass']/df['Pclass'].max()
df_simple['Age'] = df['Age']/df['Age'].max()
df_simple['SibSp'] = df['SibSp']/df['SibSp'].max()
df_simple['Fare'] = df['Fare']/df['Fare'].max()

acc_simp = []

predict_simp = {k : [] for k in k_size}

for row_id in range(survived.size) :
    mod_df = df_simple.drop(row_id)
    mod_survived = survived.drop(row_id)
    for neigh_numb in k_size :
        KNN = KNeighborsClassifier(n_neighbors=neigh_numb)
        KNN.fit(mod_df, mod_survived)
        predict_simp[neigh_numb].append(KNN.predict([df_simple.iloc[row_id]]))

for predictions in predict_simp.values() :
    acc_simp.append(find_accuracy(predictions,survived))
#'''
print('fin simp')
##Min-max
'''
df_min_max = df.copy()
df_min_max['Pclass'] = (df['Pclass']-df['Pclass'].min())/(df['Pclass'].max() - df['Pclass'].min())
df_min_max['Age'] = (df['Age']-df['Age'].min())/(df['Age'].max() - df['Age'].min())
df_min_max['SibSp'] = (df['SibSp']-df['SibSp'].min())/(df['SibSp'].max() - df['SibSp'].min())
df_min_max['Fare'] = (df['Fare']-df['Fare'].min())/(df['Fare'].max() - df['Fare'].min())

acc_min_max = []

predict_min_max = {k : [] for k in k_size}

for row_id in range(survived.size) :
    mod_df = df_min_max.drop(row_id)
    mod_survived = survived.drop(row_id)
    for neigh_numb in k_size :
        KNN = KNeighborsClassifier(n_neighbors=neigh_numb)
        KNN.fit(mod_df, mod_survived)
        predict_min_max[neigh_numb].append(KNN.predict([df_min_max.iloc[row_id]]))

for predictions in predict_min_max.values() :
    acc_min_max.append(find_accuracy(predictions,survived))
#'''
print('fin min max')
##z-score
'''
df_z = df.copy()
df_z['Pclass'] = (df['Pclass']-df['Pclass'].mean())/(df['Pclass'].std())
df_z['Age'] = (df['Age']-df['Age'].mean())/(df['Age'].std())
df_z['SibSp'] = (df['SibSp']-df['SibSp'].mean())/(df['SibSp'].std())
df_z['Fare'] = (df['Fare']-df['Fare'].mean())/(df['Fare'].std())

acc_z = []

predict_z = {k : [] for k in k_size}

for row_id in range(survived.size) :
    mod_df = df_z.drop(row_id)
    mod_survived = survived.drop(row_id)
    for neigh_numb in k_size :
        KNN = KNeighborsClassifier(n_neighbors=neigh_numb)
        KNN.fit(mod_df, mod_survived)
        predict_z[neigh_numb].append(KNN.predict([df_z.iloc[row_id]]))

for predictions in predict_z.values() :
    acc_z.append(find_accuracy(predictions,survived))
#'''
print('fin z')

#'''
import matplotlib.pyplot as plt
plt.style.use('bmh')

plt.plot(k_size, acc_un)
#plt.plot(k_size, acc_simp)
#plt.plot(k_size, acc_min_max)
#plt.plot(k_size, acc_z)
#plt.xlim((1,19))
#plt.legend(['Unaltered', 'Simple', 'Min-Max', 'Z'])

plt.savefig('KNN_un.png')

print('\n\n', timer() - start_time, 'sec \n\n\n')
#'''