## For the times I got: 2.06, 2.37, 0.56, 1.33, 1.58, for an average of 1.58, relative speed of 10.54, and a predicted time of 474.18 sec
import time
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

start = time.time()

df = pd.read_csv('/home/runner/kaggle/titanic/processed_data.csv')

keep_cols = ['Survived', 'Sex', 'Pclass', 'Fare', 'Age', 'SibSp']

df = df[keep_cols]
df = df[:100]

survived = df['Survived']
del df['Survived']
columns = list(df.columns)

def find_accuracy (predict, actual) :
    num_correct = 0
    for i in range(len(predict)) :
        if predict[i] == actual[i] :
            num_correct += 1
    return num_correct/len(predict)

k_vals = range(1,100,2)

df_simp = df.copy()
df_min_max = df.copy()
df_z = df.copy()

for column in columns :
    df_simp[column] = df[column]/df[column].max()

    df_min_max[column] = (df[column]-df[column].min())/(df[column].max() - df[column].min())

    df_z[column] = (df[column]-df[column].mean())/(df[column].std())

acc_un = []
predict_un = {k : [] for k in k_vals}

acc_simp = []
predict_simp = {k : [] for k in k_vals}

acc_min_max = []
predict_min_max = {k : [] for k in k_vals}

acc_z = []
predict_z = {k : [] for k in k_vals}

for row_id in range(survived.size) :
    mod_df = df.drop(row_id)
    mod_df_simp = df_simp.drop(row_id)
    mod_df_min = df_min_max.drop(row_id)
    mod_df_z = df_z.drop(row_id)

    mod_survived = survived.drop(row_id)
    for k in k_vals :
        KNN = KNeighborsClassifier(n_neighbors=k)

        KNN.fit(mod_df, mod_survived)
        predict_un[k].append(KNN.predict([df.iloc[row_id]]))

        KNN.fit(mod_df_simp, mod_survived)
        predict_simp[k].append(KNN.predict([df_simp.iloc[row_id]]))

        KNN.fit(mod_df_min, mod_survived)
        predict_min_max[k].append(KNN.predict([df_min_max.iloc[row_id]]))

        KNN.fit(mod_df_z, mod_survived)
        predict_z[k].append(KNN.predict([df_z.iloc[row_id]]))

for k in k_vals :
    acc_un.append(find_accuracy(predict_un[k],survived))
    acc_simp.append(find_accuracy(predict_simp[k],survived))
    acc_min_max.append(find_accuracy(predict_min_max[k],survived))
    acc_z.append(find_accuracy(predict_z[k],survived))

end = time.time()
print(end - start)
#It appears to be spiting out either like ~430 sec or ~250 sec, not really any in between
#Just got a time for 645 sec, no idea where that came from
#'''
import matplotlib.pyplot as plt
#plt.style.use('bmh')

plt.plot(k_vals, acc_un)
plt.plot(k_vals, acc_simp)
plt.plot(k_vals, acc_min_max)
plt.plot(k_vals, acc_z)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.legend(['Unaltered', 'Simple', 'Min-Max', 'Z'])
plt.title('Leave One Out Cross Validation')

plt.savefig('titanic/KNN_Modeling/KNN_normalized.png')
#'''