import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('/home/runner/kaggle/titanic/processed_data.csv')

keep_cols = ['Survived', 'Sex', 'Pclass', 'Fare', 'Age', 'SibSp']

df = df[keep_cols]
df = df[:100]

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

k_size = [1,3,5,10,15,20,30,40,50,75]

#'''
acc = []
predict = {k : [] for k in k_size}

for row_id in range(survived.size) :
    mod_df = df.drop(row_id)
    mod_survived = survived.drop(row_id)
    for neigh_numb in k_size :
        KNN = KNeighborsClassifier(n_neighbors=neigh_numb)
        KNN.fit(mod_df, mod_survived)
        predict[neigh_numb].append(KNN.predict([df.iloc[row_id]]))

for predictions in predict.values() :
    acc.append(find_accuracy(predictions,survived))
#'''

'''
import matplotlib.pyplot as plt
plt.style.use('bmh')

plt.plot(k_size, acc)

plt.xlabel('K')
plt.ylabel('Accuracy')
plt.title('Leave One Out Cross Validation')

plt.savefig('titanic/KNN_Modeling/KNN_titanic.png')
#'''