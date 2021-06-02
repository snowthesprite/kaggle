import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('final.csv')

del df['Id']

print(df.groupby(['Species']).mean())

df = df.sample(frac=1).reset_index(drop=True)

species = df['Species']
del df['Species']

df_normal = df.copy()

for column in list(df.columns) :
    df_normal[column] = (df[column]-df[column].min())/(df[column].max() - df[column].min())

acc = []
x_train = df_normal[:75]

y_train = species[:75]

x_test = np.array(df_normal[75:])
y_test = np.array(species[75:])



def find_accuracy (predict, actual) :
    num_correct = 0
    num_incorrect = 0
    for i in range(len(predict)) :
        if predict[i] == actual[i] :
            num_correct += 1
        else :
            num_incorrect+=1 
    return num_correct/(num_correct + num_incorrect)



for neigh_numb in range(1, y_train.size) :
    KNN = KNeighborsClassifier(n_neighbors=neigh_numb)
    KNN.fit(x_train, y_train)
    predictions = []
    for predict in x_test :
        print(predict)
        predictions.append(KNN.predict(predict))
    acc.append(find_accuracy(predictions,y_test))
    
index = acc.index(max(acc))

print('k:', index + 1)
print("best test:", max(acc))