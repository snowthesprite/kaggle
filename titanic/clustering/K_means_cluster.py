import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv('/home/runner/kaggle/titanic/processed_data.csv')

survived = df['Survived']
cols = ['Sex', 'Pclass', 'Fare', 'Age', 'SibSp']
df = df[cols]

normal_df = df.copy()
for column in cols :
    normal_df[column] = (df[column]-df[column].min())/(df[column].max() - df[column].min())



'''
k_num = range(1,26)

all_errors = []

for num_k in k_num :
    kmeans = KMeans(n_clusters=num_k).fit(df)
    all_errors.append(kmeans.inertia_)

import matplotlib.pyplot as plt
#plt.style.use('bmh')

plt.plot(k_num, all_errors)
plt.xticks(k_num)
plt.xlabel("k")
plt.ylabel("Sum Squared Error")
plt.title('KMeans Clustering w/ the Titanic')

plt.savefig('titanic/clustering/KMeans_elbow.png')
#'''



kmeans = KMeans(n_clusters = 4).fit(normal_df)

df['cluster'] = kmeans.labels_
df['Survived'] = survived

print(df.groupby(['Sex','Pclass']).count())
print('\n\n')

group_df = df.groupby(['cluster']).mean()
group_df['count'] = df.groupby(['cluster']).count()['Sex']

print(group_df)
print(group_df['Age'])

