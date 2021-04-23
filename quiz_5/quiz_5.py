import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('/home/runner/kaggle/quiz_5/StudentsPerformance.csv')
df = df[['math score', "test preparation course", 'parental level of education']]

#''''

m_score = df['math score']

print('Last 3 scores:\n', m_score[-3:], '\n')

print('Avg math score:', m_score.mean())

#'''
#''''

def completed_to_true(completed) :
    if completed == 'none' :
        return 0
    elif completed == 'completed' :
        return 1

df["test preparation course"] = df["test preparation course"].apply(completed_to_true)

#print(math_prep)

prep_true = (df['test preparation course'] == 1)
prep_false = (df['test preparation course'] == 0)

print('Avg math score w/ test prep:', m_score[prep_true].mean())
print('Avg math score w/o test prep:', m_score[prep_false].mean())
print('# of parent education levels :', len(df['parental level of education'].unique()))

for ed_lvl in df['parental level of education'].unique() :
    dummy_vals = df['parental level of education'].apply(lambda entry : int(entry == ed_lvl))
    df[ed_lvl] = dummy_vals

del df['parental level of education']

#'''
df_train = df[:-3]
df_test = df[-3:]

arr_train = np.array(df_train)
arr_test = np.array(df_test)

y_train = arr_train[:,0]
y_test = arr_test[:,0]

x_train = arr_train[:,1:]
x_test = arr_test[:,1:]

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_test_predictions = regressor.predict(x_test)
y_train_predictions = regressor.predict(x_train)

print('Predicted Values:', y_test_predictions)

#'''