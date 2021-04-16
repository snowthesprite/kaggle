import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import sys

df = pd.read_csv('/home/runner/kaggle/quiz_5/StudentsPerformance.csv')
df = df[['math score', "test preparation course"]]

''''

m_score = list(df['math score'])

print('Last 3 scores:', m_score[-3:], '\n')

print('Avg math score:', sum(m_score)/len(m_score))

#'''
#''''

def completed_to_true(completed) :
    if completed == 'none' :
        return 0
    elif completed == 'completed' :
        return 1

math_prep = df.copy()[['math score', "test preparation course"]]

math_prep["test preparation course"] = math_prep["test preparation course"].apply(completed_to_true)

print(math_prep)

prep_true = []
prep_false = []





'''
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

#print('\n', feature_coeffs, '\n')

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



#print("\n",'features:', used_features)
#print('train:', find_accuracy(y_train_predictions, y_train))
#print('test:', round(find_accuracy(y_test_predictions, y_test),4), "\n")

print(coeff_dict)
#'''