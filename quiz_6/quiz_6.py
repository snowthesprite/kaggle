import pandas as pd
#import numpy as np

df = pd.read_csv('/home/runner/kaggle/quiz_6/students.csv')

#''''

m_score = df['training_hours']

print('Avg training hours:', m_score.mean())

target = df['target']
searching = 0
for search in target :
    if search == 1 :
        searching += 1

print('Percentage Looking for New Job:', searching/target.size)

city_data = df['city']
city_pop = {city : 0 for city in city_data.unique()}

for living in city_data :
    city_pop[living] += 1

highest_pop = [0,0]

for (city, pop) in city_pop.items() :
    if pop > highest_pop[1] :
        highest_pop[0] = city
        highest_pop[1] = pop 

print('City w/ highest pop:', highest_pop[0], 'Pop:', highest_pop[1])

highest_id = 0

for city in city_pop :
    id = int(city.split('_')[1])
    if id > highest_id :
        highest_id = id

print('Highest city ID:', highest_id)

comp_employee = df['company_size']

less_10 = 0
less_100 = 0

for num in comp_employee :
    #print(num)
    if type(num) != float :
        if '-' in num :
            num = num.split('-')[1]
        elif '<' in num :
            num = num.split('<')[1]
            num = float(num) -1
        elif '+' in num :
            continue
        elif '/' in num :
            num = num.split('/')[0]
        num = float(num)
    if num < 10 :
        less_10 += 1
        less_100 += 1
    elif num < 100 :
        less_100 += 1

print('Students from companies w/ less than 10 employees:',less_10)
print('Students from companies w/ less than 100 employees:',less_100)


#'''