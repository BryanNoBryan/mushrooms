#ML and plotting
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
#One hot encoding
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
#import pandas
import pandas as pd
#to split the array
import numpy as np

#Auxillary
import math
from collections import OrderedDict

#pass into column of class
def calc_entropy(data: pd.Series) -> int:
    terms = {}
    sum = 0
    #dictionary containing times each type exists
    for i in range(data.size):
        sum += 1
        term = data.iloc[i]
        if (term in terms):
            terms[term] = terms[term] + 1
        else:
            terms[term] = 1

    #formula
    entropy = 0
    for key in terms:
        entropy += (terms[key]/sum) * math.log(terms[key]/sum)
    entropy = -entropy
    return entropy

#pass in column of attribute + class
def entropy_attribute(df: pd.DataFrame) -> int:
    print('attribute calc')
    terms = {}
    sum = 0
    #obtain subsets(as type DataFrame) filtered by attribute into terms
    for i in range(df.index.size):
        sum += 1
        term = df.iloc[[i]]
        type = term.iloc[0].iloc[0]
        if (type in terms):
            terms[type] = pd.concat([terms[type], term])
        else:
            terms[type] = term

    #formula
    entropy = 0
    for key in terms:
        subset = terms[key]
        #entropy of edible or not category
        entropy_subset = calc_entropy(subset.iloc[:, 1])
        entropy += (subset.index.size/sum) * entropy_subset
    return entropy

#pass in column of attribute + class
def info_gain(df: pd.DataFrame) -> int:
    entropy = calc_entropy(df.iloc[:, 1])
    attr_entropy = entropy_attribute(df)
    return entropy - attr_entropy

#pass in column of class + attribute(all)
def info_gain_all(df: pd.DataFrame) -> dict[str, float]:
    gains = []
    #iterate through attributes + class
    for i in range(1, df.columns.size):
        gain = info_gain(df.iloc[:, [i, 0]])

        print(f'{df.columns[i]} Gain: {gain}')
        #insert into gain(ordered)
        for j in range(len(gains)):
            if (gain > gains[j][1]):
                gains.insert(j, [df.columns[i], gain])
                break
        else:
            gains.append([df.columns[i], gain])
    return gains


###START

#load the dataset
df = pd.read_csv('mushrooms.csv',na_values=["?"])

#drops all rows that have a missing element '?' inside it
df.dropna(inplace = True)

###START CALCULATIONS
entropy = calc_entropy(df['class'])
print(f'class_entropy: {entropy}')
# print(df[['cap-shape', 'class']])
# attribute = entropy_attribute(df[['cap-shape', 'class']])
# print(attribute)
# gain = info_gain(df[['cap-shape', 'class']])
# print(gain)
print('Leave this running, takes a long while on replit')
gains = info_gain_all(df)
print('GAINS: ')
print(*gains, sep='\n')
###END CALCULATIONS

#One hot encode the dataset
df = pd.get_dummies(df, dtype=int)
#drop poisonous column, we can infer poisonous from edible
df = df.drop(df.columns[[1]], axis=1)

#break dataset into training and sample data
len = len(df)
sample_size = 10
tdf = df.iloc[:(len-sample_size)]
sdf = df.iloc[(len-sample_size):]
#Note: I know I can use train_test_split, but this lets me understand how pands interacts with Scikit

#labels: 1 edible, 0 poisonous
y = tdf['class_e']
#get training data, exclude labels
X = tdf[tdf.columns[1:]]

dt = tree.DecisionTreeClassifier()
#train the model
#X is dataframe with labels, X.values is dataframe alone
#I want to separate them to access them easier later
dt.fit(X.values, y)

#sample correct answer
correct_answers = sdf['class_e']
#sample data-to-test
test_data = sdf[sdf.columns[1:]]

print('TEST CASES CA=Correct Answer, Q: Model Answer')
#range equivalent to var sample_size
for i in range(test_data.index.size):
    #[test_data.iloc[i]] b/c we need 2D array i.e. Dataframe not a Series
    #()[0] as the return type is a numpy list
    result = int(dt.predict([test_data.iloc[i]])[0])
    answer = int(correct_answers.iloc[i])
    print('CA: ' + str(answer) + ' Q: ' + str(result))


fig = plt.figure(figsize=(25,25))
tree.plot_tree(dt,feature_names = X.columns, rounded = True, filled = True) 
fig.savefig('tree.png')
print('tree.png saved')