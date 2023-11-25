'''
Video 20-22

Intro to Support Vector Machines
Vectors EX: Ā = [3,4]
the magnitude of a vector can be defined
SQRT of (Ā[0]**2 + Ā[1]**2)
Magnitude of Ā or ||Ā|| = 5


dot products of vectors would look like:
A = [1,3]
B = [4,2]
A . B = (1 * 4) + (3 * 2)
A . B = 10



'''

import numpy as np
from sklearn import preprocessing, neighbors, svm
from sklearn.model_selection import train_test_split
import pandas as pd


df = pd.read_csv('./breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(columns=['id'], axis=1, inplace=True)

X = np.array(df.drop(columns=['class'], axis=1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = svm.SVC()

clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)
# example_measures = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,2,2,2,3,2,1]])
# example_measures = example_measures.reshape(len(example_measures), -1)

# prediction = clf.predict(example_measures)
# print(prediction)

