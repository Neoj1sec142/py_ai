'''
Classification - goal is to create
a model that separates or divides
our data
'''
import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd

accuracies = []

for i in range(25):
    df = pd.read_csv('./breast-cancer-wisconsin.data')
    df.replace('?', -99999, inplace=True)
    df.drop(columns=['id'], axis=1, inplace=True)

    X = np.array(df.drop(columns=['class'], axis=1))
    y = np.array(df['class'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = neighbors.KNeighborsClassifier()

    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    # print(accuracy)
    # example_measures = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,2,2,2,3,2,1]])
    # example_measures = example_measures.reshape(len(example_measures), -1)

    # prediction = clf.predict(example_measures)
    # print(prediction)
    
    accuracies.append(accuracy)
print(sum(accuracies)/len(accuracies))