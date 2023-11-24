'''
# Machine Learning pt4

when browsing algorithms
for predictions, check sklearn 
website and check n_jobs to 
determine how many threads can
run on this algorithm

default for regression = 1 (Liner)

can be specified: 
clf = LinearRegression(n_jobs=10)

clf = LinearRegression(n_jobs=-1)
**moves as fast as processor will allow
'''
import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forcast_col = 'Adj. Close'

df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.00025*len(df)))

print('-'*50)
print("Forcast Days:")
print(forecast_out)
print('-'*50)

df['label'] = df[forcast_col].shift(-forecast_out)

df.dropna(inplace=True)


X = np.array(df.drop(['label'], axis=1))
y = np.array(df['label'])

X = preprocessing.scale(X)

y = np.array(df['label'])


X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
# clf = svm.SVR()
# Fit is synonymous with train
clf.fit(X_train, y_train)

# Score is synonymous with test
accuracy = clf.score(X_test, y_test)
print(accuracy)