'''
# Machine Learning pt3

forecast_out = int(math.ceil(0.1*len(df)))
this is grabing how many days out we want to
predict. it is rounding with math.ceil
turning to an INT and this is 10 days
for 1 day .01

'''
import pandas as pd
import quandl
import math

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forcast_col = 'Adj. Close'

df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.1*len(df)))


df['label'] = df[forcast_col].shift(-forecast_out)

df.dropna(inplace=True)
print(df.head)