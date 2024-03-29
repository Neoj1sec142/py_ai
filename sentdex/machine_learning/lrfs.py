'''
Linear Regression from Scratch
Best fit slope 8-12
W/ Testing
'''
from statistics import mean
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

# xs = np.array([1,2,3,4,5,6], dtype=np.float64)
# ys = np.array([5,4,6,5,6,7], dtype=np.float64)

def create_dataset(hm, variance, step=2, correlation=False):
    '''
    Change variance to test
    influence of regression
    '''
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

def best_fit_slope_and_intercept(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
         ((mean(xs) * mean(xs)) - mean(xs**2)))
    b = mean(ys) - m*mean(xs)
    return m, b

def squared_error(ys_orig, ys_line):
    '''
    The squared error is the distance
    of y from the line squared.
    '''
    return sum((ys_line - ys_orig)**2)

def coefficient_of_determination(ys_orig, ys_line):
    '''
    R Theroy 
    1 - the squared error of the
    regression line divided by squared
    error of the y mean line
    '''
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_err_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_err_y_mean)

xs, ys = create_dataset(40, 10, 2, correlation='pos')

m,b = best_fit_slope_and_intercept(xs, ys)
# print(m,b)

regression_line = [ (m*x)+b for x in xs ]

predict_x = 8
predict_y = (m*predict_x)+b

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)




plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, s=100, color='red')
plt.plot(xs, regression_line)
plt.show()