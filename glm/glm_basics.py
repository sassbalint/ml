import numpy as np
from sklearn.linear_model import TweedieRegressor

POWER = 0
ALPHA = 0

np.set_printoptions(precision=2)

def regression(linkfunc, x, y, test):

    reg = TweedieRegressor(power=POWER, alpha=ALPHA, link=linkfunc)

    # reshaping when there is only 1 feature (= dependent variable)
    if len(x.shape) == 1:
        x = x.reshape(-1, 1) # ! convert to "column" vector
                             #   as data should be in rows
        test = test.reshape(-1, 1)

    variable_cnt = x.shape[1]

    plur = "s" if variable_cnt > 1 else ""
    print()
    print('generalized linear regression parametrized as')
    print(f' -- link = \'{linkfunc}\'')
    print(f' -- {variable_cnt} dependent variable{plur}')
    print(reg)

    print()
    print(f'train: {x} -> {y}')
    print(f'test: {test} -> ???')

    reg.fit(x, y)

    predicted = reg.predict(test)

    print()
    print('predicted:')
    print(predicted)

    print()
    print('y = reg.coef_ * x + reg.intercept_')
    print(f'reg.coef_ = {reg.coef_}')
    print(f'reg.intercept_ = {reg.intercept_:.2f}')
    for t in test:
        x_val = t
        y_val = reg.coef_ * t + reg.intercept_
        print(f'{x_val} -> {y_val}')

    strs = []
    if variable_cnt > 1:
        strs.append('sum')
    if linkfunc != 'identity':
        strs.append(f'inverse of link function \'{linkfunc}\'')
    basic_str = 'to be applied!'

    if len(strs) > 0:
        print(' and '.join(strs), basic_str)

    print()
    print(f'n_iter_ = {reg.n_iter_}')

    print()


print("""
Linear regression and generalized linear models (GLM) with sklearn

First: https://en.wikipedia.org/wiki/Generalized_linear_model#Intuition

I guess TweedieRegressor is a fully general solution,
to which every regression type can be traced back.
In other words, every regression type can be obtained
as a specific parametrization of TweedieRegressor.
Not completely sure... Let\'s see some examples.

No idea yet how to do logistic regression
as there is no 'logit' value to the link parameter. Hm..
https://stackoverflow.com/questions/66390041
""")

# 1 dependent variable
x1 = np.array([0, 2, 4])
y1 = np.array([0, 2, 5])
test1 = np.array([-10, 0, 1, 2, 4, 10])

# 2 dependent variables
x2 = np.array([[0, 0], [1, 1], [2, 3]])
y2 = np.array([0, 2, 5])
test2 = np.array([[0, 0], [1, 1], [4, 5]])

print()
print('===== simple linear regression (one variable)')
regression('identity', x1, y1, test1)
print('===== multivariable linear regression (two variables)')
regression('identity', x2, y2, test2)

print('===== log-linear regression with one variable')
regression('log', x1, y1, test1)
print('===== log-linear regression with two variables')
regression('log', x2, y2, test2)

