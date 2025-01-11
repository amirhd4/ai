import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score

df = pd.read_csv('files/edible_oil.csv')
cdf = df[["YEAR", "PRICE"]]
viz = cdf[["YEAR", "PRICE"]]

# msk = np.random.rand(len(df)) < 0.80
# train = cdf[msk]
# test = cdf[~msk]
msk = np.random.rand(len(df)) < 1
train, test = cdf[msk], cdf[msk]

fig = plt.figure()
pic1 = fig.add_subplot(111)

regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[["YEAR"]])
train_y = np.asanyarray(train[["PRICE"]])
regr.fit(train_x, train_y)

print("Coefficient: ", regr.coef_)
print("Intercept: ", regr.intercept_)

plt.scatter(train.YEAR, train.PRICE, color="blue")
plt.plot(train_x, regr.coef_ * train_x + regr.intercept_, color="red")
plt.xlabel("Year")
plt.ylabel("Price")

test_x = np.asanyarray(test[["YEAR"]])
test_y = np.asanyarray(test[["PRICE"]])
test_y_ = regr.predict(test_x)

ys = np.array([[1404], [1405]])
res = regr.predict(np.asanyarray(ys))
print(res)
# print("score: ", r2_score(test_y, test_y_))