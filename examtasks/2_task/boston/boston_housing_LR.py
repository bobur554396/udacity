from sklearn.datasets import load_diabetes
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


boston = load_boston()

# ////////////////////////////////////////////////////////////////////////////////////////////
model = LinearRegression()

model.fit(boston.data, boston.target)

expected = boston.target
predicted = model.predict(boston.data)

print "\n\n\n"
print "Linear regression model \n Boston dataset"
print "Mean squared error = %0.3f" % mse(expected, predicted)
print "R2 score = %0.3f" % r2_score(expected, predicted)


# ////////////////////////////////////////////////////////////////////////////////////////////
from sklearn.ensemble import RandomForestRegressor

model.fit(boston.data, boston.target)

expected = boston.target
predicted = model.predict(boston.data)

print "\n\n\n"
print "Random Forest model \n Boston dataset"
print "Mean squared error = %0.3f" % mse(expected, predicted)
print "R2 score = %0.3f" % r2_score(expected, predicted)


# ////////////////////////////////////////////////////////////////////////////////////////////
from sklearn.linear_model import Ridge

model = Ridge(alpha=0.1)
model.fit(boston.data, boston.target)

expected = boston.target
predicted = model.predict(boston.data)

print "Ridge regression model \n Boston dataset"
print "Mean squared error = %0.3f" % mse(expected, predicted)
print "R2 score = %0.3f" % r2_score(expected, predicted)