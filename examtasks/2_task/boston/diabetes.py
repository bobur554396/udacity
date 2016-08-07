from sklearn.datasets import load_diabetes
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


diabetes = load_diabetes()

# ////////////////////////////////////////////////////////////////////////////////////////////
model = LinearRegression()
model.fit(diabetes.data, diabetes.target)

expected = diabetes.target
predicted = model.predict(diabetes.data)

print "\n\n\n"
print "Linear regression model \n Diabetes dataset"
print "Mean squared error = %0.3f" % mse(expected, predicted)
print "R2 score = %0.3f" % r2_score(expected, predicted)


# ////////////////////////////////////////////////////////////////////////////////////////////
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(diabetes.data, diabetes.target)

expected = diabetes.target
predicted = model.predict(diabetes.data)

print "\n\n\n"
print "Random Forest model \n Diabetes dataset"
print "Mean squared error = %0.3f" % mse(expected, predicted)
print "R2 score = %0.3f" % r2_score(expected, predicted)


# ////////////////////////////////////////////////////////////////////////////////////////////
from sklearn.linear_model import Ridge

model = Ridge(alpha=0.1)
model.fit(diabetes.data, diabetes.target)

expected = diabetes.target
predicted = model.predict(diabetes.data)

print "\n\n\n"
print "Ridge regression model \n Diabetes dataset"
print "Mean squared error = %0.3f" % mse(expected, predicted)
print "R2 score = %0.3f" % r2_score(expected, predicted)