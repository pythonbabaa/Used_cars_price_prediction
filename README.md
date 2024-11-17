# Used_cars_price_prediction
# **Setup**
#from mlwpy import *
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
%matplotlib inline

import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# **Data Preparation**
cars_data = pd.read_csv('Span_new.csv')
cars_data.head()
cars_data = cars_data.drop_duplicates()
cars_data = cars_data.dropna()
cars_data = cars_data.drop(['Unnamed: 0', 'ID'], axis=1)
cars_data = cars_data.rename({'age': 'model_year'}, axis=1)
cars_data.shape
variables_in_study = cars_data[['months_old', 'power', 'kms', 'price']]

scaler = StandardScaler()
scaler.fit(variables_in_study)
variables_in_study = scaler.transform(variables_in_study)

variables_in_study = pd.DataFrame(variables_in_study, columns=['months_old', 'power', 'kms', 'price'])
independent_variables = variables_in_study[['months_old', 'power', 'kms']]
dependent_variable = variables_in_study['price']
variables_in_study.head()
independent_variables.head()
dependent_variable.head()
x_train, x_test, y_train, y_test = train_test_split(independent_variables, dependent_variable, test_size=0.2)
MAE_list = []
MSE_list = []
RMSE_list = []
R_Squared_list = []

# **Model Evaluation**
def model_evaluation(y_test, predictions):
    mae = metrics.mean_absolute_error(y_test, predictions)
    mse = metrics.mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y_test, predictions)
    
    MAE_list.append(mae)
    MSE_list.append(mse)
    RMSE_list.append(rmse)
    R_Squared_list.append(r2)
    
    print("Results of sklearn.metrics: \n")
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R-Squared:", r2)

# **Linear Regression**
regression_model = sm.OLS(y_train, x_train)
results = regression_model.fit()
results.summary()
print(results.rsquared)
print(results.rsquared_adj)
print(results.pvalues)
print(results.params)
predictions = results.predict(x_test)
model_evaluation(y_test, predictions)

# **Polynomial Regression**
poly_reg = PolynomialFeatures(degree=3)
x_poly = poly_reg.fit_transform(x_train)
poly_reg.fit(x_poly, y_train)
lin_reg = LinearRegression()
lin_reg.fit(x_poly, y_train)
predictions = lin_reg.predict(poly_reg.fit_transform(x_test))
model_evaluation(y_test, predictions)

# **Support Vector Regression**
from sklearn.svm import SVR
svr = SVR()
svr.fit(x_train, y_train)
predictions = svr.predict(x_test)
model_evaluation(y_test, predictions)

# **Decision Tree Regression**
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(x_train, y_train)
predictions = dtr.predict(x_test)
model_evaluation(y_test, predictions)

# **Random Forest Regression**
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(x_train, y_train)
predictions = rfr.predict(x_test)
model_evaluation(y_test, predictions)

# **Ridge Regression**
rdg = Ridge(alpha=0.5)
rdg = rdg.fit(x_train, y_train)
predictions = rdg.predict(x_test)

# **Lasso Regression**
lasso = Lasso(alpha=0.01)
lasso = lasso.fit(x_train, y_train)
prediction = lasso.predict(x_test)

# **KNN Regression**
knn = neighbors.KNeighborsRegressor(n_neighbors=3)
fit = knn.fit(x_train, y_train)
predictions = fit.predict(x_test)

# **Results and Visualization**
models_list = ['Multiple Linear Regression', 'Polynomial Regression', 'Support Vector Regression', 'Decision Tree Regression', 'Random Forest Regression', 'Ridge Regression', 'Lasso Regression', 'KNN Regression']
sns.set_style('whitegrid')

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))
ax1.plot(MAE_list, color='red', label='MAE', marker="o")
ax1.plot(MSE_list, color='blue', label='MSE', marker="o")
ax1.plot(RMSE_list, color='green', label='RMSE', marker="o")
ax2.plot(R_Squared_list, color='green', label='R-Squared', marker="o")
ax1.legend()
ax2.legend()
ax1.set_xticks(ticks=range(len(MAE_list)))
ax2.set_xticks(ticks=range(len(R_Squared_list)))
ax1.set_xticklabels(models_list, rotation=90)
ax2.set_xticklabels(models_list, rotation=90)
ax1.set_xlabel('Regression Models', labelpad=20)
ax2.set_xlabel('Regression Models', labelpad=20)
ax1.set_ylabel('MAE, MSE, RMSE')
ax2.set_ylabel('R-Squared')
plt.show()

print(np.min(RMSE_list))
print(np.argmin(RMSE_list))
print(models_list[np.argmin(RMSE_list)])
print(np.max(R_Squared_list))
print(np.argmax(R_Squared_list))
print(models_list[np.argmax(R_Squared_list)])

