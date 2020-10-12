#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 10:19:31 2020

@author: mason
"""

import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from datetime import date
from sklearn import preprocessing
from itertools import combinations
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm
from sklearn.metrics import r2_score
from xgboost import XGBRegressor


df = pd.read_csv('imdb_data.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)

# dropping all rows with nan values for gross, budget, opn_wknd, and runtime
df.drop(df[pd.isna(df['gross'])].index, inplace=True)
df.drop(df[pd.isna(df['budget'])].index, inplace=True)
df.drop(df[pd.isna(df['opn_wknd'])].index, inplace=True)
df.drop(df[pd.isna(df['runtime'])].index, inplace=True)

# dropping movies that grossed under 1 million (2 movies)
df.drop(df[df['gross'] <= 1000000].index, inplace=True)

# taking care of incorrect values for mpaa
df.loc[df[df['mpaa'] == 'for'].index, 'mpaa'] = 'PG-13'
df.loc[df[df['mpaa'] == 'PG-'].index, 'mpaa'] = 'PG-13'
df.loc[df[df['mpaa'] == 'Rated'].index, 'mpaa'] = 'PG-13'

# turning mpaa into dummies
df = pd.concat([df, pd.get_dummies(df['mpaa'])], axis=1)
df.drop(columns=['mpaa'], inplace=True)

# converting genres from string of lists to lists, then getting genre dummies
df.loc[df.index, 'genres'] = df.loc[df.index, 'genres'].apply(ast.literal_eval)
df = pd.concat([df, pd.get_dummies(df['genres'].explode()).groupby(level=0).sum()], axis=1)
df.drop(columns=['genres'], inplace=True)

df.loc[df.index, 'date'] = pd.to_datetime(df.loc[df.index, 'date'])
df['days'] = (date.today() - df['date'].dt.date).dt.days

df['Time Period'] = np.where(df['date'].dt.year < 2000, 'Before Year 2000', 'After Year 2000')
df['gross'] = df['gross'] / 100000
df['budget'] = df['budget'] / 100000
sns.relplot(data = df, x='budget', y='gross', hue='Time Period')
plt.xlabel('Budget', fontsize = 10, style = 'italic')
plt.ylabel('Gross',fontsize = 10, style = 'italic')
plt.title('Gross Domestic Sales vs. Budget',fontsize = 15,)
plt.grid()

df.drop(columns=['date'], inplace=True)

# converting directors, writers, and cast from strings of lists to lists
df.loc[df.index, 'directors'] = df.loc[df.index, 'directors'].apply(ast.literal_eval)
df.loc[df.index, 'writers'] = df.loc[df.index, 'writers'].apply(ast.literal_eval)
df.loc[df.index, 'cast'] = df.loc[df.index, 'cast'].apply(ast.literal_eval)

# dropping some incorrect values for directors
nums = []
for i in range(0, 11):
    nums.append(str(i))
[x.pop(-1) for x in df.loc[df.index, 'directors'] if x[-1].split()[0] in nums]

# getting each directors mean opn_wknd and sorting
dir_means = df.loc[df.index, ['opn_wknd', 'directors']].explode('directors').groupby('directors').mean()
dir_means.sort_values('opn_wknd', inplace=True)

dir_means['dir_rank'] = list(range(0, len(dir_means)))
dir_means.reset_index(level=0, inplace=True)

# assigning movies a rank based on their best director
df_exp = df.explode('directors')
df_merge = df_exp.merge(dir_means, left_on='directors', right_on = 'directors')
df_merge = df_merge[['title', 'opn_wknd_x', 'dir_rank']].groupby('title').max()

plt.scatter(df_merge['dir_rank'], df_merge['opn_wknd_x'])

# grouping movies based off director rank
floors = np.arange(0, 1200, 200)
for i in range(0, 6):
    df_merge.loc[df_merge[df_merge['dir_rank'] >= floors[i]].index, 'director_groups'] = i
df = df.merge(df_merge[['dir_rank']], left_on='title', right_on='title')

corr_matrix = df.corr()

# dropping some incorrect values in writers
nums = []
for i in range(0, 40):
    nums.append(str(i))    
[x.pop(-1) for x in df.loc[df.index, 'writers'] if x[-1].split()[0] in nums]

# getting each writers mean opn_wknd and sorting
wri_means = df.loc[df.index, ['opn_wknd', 'writers']].explode('writers').groupby('writers').mean()
wri_means.sort_values('opn_wknd', inplace=True)

wri_means['wri_rank'] = list(range(0, len(wri_means)))
wri_means.reset_index(level=0, inplace=True)

# assinging movies a rank based on their best writer
df_exp = df.explode('writers')
df_merge = df_exp.merge(wri_means, left_on='writers', right_on = 'writers')
df_merge = df_merge[['title', 'opn_wknd_x', 'wri_rank']].groupby('title').max()

plt.scatter(df_merge['wri_rank'], df_merge['opn_wknd_x'])

# grouping movies based off writer rank
floors = np.arange(0, 2500, 500)
for i in range(0, 5):
    df_merge.loc[df_merge[df_merge['wri_rank'] >= floors[i]].index, 'writer_groups'] = i
df = df.merge(df_merge[['wri_rank']], left_on='title', right_on='title')

corr_matrix = df.corr()

# getting each cast members mean opn_wknd and sorting
cast_means = df.loc[df.index, ['opn_wknd', 'cast']].explode('cast').groupby('cast').mean()
cast_means.sort_values('opn_wknd', inplace=True)

cast_means['cast_rank'] = list(range(0, len(cast_means)))
cast_means.reset_index(level=0, inplace=True)

# assiging movies a rank based on best actor
df_exp = df.explode('cast')
df_merge = df_exp.merge(cast_means, left_on='cast', right_on = 'cast')
df_merge = df_merge[['title', 'opn_wknd_x', 'cast_rank']].groupby('title').max()

plt.scatter(df_merge['cast_rank'], df_merge['opn_wknd_x'])

# grouping movies based off actor rank
floors = np.arange(7500, 20000, 2500)
for i in range(0, 5):
    if floors[i] == 7500:
        df_merge.loc[df_merge[df_merge['cast_rank'] < floors[i]].index, 'cast_groups'] = i
    df_merge.loc[df_merge[df_merge['cast_rank'] >= floors[i]].index, 'cast_groups'] = i+1
df = df.merge(df_merge[['cast_rank']], left_on='title', right_on='title')

corr_matrix = df.corr()

# setting Y variables for modeling and dropping them from the dataframe
opn_wknd = df.loc[df.index, 'opn_wknd']
gross = df.loc[df.index, 'gross']
log_gross = np.log(gross)

df.drop(columns=['gross', 'opn_wknd'], inplace=True)
df.drop(columns=['title', 'directors', 'writers', 'cast'], inplace=True)

# dropping columns with low correlation with response
df.drop(columns=['Biography', 'Comedy', 'History', 'Music', 'Musical', 'Mystery', 'Sport', 'War', 'Western'], inplace=True)

df = sm.add_constant(df)

model = sm.OLS(gross, df[['const', 'budget', 'runtime']])
fit = model.fit()
fit.summary()

model = sm.OLS(log_gross, df)
fit = model.fit()
fit.summary()

df, df_test, y, y_test = train_test_split(df, log_gross, test_size=.2, random_state=10)

df_scaled = pd.DataFrame(preprocessing.scale(df), columns=(df.columns))
df_test_scaled = pd.DataFrame(preprocessing.scale(df_test), columns=(df_test.columns))



x_combos = []
for n in range(1, len(df_scaled.iloc[:, 7:].columns)+1):
   combos = combinations(df_scaled.iloc[:, 7:].columns, n)
   x_combos.extend(combos)


r_sqd = {}
mse = {}    
for n in tqdm((range(0, len(x_combos)))):
       combo_list = list(x_combos[n]) + list(df_scaled.iloc[:, :7].columns)
       x = df_scaled[combo_list]
       ols = LinearRegression()
       cv_scores = cross_validate(ols, x, y, cv=10, scoring=('neg_mean_squared_error', 'r2'), n_jobs=-1)
       r_sqd[str(combo_list)] = np.mean(cv_scores['test_r2'])
       mse[str(combo_list)] = np.mean(cv_scores['test_neg_mean_squared_error'])
print("Outcomes from the Best Linear Regression Model:")
max_r = max(r_sqd.values())
min_mse = abs(max(mse.values()))
print("Maximum Average Test R-Squared:", max_r.round(5))
print("Minimum Average Test MSE:", min_mse.round(3))
for possibles, m in mse.items():
    if -m == min_mse:
        print("The Combination of Variables:", possibles)
        ols_features = eval(possibles)


ols = LinearRegression()
fit = ols.fit(df_scaled[ols_features], y)
predict = ols.predict(df_test_scaled[ols_features])
ols_test_mse = sum((np.exp(y_test) - np.exp(predict))**2)/len(y_test)
ols_test_r2 = r2_score(y_test, predict)
print('OLS test RMSE:', np.sqrt(ols_test_mse))
print('OLS test R2:', ols_test_r2)

plt.figure(figsize=[10,5])
sns.distplot(predict-y_test, bins=50)
plt.xlabel('Log Domestic Gross Sales Residuals', fontsize = 10, style = 'italic')
plt.ylabel('Probability',fontsize = 10, style = 'italic')
plt.title('Distribution of OLS Residuals',fontsize = 15,)

plt.hist(predict-y_test, bins=50)
plt.scatter(predict, y_test)

poly = preprocessing.PolynomialFeatures(2)
poly_x = poly.fit_transform(df)
poly_x_test = poly.fit_transform(df_test)
feature_names = poly.get_feature_names(df.columns)
df_poly = pd.DataFrame(poly_x, columns=feature_names)
df_test_poly = pd.DataFrame(poly_x_test, columns=feature_names)

poly_scaled = pd.DataFrame(preprocessing.scale(df_poly), columns=(df_poly.columns))
poly_test_scaled = pd.DataFrame(preprocessing.scale(df_test_poly), columns=(df_test_poly.columns))

lasso_parameters = {'alpha': np.arange(.0005,.005,.0001)}
grid_search = GridSearchCV(estimator=Lasso(), param_grid = lasso_parameters, cv=10, scoring='neg_mean_squared_error', verbose = 0, n_jobs = -1)
grid_search.fit(poly_scaled, y)
print("Outcomes from the Best Lasso Regression Model:")
print("Average Test MSE:", -grid_search.best_score_.round(3))
print("The optimal alpha (rounded to nearest whole number):", grid_search.best_params_['alpha'])

lasso = grid_search.best_estimator_
cv_scores = cross_validate(lasso, poly_scaled, y, cv=10, scoring=('neg_mean_squared_error', 'r2'))
r = np.mean(cv_scores['test_r2'])
m = np.mean(cv_scores['test_neg_mean_squared_error'])
print("Outcomes from the Best Lasso Regression Model:")
print("Average Test R-Squared:", r.round(4))
print("Average Test MSE:", -m.round(3))
lasso.fit(poly_scaled, y)
lasso.coef_

features = []
list(features)
for i in range(0,len(poly_scaled.columns)):
    if lasso.coef_[i] != 0:
        features.append(poly_scaled.columns[i])
print(features)
len(features)

lasso_parameters = {'alpha': np.arange(.00005,.001,.00005)}
grid_search = GridSearchCV(estimator=Lasso(), param_grid = lasso_parameters, cv=10, scoring='neg_mean_squared_error', verbose = 0, n_jobs = -1)
grid_search.fit(poly_scaled[features], y)
print("Outcomes from the Best Lasso Regression Model:")
print("Average Test MSE:", -grid_search.best_score_.round(3))
print("The optimal alpha (rounded to nearest whole number):", grid_search.best_params_['alpha'])

lasso = grid_search.best_estimator_
cv_scores = cross_validate(lasso, poly_scaled[features], y, cv=10, scoring=('neg_mean_squared_error', 'r2'))
r = np.mean(cv_scores['test_r2'])
m = np.mean(cv_scores['test_neg_mean_squared_error'])
print("Outcomes from the Best Lasso Regression Model:")
print("Average Test R-Squared:", r.round(4))
print("Average Test MSE:", -m.round(3))
lasso.fit(poly_scaled[features], y)
lasso.coef_

predict = lasso.predict(poly_test_scaled[features])
lasso_test_mse = sum((np.exp(y_test) - np.exp(predict))**2)/len(y_test)
lasso_test_r2 = r2_score(y_test, predict)
print('Lasso test RMSE:', np.sqrt(lasso_test_mse))
print('Lasso test R2:', lasso_test_r2)

# RIDGE
ridge_parameters = {'alpha': np.arange(.00005,.001,.00005)}
grid_search = GridSearchCV(estimator=Ridge(), param_grid = ridge_parameters, cv=10, scoring='neg_mean_squared_error', verbose = 0, n_jobs = -1)
grid_search.fit(poly_scaled[features], y)
print("Outcomes from the Best Ridge Regression Model:")
print("Average Test MSE:", -grid_search.best_score_.round(3))
print("The optimal alpha (rounded to nearest whole number):", grid_search.best_params_['alpha'])

ridge = grid_search.best_estimator_
cv_scores = cross_validate(ridge, poly_scaled[features], y, cv=10, scoring=('neg_mean_squared_error', 'r2'))
r = np.mean(cv_scores['test_r2'])
m = np.mean(cv_scores['test_neg_mean_squared_error'])
print("Outcomes from the Best Lasso Regression Model:")
print("Average Test R-Squared:", r.round(4))
print("Average Test MSE:", -m.round(3))
ridge.fit(poly_scaled[features], y)
ridge.coef_

predict = ridge.predict(poly_test_scaled[features])
ridge_test_mse = sum((np.exp(y_test) - np.exp(predict))**2)/len(y_test)
ridge_test_r2 = r2_score(y_test, predict)
print('Ridge test RMSE:', np.sqrt(ridge_test_mse))
print('Ridge test R2:', ridge_test_r2)


# KNN

# Finding optimal number of neighbors using gridsearch
KNN_parameters = {'n_neighbors': np.arange(1,21,1)}
grid_search = GridSearchCV(estimator=KNeighborsRegressor(), param_grid = KNN_parameters, cv=10, scoring='neg_mean_squared_error', verbose = 0, n_jobs = -1)
grid_search.fit(df_scaled, y)
print("Outcomes from the Best KNN Regression Model:")
print("Minimum Average Test MSE:", -grid_search.best_score_.round(3))
print("The optimal n:", grid_search.best_params_['n_neighbors'])

# Fitting the best model
knn = grid_search.best_estimator_
cv_scores = cross_validate(knn, df_scaled, y, cv=10, scoring=('neg_mean_squared_error', 'r2'))
r = np.mean(cv_scores['test_r2'])
m = np.mean(cv_scores['test_neg_mean_squared_error'])
print("Outcomes from the Best Linear Regression Model:")
print("Maximum Average Test R-Squared:", r.round(4))
print("Minimum Average Test MSE:", -m.round(3))
knn.fit(df_scaled, y)

predict = knn.predict(df_test_scaled)
knn_test_mse = sum((np.exp(y_test) - np.exp(predict))**2)/len(y_test)
knn_test_r2 = r2_score(y_test, predict)
print('KNN test RMSE:', np.sqrt(knn_test_mse))
print('KNN test R2:', knn_test_r2)

plt.hist(predict-y_test, bins=50)
plt.scatter(predict, y_test)

# XGBOOOOOOOOOOOST

learning_rate = [round(x,2) for x in np.linspace(start = .01, stop = .6, num = 60)]
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 20)]
max_depth = range(3,10,1)
child_weight = range(1,6,2)
gamma = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1,1.1,1.2,1.3,1.4,1.5,2]
subsample = [.6, .7, .8, .9, 1]
col_sample = [.6, .7, .8, .9, 1]

# Tuning the learning_rate:
xgb_tune = XGBRegressor(n_estimators = 100,max_depth = 3, min_child_weight = 1, subsample = .8, colsample_bytree = 1,gamma = 1, n_jobs = -1)
xgb_grid = GridSearchCV(estimator=xgb_tune, param_grid = {'learning_rate':learning_rate},cv=10, scoring='neg_mean_squared_error', verbose = 0, n_jobs = -1)
xgb_grid.fit(df,y)
best_learning_rate = xgb_grid.best_params_['learning_rate']
print("Best learning_rate:", best_learning_rate)

# Tuning the n_estimators:
xgb_tune = XGBRegressor(learning_rate = best_learning_rate, max_depth = 3, min_child_weight = 1, subsample = .8, colsample_bytree = 1,gamma = 1, n_jobs = -1)
xgb_grid = GridSearchCV(estimator=xgb_tune, param_grid = {'n_estimators': n_estimators},cv=10, scoring='neg_mean_squared_error', verbose = 0, n_jobs = -1)
xgb_grid.fit(df,y)
best_n = xgb_grid.best_params_['n_estimators']
print("Best n_estimators:", best_n)

# Tuning max_depth and min_child_weight:
xgb_tune = XGBRegressor(learning_rate = best_learning_rate, n_estimators = best_n, subsample = .8, colsample_bytree = 1,gamma = 1, n_jobs = -1)
xgb_grid = GridSearchCV(estimator=xgb_tune, param_grid = {'max_depth': max_depth, 'min_child_weight': child_weight},cv=10, scoring='neg_mean_squared_error', verbose = 0, n_jobs = -1)
xgb_grid.fit(df,y)
best_depth = xgb_grid.best_params_['max_depth']
best_weight = xgb_grid.best_params_['min_child_weight']
print("Best max_depth:", best_depth)
print("Best min_child_weight:", best_weight)

# Tuning gamma:
xgb_tune = XGBRegressor(learning_rate = best_learning_rate, n_estimators = best_n, max_depth = best_depth, min_child_weight = best_weight, subsample = .8, colsample_bytree = 1, n_jobs = -1)
xgb_grid = GridSearchCV(estimator=xgb_tune, param_grid = {'gamma': gamma},cv=10, scoring='neg_mean_squared_error', verbose = 0, n_jobs = -1)
xgb_grid.fit(df,y)
best_gamma = xgb_grid.best_params_['gamma']
print("Best gamma:", best_gamma)

# Tuning subsample and colsample_bytree:
xgb_tune = XGBRegressor(learning_rate = best_learning_rate, n_estimators = best_n, max_depth = best_depth, min_child_weight = best_weight, gamma = best_gamma, n_jobs = -1)
xgb_grid = GridSearchCV(estimator=xgb_tune, param_grid = {'subsample': subsample, 'colsample_bytree': col_sample},cv=10, scoring='neg_mean_squared_error', verbose = 0, n_jobs = -1)
xgb_grid.fit(df,y)
best_subsample = xgb_grid.best_params_['subsample']
best_col_sample = xgb_grid.best_params_['colsample_bytree']
print("Best subsample:", best_subsample)
print("Best colsample_bytree:", best_col_sample)

# Rigorously tuning subsample and colsample_bytree:
subsample = [best_subsample-.02, best_subsample - .01, best_subsample, best_subsample +.01, best_subsample + .02]
col_sample = [best_col_sample-.02, best_col_sample - .01, best_col_sample, best_col_sample+.01, best_col_sample+ .02]

xgb_tune = XGBRegressor(learning_rate = best_learning_rate, n_estimators = best_n, max_depth = best_depth, min_child_weight = best_weight, gamma = best_gamma, n_jobs = -1)
xgb_grid = GridSearchCV(estimator=xgb_tune, param_grid = {'subsample': subsample, 'colsample_bytree': col_sample},cv=10, scoring='neg_mean_squared_error', verbose = 0, n_jobs = -1)
xgb_grid.fit(df,y)
best_subsample = xgb_grid.best_params_['subsample']
best_col_sample = xgb_grid.best_params_['colsample_bytree']
print("Best subsample:", best_subsample)
print("Best colsample_bytree:", best_col_sample)

# Optimal model:
xgb = XGBRegressor(learning_rate = best_learning_rate, n_estimators = best_n, max_depth = best_depth, min_child_weight = best_weight, subsample = best_subsample, colsample_bytree = best_col_sample, gamma = best_gamma, n_jobs = -1)
cv_scores = cross_validate(xgb, df, y, cv=10, scoring=('neg_mean_squared_error', 'r2'), verbose = 0, n_jobs = -1)
r = np.mean(cv_scores['test_r2'])
m = np.mean(cv_scores['test_neg_mean_squared_error'])
print("Outcomes from the Best XGBoost Regression Model:")
print("Average Test R-Squared:", r.round(5))
print("Average Test MSE:", -m.round(3))


xgb.fit(df, y)
importances = list(xgb.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature,importance in zip(df.columns, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Variable: {:20} Importance:{}'.format(*pair)) for pair in feature_importances]
sorted_features = [importance[0] for importance in feature_importances]
sorted_importances = [importance[1] for importance in feature_importances]
features = []
for i in range(0, len(sorted_features)):
    if sorted_importances[i] != 0:
        features.append(sorted_features[i])

# Printing important features:
print(features)

# Ridding the model of unimportant features and retesting:
X = df[features]
cv_scores = cross_validate(xgb, X, y, cv=10, scoring=('neg_mean_squared_error', 'r2'), verbose = 0, n_jobs = -1)
r = np.mean(cv_scores['test_r2'])
m = np.mean(cv_scores['test_neg_mean_squared_error'])
print("Outcomes from the Best Linear Regression Model:")
print("Maximum Average Test R-Squared:", r.round(5))
print("Minimum Average Test MSE:", -m.round(3))

# Re-tuning the model with only important features:

# Creating grid of parameter values:
learning_rate = [round(x,2) for x in np.linspace(start = .01, stop = .6, num = 60)]
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 20)]
max_depth = range(3,10,1)
child_weight = range(1,6,2)
gamma = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1,1.1,1.2,1.3,1.4,1.5,2]
subsample = [.6, .7, .8, .9, 1]
col_sample = [.6, .7, .8, .9, 1]

# Tuning learning_rate:
xgb_tune = XGBRegressor(n_estimators = 100,max_depth = 3, min_child_weight = 1, subsample = .8, colsample_bytree = 1,gamma = 1, n_jobs = -1)
xgb_grid = GridSearchCV(estimator=xgb_tune, param_grid = {'learning_rate':learning_rate},cv=10, scoring='neg_mean_squared_error', verbose = 0, n_jobs = -1)
xgb_grid.fit(X,y)
best_learning_rate = xgb_grid.best_params_['learning_rate']
print("Best learning_rate:", best_learning_rate)

# Tuning n_estimators:
xgb_tune = XGBRegressor(learning_rate = best_learning_rate, max_depth = 3, min_child_weight = 1, subsample = .8, colsample_bytree = 1,gamma = 1, n_jobs = -1)
xgb_grid = GridSearchCV(estimator=xgb_tune, param_grid = {'n_estimators': n_estimators},cv=10, scoring='neg_mean_squared_error', verbose = 0, n_jobs = -1)
xgb_grid.fit(X,y)
best_n = xgb_grid.best_params_['n_estimators']
print("Best n_estimators:", best_n)

# Tuning max_depth and min_child_weight
xgb_tune = XGBRegressor(learning_rate = best_learning_rate, n_estimators = best_n, subsample = .8, colsample_bytree = 1,gamma = 1, n_jobs = -1)
xgb_grid = GridSearchCV(estimator=xgb_tune, param_grid = {'max_depth': max_depth, 'min_child_weight': child_weight},cv=10, scoring='neg_mean_squared_error', verbose = 0, n_jobs = -1)
xgb_grid.fit(X,y)
best_depth = xgb_grid.best_params_['max_depth']
best_weight = xgb_grid.best_params_['min_child_weight']
print("Best max_depth:", best_depth)
print("Best min_child_weight:", best_weight)

# Tuning gamma:
xgb_tune = XGBRegressor(learning_rate = best_learning_rate, n_estimators = best_n, max_depth = best_depth, min_child_weight = best_weight, subsample = .8, colsample_bytree = 1, n_jobs = -1)
xgb_grid = GridSearchCV(estimator=xgb_tune, param_grid = {'gamma': gamma},cv=10, scoring='neg_mean_squared_error', verbose = 0, n_jobs = -1)
xgb_grid.fit(X,y)
best_gamma = xgb_grid.best_params_['gamma']
print("Best gamma:", best_gamma)

# Tuning subsample and colsample_bytree:
xgb_tune = XGBRegressor(learning_rate = best_learning_rate, n_estimators = best_n, max_depth = best_depth, min_child_weight = best_weight, gamma = best_gamma, n_jobs = -1)
xgb_grid = GridSearchCV(estimator=xgb_tune, param_grid = {'subsample': subsample, 'colsample_bytree': col_sample},cv=10, scoring='neg_mean_squared_error', verbose = 0, n_jobs = -1)
xgb_grid.fit(X,y)
best_subsample = xgb_grid.best_params_['subsample']
best_col_sample = xgb_grid.best_params_['colsample_bytree']
print("Best subsample:", best_subsample)
print("Best colsample_bytree:", best_col_sample)

# Rigorously tuning subsample and colsample_bytree:
subsample = [best_subsample-.02, best_subsample - .01, best_subsample]
col_sample = [best_col_sample-.02, best_col_sample - .01, best_col_sample]

xgb_tune = XGBRegressor(learning_rate = best_learning_rate, n_estimators = best_n, max_depth = best_depth, min_child_weight = best_weight, gamma = best_gamma, n_jobs = -1)
xgb_grid = GridSearchCV(estimator=xgb_tune, param_grid = {'subsample': subsample, 'colsample_bytree': col_sample},cv=10, scoring='neg_mean_squared_error', verbose = 0, n_jobs = -1)
xgb_grid.fit(X,y)
best_subsample = xgb_grid.best_params_['subsample']
best_col_sample = xgb_grid.best_params_['colsample_bytree']
print("Best subsample:", best_subsample)
print("Best colsample_bytree:", best_col_sample)

# Checking optimal model:
xgb = XGBRegressor(learning_rate = best_learning_rate, n_estimators = best_n, max_depth = best_depth, min_child_weight = best_weight, subsample = best_subsample, colsample_bytree = best_col_sample, gamma = best_gamma, n_jobs = -1)
cv_scores = cross_validate(xgb, X, y, cv=10, scoring=('neg_mean_squared_error', 'r2'), verbose = 0, n_jobs = -1)
r = np.mean(cv_scores['test_r2'])
m = np.mean(cv_scores['test_neg_mean_squared_error'])
print("Outcomes from the Best XGBoost Regression Model:")
print("Average Test R-Squared:", r.round(5))
print("Average Test MSE:", -m.round(3))

xgb.fit(X,y)
xgb_pred = xgb.predict(df_test[features])
xgb_test_mse = sum((np.exp(y_test) - np.exp(xgb_pred))**2)/len(y_test)
xgb_test_r2 = r2_score(y_test, xgb_pred)
print("Test R2 for XGBoost:", round(xgb_test_r2, 4))
print("Test RMSE for XGBoost:", round(np.sqrt(xgb_test_mse),3))

import pickle

with open('imdb_xgb.pickle', 'wb') as to_write:
    pickle.dump(xgb, to_write)

with open('df_train.pickle', 'wb') as to_write:
    pickle.dump(X, to_write)

with open('df_test.pickle', 'wb') as to_write:
    pickle.dump(df_test[features], to_write)
    
with open('y_test.pickle', 'wb') as to_write:
    pickle.dump(y_test, to_write)

with open('dir_means.pickle', 'wb') as to_write:
    pickle.dump(dir_means, to_write)

with open('wri_means.pickle', 'wb') as to_write:
    pickle.dump(wri_means, to_write)

with open('cast_means.pickle', 'wb') as to_write:
    pickle.dump(cast_means, to_write)


plt.figure(figsize=[10,5])
sns.distplot(predict-y_test, bins=50)
plt.xlabel('Log Domestic Gross Sales Residuals', fontsize = 10, style = 'italic')
plt.ylabel('Probability',fontsize = 10, style = 'italic')
plt.title('Distribution of XGBoost Residuals',fontsize = 15,)

plot_df = pd.concat([y_test, pd.Series(predict, index=y_test.index)], axis=1)
plot_df.columns = ['actual', 'predicted']
plt.figure(figsize=[10,5])
sns.relplot(data = plot_df, x='predicted', y='actual')
plt.xlabel('Predicted Log Domestic Gross Sales', fontsize = 10, style = 'italic')
plt.ylabel('Actual Log Domestic Gross Sales',fontsize = 10, style = 'italic')
plt.title('Predicted vs. Actual Log Domestic Sales',fontsize = 15,)
plt.plot([16, 20], [16, 20])
plt.grid()



plt.figure(figsize=[10,10])
plt.hist(predict-y_test, bins=50)
plt.figure(figsize=[10,10])
plt.scatter(predict, y_test)

sns.pairplot(pd.concat([y, df[['budget', 'runtime', 'days', 'dir_rank', 'wri_rank', 'cast_rank']]], axis=1))

plt.scatter(df['runtime'], y)
plt.scatter(df['budget'], y)

plt.scatter(np.log(df['runtime']), y)
plt.scatter(np.log(df['budget']), y)

plt.scatter((np.log(df['budget']))**2, y)

plt.scatter(df['days'], y)
plt.scatter((np.log(df['days']))**2, y)
plt.hist(y)

