---
title: Regression Basics
date: 2025-01-16
description: Initial blog post with a basic review of using sk-learn for regression
tags: 
    - scikit-learn
    - regression
    - python
categories:
    - regression
---

I decided to get a blog going for my data science notes. It's an excuse to get more familiar with github, and it's a different way to document what I'm working on without the pressure of making a presentable project for my portfolio. 

So, without further ado: 

# Basic Regression with Scikit-Learn

In linear regression, we fit a line to our data and use that line to predict values. To check accuracy we use an error function (aka loss/cost function). 
* residual: vertical distance between a data point and the line
* Ordinary Least Squares (OLS): Minimize the residual sum of squares (RSS). 
* R2: A measure that shows how much of the variance is explained by the variable, on a scale from 0 to 1. (How well does the data fit the model, aka goodnes of fit. 
    * For example, an R2 of 30% means that 30% of the variability is explained by the model. 
    * Generally, a low R2 is undesirable. At the same time, a high R2 does not necessarily mean that your model is good.
    * What counts as a "good" R2 value also depends on the application. 
* Mean Squared Error (MSE): The mean of the RSS. Measured in squared units of the target variable. 
* Root Mean Squared Error (RMSE): The root of the MSE. 


Here's a simple sample workflow of using linear regression with model assessments:

```python
# Run the imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Split the data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=99)

# Instantiate the model
reg_model = LinearRegression()

# Fit the model on the training data
reg_model.fit(X_train, y_train)

# Predict y based on the test set
y_pred = reg_model.predict(X_test)

# Compute R2
reg_model.score(X_test, y_test)

# Commpute RMSE
mean_squared_error(y_test, y_pred, squared=False)

```