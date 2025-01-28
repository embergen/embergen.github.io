---
title: "Regression: Regularization"
description: A quick review of overfitting and how ridge/lasso regression can help
date: 2025-01-21 00:00:00+0000
tags: 
    - scikit-learn
    - linear regression
    - overfitting
    - regularization
    - supervised learning
    - python
categories:
    - machine learning
---

# Overfitting and Regularization

**Overfitting** is when a machine learning model fits too closely to its training data. It might look like the model is doing a good job since it can predict so well from the training data, but it will likely give inaccurate predictions when fed novel data. 

It can happen when there is not enough training data, the training data is too noisy (i.e. irrelevant info, errors, etc.), the model trains too long one one dataset, or the model complexity is overly high. 

For example, in the visual below, the overfitted model predicts each point with complete accuracy, but this model would not handle new data well. The optimal model on the left has higher residuals, but it will generalize better to new data. 

![Overfitting example](2025-01-21_overfitting.jpg)
Source: https://www.freecodecamp.org/news/what-is-overfitting-machine-learning/

**Regularization** is a collection of techniques used to prevent overfitting. These techniques penalize the features that are less influential on the model's predictions. Two common regularization techniques are **ridge regression** and **lasso regression**. Without getting into the mathematical details, the main difference between these two is that lasso will zero out the less useful features, while ridge will just reduce them. 

Which one to use?
* Ridge may be better if you want to retain all of your predictors, as lasso can zero out some predictors. This can also be helpful if you have some highly correlated predictors. 
* Lasso may be better if you have redundant predictors with negligible influence on the target variable, as it will drop them. This can lead to a more interpretable model. 
* Because lasso tends to shrink the coefficients of less important features to zero, it can be used to assess feature importance. 

[Analytics Vidhya](https://www.analyticsvidhya.com/blog/2016/01/ridge-lasso-regression-python-complete-tutorial/) breaks down the differences in more detail. 


Parameters: 
* **alpha**: a parameter we choose (similar to 'k' in KNN). Controls the model complexity. 
    * If alpha = 0, this is just OLS, which can lead to overfitting. 
    * If alpha is very high, it can lead to underfitting (and therefore less accurate predictions)

### Ridge Regression with Scikit-Learn: 

```python
# Import the libraries
from sklearn.linear_model import Ridge

# Instantiate Ridge with various alpha scores
alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
ridge_scores = []
for alpha in alphas:
    ridge = Ridge(alpha=alpha)

    # Fit the data
    ridge.fit(X_train, y_train)

    # Obtain R2
    score = ridge_score(X_test, y_test)
    ridge_scores.append(score)

# Show the list of ridge scores
print(ridge_scores)

```

### Lasso Regression: Finding Important Features: 
Note that since we are just using this to find important features, we don't need to split into training and test sets - we can use the full dataset. 

```python
# Import the libraries
from sklearn.linear_model import Lasso

# Save features, targets, and feature variable names
X = df.drop("target_variable", axis=1).values
y = df["target_variable"].values
names = df.drop("target_variable", axis=1).columns

# Instantiate Lasso
lasso = Lasso(alpha=0.1)

# Fit the model to the data and extract coefficients
lasso_coef = lasso.fit(X, y).coef_

# Plot the coefficients for each feature
plt.bar(names, lasso_coef)
plt.xticks(rotation=45)
plt.show()
```
This will produce a bar chart showing the significance of the features. It helps us identify important predictors and communicate that to others. 

![Data Camp: Lasso for feature selection in scikit-learn](2025-01-21_lasso_coefficients.png)
Source: Data Camp

