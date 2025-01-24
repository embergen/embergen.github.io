---
title: "Regression: Cross-Validation"
description: A quick overview of what cross-validation is and why we use it
date: 2025-01-17 00:00:00+0000
tags: 
    - scikit-learn
    - linear regression
    - cross-validation
categories:
    - python
---

The R2 value that is returned is affected by how the data randomly happened to be split. Cross-validation helps by splitting the data into sets of groups called "folds". Let's say we split the data into 10 folds (k=10). We use the first fold as the test set, fit the model on the other folds, make predictions, and then calculate the test metric (e.g. R2). Then, the preocess is repeated but with the other folds taking turn as the test set. In the end, we will have 10 values for our test metric and can calculate their mean/median etc. 

The more folds, the more computationally expensive the cross-validation is. 

Basic setup with scikit-learn: 
```python
# Import the libraries
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression

# Call KFold (default n_splits is 5. Shuffles the dataset before splitting into folds)
kf = KFold(n_splits=10, shuffle=True, random_state=99)

# Instantiate the model
reg_model = LinearRegression()

# Call cross_val_score, returning an array of CV scores. The default metric is R2
cv_results = cross_val_score(reg_model, X, y, cv=kf)

# Print the array of CV scores
print(cv_results)

# Print the mean, standard deviation
print(np.mean(cv_results), np.std(cv_results))

# Print the 95% confidence interval
print(np.quantile(cv_results, [0.025, 0.975]))
```
