---
title: Preprocessing Data for ML
description: How to use pandas' get_dummies() to prepare categorical features for machine learning
date: 2025-01-28 00:00:00+0000
tags: 
    - scikit-learn
    - supervised learning
    - python
    - pandas
    - data preparation
categories:
    - machine learning
---

In scikit-learn, our data needs to be numeric and have no missing values. However, data rarely starts out that way. **Preprocessing** is when we prepare and transform raw data so that it is suitable for our ML models. 

### Categorical Features

Categorical features (e.g. color, country, sex) can be converted into numeric binary values by using a dummy variable for each category. For example, let's say there was a category "color" with three possible values: green, purple, and yellow. Instead of having one column called "color," we could have one column named after each color. Then, the value for each observation would be either 0 or 1 for these columns. If an observation was green, it would have a 1 under green and a 0 under purple and yellow. 

However, we can actually accomplish this with one less column. If we have a column for "green" and "purple" but not "yellow", we can still express that something is yellow by giving it a 0 in "green" and "purple." Since there are only three possible colors, if it is not green or purple it must be yellow. 

There are two common ways to create dummy variables in python: pandas' **get_dummies()** and scikit-learn's **OneHotEncoder()**. 

### get_dummies()

```python
# Import pandas
import pandas as pd

# Read in the df
my_df = pd.read_csv('mydata.csv')

# Create dummy variables for our categorical feature. 
    # drop_first will drop one of the dummmies so that it is represented by all 0s in the other dummies
my_dummies = pd.get_dummies(my_df["color"], drop_first=True)

# Check the results
print(my_dummies.head())

# Combine the dummies with your original df and remove the original categorical feature
my_dummies = pd.concat([my_df, my_dummies], axis=1)
my_dummies = my_dummies.drop("color", axis=1)
```

Sidenote: If there is only one categorical feature, you can just pass the whole df into get_dummies() instead of specifying the columm. If you do so, the dummy columns will be prefixed with the original column name (e.g. color_green) and the original categorical feature will be dropped automatically. 

Either way, the df with the dummy variables can now be used for ML: 
```python
# Run the imports
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression

# Separate the target from the features
X = my_dummies.drop("target", axis=1).values
y = my_dummies["target"].values

# Create the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)

# Create a KFold object
kf = KFold(n_splits = 5, shuffle = True, random_state=99)

# Instantiate the model
linreg = LinearRegression()

# Call cross_val_score
    # sklearn's cross-val metrics presume higher scores are better, so we use negative MSE
linreg_cv = cross_val_score(linreg, X_train, y_train, cv=kf, scoring="neg_mean_squared_error")

# Calculate training RMSE from negative MSE
print(np.sqrt(-linreg_cv))

```

