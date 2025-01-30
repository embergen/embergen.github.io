---
title: Missing Data and Pipelines
description: How to drop or impute missing data, with or without a pipeline
date: 2025-01-29 00:00:00+0000
tags: 
    - scikit-learn
    - supervised learning
    - python
    - pandas
    - data preparation
    - pipelines
categories:
    - machine learning
---

A missing value referes to a cell that is blank or null in some way. It might be represented as NaN, null, NA, or given a code. 
* **Missing Completely at Random (MCAR)**: The missing value is completely independent of other variables and fully random. 
* **Missing at Random (MAR)**: The missing value is related to other variables. 
* **Missing Not at Random (MNAR)**: The missing value is related to the data point itself. 

The missing data needs to be dealt with. We can check how many missing values there are in each feature. The following will return a table with the feature nammes on the left and the sum of missing values on the right. 
```python
# Show missing values per column in ascending order of missing values
print(df.isna().sum().sort_values())
```

So, what are the options for handling missing values? 

### Dropping Observations

We can use pandas' dropna() function to remove missing observations. We can specify a subset of features that we want to check for null values. This removes all rows with missing values in those columns. As a result, you wouldn't want to use this strategy indiscriminately. 

```python
df = df.dropna(subset=["feature1", "feature2", "feature3"])
```

### Imputing Values

We can also **impute** the missing values, filling them with logical guesses bases on knowledge about the data/subject. 
* Numeric features: It's common to use the mean of the non-missing values in the feature, but you can also use the median etc. 
* Category features: It's common to use the mode of the feature

**Note:** You must split the data _before_ imputing values. Otherwise, you will end up with data leakage. **Data leakage** in machine learning is when a model uses information during training that it shouldn't have access to. The reason this specific instance would cause data leaking is that the calculated mean/mode of a feature for the entire dataset may be different than the mean/mode of the training set. 

We can conduct imputation using sklearn: 

```python
# Import SimpleImputer
from sklearn.impute import SimpleImputer

# Since categorical and numeric features will be treated differently, we separate them
X_cat = df["color"].values.reshape(-1,1)
X_num = df.drop(["color", "target"], axis=1).values
y = df["target"].values

# Split the categorical/numeric into training and test sets
X_train_cat, X_test_cat, y_train, y_test = train_test_split(X_cat, y, test_size=0.2, random_state=99)
X_train_num, X_test_num, y_train, y_test = train_test_split(X_num, y, test_size=0.2, random_state=99)

# Instantiate the imputer for categorical features using mode
imp_cat = SimpleImputer(strategy="most_frequent")

# Call fit_transform on the training categorical features and transform on the test categorical features
X_train_cat = imp_cat.fit_transform(X_train_cat)
X_test_cat = imp_cat.transform(X_test_cat)

# Instantiate a new imputer for the numeric features (default = mean)
imp_num = SimpleImputer()

# Call fit transform on the training numeric features and transform on the test numeric features
X_train_num = imp_num.fit_transform(X_train_num)
X_test_num = imp_num.transform(X_test_num)

# Combine the categorical and numeric training features into X_train and X_test
X_train = np.append(X_train_num, X_train_cat, axis=1)
X_test = np.append(X_test_num, X_test_cat, axis=1)
```

### Pipelines
We can simplify this process with a pipeline. A **pipeline** will run transformations and build a model using a single workflow. 

In the example below, we will do binary classification to predict if an observation is purple or not. 

```python
# Import the pipeline
from sklearn.pipeline import Pipeline

# Make color a binary column where all purples are set to 1 and other values are set to 0
df["color"] = np.where(df["genre"] == "purple", 1, 0)

# Separate the features and target variable into X and y
X = df.drop("color", axis=1).values
y = df["color"].values

# Create a list of tuples telling the pipeline what to do
    # Each tuple has a step name as a string and a transformer/model instantiation
    # Each step except the last needs to be a transformer
steps = [("imputation", SimpleImputer()), ("log_regression", LogisticRegression())]

# Instantiate the pipeline, passing the list of steps
pipeline = Pipeline(steps)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=99)

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Compute accuracy
pipeline.score(X_test, y_test)

```

