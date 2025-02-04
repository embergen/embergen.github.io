---
title: Normalization
description: Use StandardScaler() to put numerical features on the same scale for ML
date: 2025-01-30 00:00:00+0000
tags: 
    - scikit-learn
    - supervised learning
    - python
    - data preparation
    - pipelines
categories:
    - machine learning
---

Why is it important to scale our data? Different numeric variables have have very different ranges. One might be represented only by values from 0 to 1, while another might be in the millions (or millionths). 

Many ML models (e.g. KNN, linear/ridge/lasso, log reg, ANN) are influenced by distance, so a feature with a large scale would impact the model more than a feature with a small scale. So, we **standardize** our data so that our features are represented by similar scales. (There are other methods as well.)

## Standardization

What does standardization do?
1. Subtract the mean
2. Divide by the variance

Result? All features are centered around zero and have a variance of one. 

How-to: 

```python
# Import StandardScaler
from sklearn.preprocessing import StandardScaler

# Create X and y and split into train/test sets
X = df.drop("color", axis=1").values
y = df["color"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 99)

# Instantiate the scaler
scaler = StandardScaler()

# Fit transform the training features
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test features
X_test_scaled = scaler.transform(X_test)

# Optionally, vertify the changes by comparing the old and new mean/std
print(np.mean(X), np.std(X))
print(np.mean(X_train_scaled), np.std(X_train_scaled))
```

Scalers also work well in pipelines, like I used in yesterday's notes on handling missing data. 

```python
# Create a list of tuples with step names and our transformer/model
steps = [("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=6))]

# Create the pipeline and feed it the steps
pipeline = Pipeline(steps)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 99)

# Fit the pipeline to the training set
knn_scaled = pipeline.fit(X_train, y_train)

# Predict from the test set
y_pred = knn_scaled.predict(X_test)

# Check the model's accuracy
print(knn_scaled.score(X_test, y_test))
```

Let's add another layer. In this pipeline, we will also include a Grid Search to see which value for n_neighbors performs best. 

```python
# Import Grid Search
from sklearn.model_selection import GridSearchCV

# Create steps and pipeline
steps = [("scaler", StandardScaler()), ("knn", KNeighborsClassifier())] 
pipeline = Pipeline(steps)

# Create an array of values to try for n_neighbors (range of integers from 1 to 49)
params = {"knn__n_neighbors": np.arange(1, 50)}

# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)

# Perform a grid search
cv = GridSearchCV(pipeline, param_grid = params)

# Fit the grid search object to the training data
cv.fit(X_train, y_train)

# Predict from the test set
y_pred = cv.predict(X_test)

# Show the best score and best value for n_neighbors from the grid search CV
print(cv.best_score_, "\n", cv.best_params_)
```
