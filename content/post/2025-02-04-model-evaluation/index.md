---
title: Evaluating Multiple Models
description: 
date: 2025-02-04 00:00:00+0000
tags: 
    - scikit-learn
    - supervised learning
    - python
    - data preparation
    - pipelines
categories:
    - machine learning
---

How do you know which model to use for Machine Learning? There are a few factors that can affect the decision: 
* Size of the dataset
    * If there are fewer features, the training time is faster
    * Some models won't work well with smaller datasets (e.g. ANN)
* Interpretability
    * The results of some models (e.g. linear regression) are easier to explain to people outside of data science
* Flexibility
    * A flexible model may make fewer assumptions about the data (e.g. KNN not assuming linear relationship) which can boost accuracy

We use different metrics to evaluate different models: 
* **Regression**: RMSE, R-squared
* **Classification**: Accuracy/precision/recall/F1 score, confusion matrix, ROC AUC

Using evaluation metrics, you can simply train multiple models and compare the results. Models that typically require scaling should still undergo scaling before running the evaluations. 

```python
# Run the imports for plotting, scaling, and model creation
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Create feature and target arrays
X = df.drop("target", axis=1).values
y = df["target"].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=99)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a dictionary of model names
models = {"Logistic Regression": LogisticRegression(), 
            "KNN" = KNeighborsClassifier(), 
            "Decision Tree": DecisionTreeClassifier()}

# Create an empty list to store the results
results = []

# Loop through the models with default scoring (accuracy) and append results for each to list
for model in models.values(): 
    kf = KFold(n_splits=6, random_state=99, shuffle+=True)
    cv_results = cross_val_score(model, X_train_scaled, y_train, cv=kf)
    results.append(cv_results)

# Create a boxplot of the results
plt.boxplot(results, labels=models.keys())
plt.show()

# To evaluate on the test set
for name, model in models.items():
    
    # Fit the model
    model.fit(X_train_scaled, y_train)

    # Calculate accuracy and print the results
    test_score = model.score(X_test_scaled, y_test)
    print("{} Test Set Accuracy: {}".format(name, test_score))

```
