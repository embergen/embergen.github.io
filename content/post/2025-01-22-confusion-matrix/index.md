---
title: Classification Metrics and the Confusion Matrix
description: Ways of defining "accuracy" and a consideration of different concerns
date: 2025-01-22 00:00:00+0000
tags: 
    - scikit-learn
    - metrics
    - classification
categories:
    - python
#image: 2025-01-22_code-output.png
---

# Metrics for Classification Models

The word "acccuracy" is often used broadly to cover how close to the truth something is, but in data science the meaning is more specific. There are also metrics other than accuracy that will judge how correct a model is, and the one you choose depends on the purpose of your study. It might be preferable to have a false positive, or it might be more preferable to avoid a false negative, for example. 

## Negatives/Positives in Classification
There are four potential outcomes for your predictions when performing binary classification: 
* **True Positive (TP)**: You correctly labeled a positive data point as positive. 
* **False Positive (FP)**: You incorrectly labeled a negative data point as positive. 
* **True Negative (TN)**: You correctly labeled a negative data point as negative. 
* **False Negative (FN)**: You incorrectly labeled a positive data point as negative. 

As I mentioned, you will have different concerns based on the purpose of your analysis. For example, I did a machine learning assignment in school where I was using features to predict which students might need additional academic support. In my case, I was not overly concerned with false positives - if the school started academic intervention for a student who ended up not needing it, it would not be harmful for the student and the staff leading the support group would soon notice. 

In other cases, you might be more concerned about false positives. For example, a spam detector that flags important emails as spam would not be useful to a client. 

A **confusion matrix** is a two-by-two grid which displays all of the values for TP, TN, FP, and FN.

![Confusion matrix](2025-01-22_confusion-matrix.png)
[Source](https://rumn.medium.com/precision-recall-and-f1-explained-with-10-ml-use-case-6ef2fbe458e5)

## Metrics

These four classification outcomes lead to four metrics for assessing a classification model:

* **Accuracy**: Out of all predictions, how many were correct?
    * Formula: (TP + TN) / (TP + TN + FP + FN)
* **Precision**: Out of all the times the model predicted a positive, how often was that positive correct?
    * Formula: TP / (TP + FP)
    * High precision = lower FP rate
* **Recall**: Out of all the data points that should have been predicted as positive, how many were accurately predicted?
    * Formula: TP / (TP + FN)
    * High recall = lower FN rate
* **F1 Score**: The harmonic mean of precision and recall, giving them equal weight. Useful when you have imbalanced data or want to account for both FP and FN. 
    * Formula: 2 ((Precision * Recall) / (Precision + Recall))

Relying on the wrong metric can lead to a false understanding of how well your model is performing. For example, a model might have a high accuracy score, but this might simply be the result of imbalanced data. 

Let's say you are trying to classify if something is a fish or a cow. If your data is 99% fish, and your model predicts that every sincle datapoint is a fish, then the accuracy score would be 99%. It sounds good, but it does not actually mean your model is good at differentiating between fish and cows. 

## Python Applications

```python
# Import the libraries
from sklearn.metrics import classification_report, confusion_matrix

# Instantiate the classifier
knn = KNeighborsClassifier(n_neighbors = 7)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 99)

# Fit the model with the training data
knn.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = knn.predict(X_test)

# Generate a confusion matrix from the predicted labels and test labels
print(confusion_matrix(y_test, y_pred))

# Generate metrics for the predicted and test labels
print(classification_report(y_test, y_pred)

```

Sample results of confusion matrix and classification report: 

![code output](2025-01-22_code-output.png)

