---
title: Generalization Error and the Bias-Variance Tradeoff
description: 
date: 2025-02-14 00:00:00+0000
tags: 
    - scikit-learn
    - supervised learning
    - python
    - decision trees
categories:
    - machine learning
---

Remember, we want to avoid **overfitting** and **underfitting**. In overfitting, the model fits the noise in the training set and will not handle new data well. In underfitting, the model is not flexible enough to capture the relationship between variables and thus will also not be helpful in making predictions from new data. 

![Source: DataCamp](overfitting_underfitting.png)

A **generalization error** is a measure of how well a model generalizes to new data. It is based on bias, variance, and irreducible error. We want the lowest possible generalization error, finding a balance between bias and variance. 
* **Bias**: The errors a model makes due to its assumptions. Can lead to **underfitting** if the model fails to capture important patterns. 
    * Example: If a simple linear regression model is trained on complex non-linear data, it will likely have high bias. 
    * To reduce bias: Use more complex models, add features, or improve feature engineering. 
* **Variance**: How sensitive the model is to fluctuations in the data. Can lead to **overfitting** if the model is trained to the noise rather than the underlying patterns. 
    * Example: A complex model with a lot of parameters fit to a small dataset may have high variance. 
    * To reduce variance: Use regularization techniques, cross-validation, or reduce model complexity. 
* **Irreducible error**: Noise. Errors that cannot be reduced due to choosing a "better" model. 

As bias increases, variance decreases, and vice versa. This is the bias-variance tradeoff. 

![Source: DataCamp](bias_variance_tradeoff.png)

So, how do we practically deal with this?

