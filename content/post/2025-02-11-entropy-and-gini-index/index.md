---
title: Decision Tree Learning
description: How entropy and the Gini Index affect how a decision tree model constructs itself
date: 2025-02-11 00:00:00+0000
tags: 
    - scikit-learn
    - supervised learning
    - python
    - decision trees
categories:
    - machine learning
---

Important vocab: 
* **Decision Tree**: A data structure with a hierarchy of nodes
* **Node**: A question/prediction point in the tree
    * **Root Node**: The initial node. No parents, two children. 
    * **Internal Node**: A node with one parent and two children. 
    * **Leaf**: A node with one parent and NO children. Where the prediction is finally made. 

Each with children asks a question involving one feature (f) and a split point (sp). How does it know which feature and which split point to pick? 

The model tries to maximize the **information gain** from each split. I found [this link](https://medium.com/@ompramod9921/decision-trees-6a3c05e9cb82#:~:text=Information%20gain%20is%20a%20measure,It%20specifies%20randomness%20in%20data.) helfpul in getting a better idea of what this means. Basically, you can calculate the entropy (impurity/randomness) the parent and child nodes. Ideally, we want the data in each split group to belong to the same class - the one we are trying to predict. 

The difference between the entropy of the parent and the weighted average of the entropy of the child nodes is the information gain. The feature with the highest potential information gain at each node is the one that should be used to split the data. 

If the **information gain** for a node is 0, then that node is a **leaf**. A node is also a leaf if it is at the maximum depth declared for the decision tree. 

The **Gini Index** is another way of measuring the purity of the data group. 
- A lower index indicates higher purity/lower diversity
- A higher index indicates lower purity/higher diversity

So if you set the criterion to "gini" when building your model, the model will try different splits and compare the Gini Indexes to choose the ideal split question for that node. The Gini Index is slightly faster than entropy and is the default criterion for DecisionTreeClassifier(). 

```python
# Run the imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=99)

# Instantiate the model with Gini Index (alternative: criterion='entropy')
dt = DecisionTreeClassifier(criterion='gini', random_state=99)

# Fit and predict
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

# Evaluate accuracy
accuracy_score(y_test, y_pred)
```
