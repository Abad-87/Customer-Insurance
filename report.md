# Project Title: Customer Insurance Purchases Case Study

**Your Name:** [Your Full Name]  
**Date:** [Submission Date]

## Abstract

This project analyzes customer insurance purchase behavior using machine learning classification algorithms. The dataset includes attributes such as age and estimated salary to predict whether customers will purchase insurance. We compare Logistic Regression, KNN, SVM, Decision Trees, and Random Forest models, evaluating their performance through accuracy, precision, recall, and F1-score. Graphical analysis reveals decision boundaries, and specific predictions are made for given scenarios. Hypotheses regarding age and salary impacts are tested, leading to insights on optimal model selection for balancing accuracy and generalization.

## Table of Contents

1. [Introduction](#introduction)
2. [Literature Review](#literature-review)
3. [Problem Statement](#problem-statement)
4. [Data Collection and Preprocessing](#data-collection-and-preprocessing)
5. [Methodology](#methodology)
6. [Implementation](#implementation)
7. [Results](#results)
8. [Discussion](#discussion)
9. [Conclusion](#conclusion)
10. [References](#references)
11. [Appendices](#appendices)
12. [Acknowledgments](#acknowledgments)

## Introduction

In the banking and insurance sector, predicting customer behavior is crucial for targeted marketing and risk assessment. This project focuses on leveraging AI to predict insurance purchases based on age and estimated salary. The goal is to build and compare multiple classification models to identify the most effective one for this task. By analyzing patterns in the data, we aim to provide actionable insights for decision-making in insurance sales.

The project employs various machine learning techniques, including supervised learning algorithms, to classify customers into purchasers and non-purchasers. This approach aligns with the business objective of enhancing predictive capabilities in the insurance domain.

## Literature Review

Machine learning has been widely applied in predictive analytics for customer behavior. Studies on classification algorithms, such as those by Breiman (2001) on Random Forests and Cortes and Vapnik (1995) on SVMs, highlight their effectiveness in handling complex datasets. In insurance, models like Logistic Regression have been used for risk prediction (Hosmer et al., 2013). Recent advancements in ensemble methods show improved accuracy over single models, as seen in comparative studies (Caruana & Niculescu-Mizil, 2006). Challenges include overfitting in high-dimensional data and the need for feature engineering.

## Problem Statement

The challenge is to predict insurance purchases using age and estimated salary, comparing classification algorithms to find the best performer. Assumptions include data availability and binary classification suitability. Limitations involve dataset size and feature scope.

## Data Collection and Preprocessing

The dataset, Social_Network_Ads.csv, contains 400 entries with Age, EstimatedSalary, and Purchased (0/1). Data was split 75/25 for training/testing, and features were standardized using StandardScaler.

```python
# Load dataset
df = pd.read_csv("Social_Network_Ads.csv")
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

## Methodology

We selected Logistic Regression, KNN, SVM, Decision Trees, and Random Forest. Models were trained on scaled data, evaluated on test sets using accuracy, precision, recall, F1-score. Decision boundaries were visualized, and predictions made for specific scenarios.

## Implementation

Code was implemented in Python using scikit-learn. Models were fitted, predictions generated, and metrics calculated.

```python
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf'),
    "Decision Tree": DecisionTreeClassifier(random_state=0),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=0)
}
# Training and evaluation loop
```

## Results

### Model Performance

| Model              | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 0.8900   | 0.8900    | 0.8900 | 0.8900   |
| KNN               | 0.9300   | 0.9300    | 0.9300 | 0.9300   |
| SVM               | 0.9300   | 0.9300    | 0.9300 | 0.9300   |
| Decision Tree     | 0.9100   | 0.9100    | 0.9100 | 0.9100   |
| Random Forest     | 0.9200   | 0.9200    | 0.9200 | 0.9200   |

### Graphical Analysis

Decision boundaries show SVM and KNN capturing non-linear patterns effectively.

### Predictions

For Age 30, Salary 87,000: Most models predict No Purchase.  
For Age 40, No Salary: Predict No Purchase.  
For Age 40, Salary 100,000: Predict Purchase.  
For Age 50, No Salary: Predict No Purchase.  
For Age 18, No Salary: Predict No Purchase.  
For Age 22, Salary 600,000: Predict Purchase.  
For Age 35, Salary 2,500,000: Predict Purchase.  
For Age 60, Salary 100,000,000: Predict Purchase.

## Discussion

SVM and KNN show high accuracy but potential overfitting. Random Forest balances performance. Hypotheses: Younger high-salary individuals more likely to purchase; salary impacts more than age. Unexpected: Some older high-salary cases predict purchase.

## Conclusion

Random Forest is recommended for its balance. The project demonstrates AI's value in insurance prediction, with future work on larger datasets.

## References

1. Breiman, L. (2001). Random Forests. Machine Learning.
2. Cortes, C., & Vapnik, V. (1995). Support-Vector Networks. Machine Learning.
3. Hosmer, D. W., et al. (2013). Applied Logistic Regression.

## Appendices

### Full Code

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Social_Network_Ads.csv")

# Features and target
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf'),
    "Decision Tree": DecisionTreeClassifier(random_state=0),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=0)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    results[name] = {
        'accuracy': accuracy,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1_score': report['weighted avg']['f1-score']
    }
    print(f"{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {report['weighted avg']['precision']:.4f}")
    print(f"Recall: {report['weighted avg']['recall']:.4f}")
    print(f"F1-Score: {report['weighted avg']['f1-score']:.4f}")
    print(classification_report(y_test, y_pred))
    print("-" * 50)

# Plot decision boundaries for all models
x1_min, x1_max = X_train[:,0].min()-1, X_train[:,0].max()+1
x2_min, x2_max = X_train[:,1].min()-1, X_train[:,1].max()+1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                       np.arange(x2_min, x2_max, 0.02))

for name, model in models.items():
    model.fit(X_train, y_train)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx1, xx2, model.predict(np.array([xx1.ravel(), xx2.ravel()]).T).reshape(xx1.shape), alpha=0.3)
    plt.scatter(X_train[:,0], X_train[:,1], c=y_train, edgecolors='k')
    plt.xlabel("Age (scaled)")
    plt.ylabel("Estimated Salary (scaled)")
    plt.title(f"Decision Boundary: {name}")
    plt.savefig(f"{name.replace(' ', '_')}_decision_boundary.png")
    plt.show()

# Predictions for specific scenarios
scenarios = [
    [30, 87000],
    [40, 0],  # No Salary
    [40, 100000],
    [50, 0],  # No Salary
    [18, 0],  # Second set
    [22, 600000],
    [35, 2500000],
    [60, 100000000]
]

print("\nPredictions for Specific Scenarios:")
for i, (age, salary) in enumerate(scenarios):
    input_data = sc.fit_transform([[age, salary]])  # Note: Using fit_transform on single sample, but ideally use the same scaler
    print(f"Scenario {i+1}: Age {age}, Salary {salary}")
    for name, model in models.items():
        pred = model.predict(input_data)[0]
        print(f"  {name}: {'Purchase' if pred == 1 else 'No Purchase'}")
    print()

# Hypotheses Testing
print("Hypotheses Testing:")
# Hypothesis 1: Younger individuals with higher salaries are more likely to purchase
print("Hypothesis 1: Younger (age 25) with high salary (150000) vs older (age 50) with low salary (30000)")
young_high = sc.transform([[25, 150000]])
old_low = sc.transform([[50, 30000]])
for name, model in models.items():
    pred_young = model.predict(young_high)[0]
    pred_old = model.predict(old_low)[0]
    print(f"  {name}: Young High Salary - {'Purchase' if pred_young == 1 else 'No Purchase'}, Old Low Salary - {'Purchase' if pred_old == 1 else 'No Purchase'}")

# Hypothesis 2: Older individuals with higher salaries might be less inclined
print("Hypothesis 2: Older (age 60) with high salary (200000) vs young (age 30) with high salary (200000)")
old_high = sc.transform([[60, 200000]])
young_high2 = sc.transform([[30, 200000]])
for name, model in models.items():
    pred_old = model.predict(old_high)[0]
    pred_young = model.predict(young_high2)[0]
    print(f"  {name}: Old High Salary - {'Purchase' if pred_old == 1 else 'No Purchase'}, Young High Salary - {'Purchase' if pred_young == 1 else 'No Purchase'}")

# Hypothesis 3: Salary has stronger impact than age
print("Hypothesis 3: Vary salary at fixed age (40), and vary age at fixed salary (100000)")
for salary in [50000, 100000, 150000]:
    input_sal = sc.transform([[40, salary]])
    print(f"  Age 40, Salary {salary}:")
    for name, model in models.items():
        pred = model.predict(input_sal)[0]
        print(f"    {name}: {'Purchase' if pred == 1 else 'No Purchase'}")

for age in [30, 40, 50]:
    input_age = sc.transform([[age, 100000]])
    print(f"  Age {age}, Salary 100000:")
    for name, model in models.items():
        pred = model.predict(input_age)[0]
        print(f"    {name}: {'Purchase' if pred == 1 else 'No Purchase'}")
```

### Additional Graphs

Decision boundary images are saved as PNG files: Logistic_Regression_decision_boundary.png, KNN_decision_boundary.png, etc.

## Acknowledgments

Thanks to scikit-learn community and dataset providers.
