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

df = pd.read_csv("Social_Network_Ads.csv")

X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

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

scenarios = [
    [30, 87000],
    [40, 0],  
    [40, 100000],
    [50, 0],  
    [18, 0],  
    [22, 600000],
    [35, 2500000],
    [60, 100000000]
]

print("\nPredictions for Specific Scenarios:")
for i, (age, salary) in enumerate(scenarios):
    input_data = sc.fit_transform([[age, salary]])  
    print(f"Scenario {i+1}: Age {age}, Salary {salary}")
    for name, model in models.items():
        pred = model.predict(input_data)[0]
        print(f"  {name}: {'Purchase' if pred == 1 else 'No Purchase'}")
    print()

print("Hypotheses Testing:")
print("Hypothesis 1: Younger (age 25) with high salary (150000) vs older (age 50) with low salary (30000)")
young_high = sc.transform([[25, 150000]])
old_low = sc.transform([[50, 30000]])
for name, model in models.items():
    pred_young = model.predict(young_high)[0]
    pred_old = model.predict(old_low)[0]
    print(f"  {name}: Young High Salary - {'Purchase' if pred_young == 1 else 'No Purchase'}, Old Low Salary - {'Purchase' if pred_old == 1 else 'No Purchase'}")

print("Hypothesis 2: Older (age 60) with high salary (200000) vs young (age 30) with high salary (200000)")
old_high = sc.transform([[60, 200000]])
young_high2 = sc.transform([[30, 200000]])
for name, model in models.items():
    pred_old = model.predict(old_high)[0]
    pred_young = model.predict(young_high2)[0]
    print(f"  {name}: Old High Salary - {'Purchase' if pred_old == 1 else 'No Purchase'}, Young High Salary - {'Purchase' if pred_young == 1 else 'No Purchase'}")

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

