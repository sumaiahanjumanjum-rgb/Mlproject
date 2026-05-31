import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
 
# -------------------------------------------------------
# STEP 1: Load Dataset
# -------------------------------------------------------
# Using a sample retail dataset (Online Retail Dataset)
# Replace the path below with your actual dataset path
# df = pd.read_csv("your_dataset.csv")
 
# Sample data for demonstration
np.random.seed(42)
n = 500
countries = ['United Kingdom', 'Germany', 'France', 'Spain', 'Netherlands']
df = pd.DataFrame({
    'Quantity': np.random.randint(1, 50, n),
    'UnitPrice': np.round(np.random.uniform(0.5, 50.0, n), 2),
    'Country': np.random.choice(countries, n),
    'HighValue': np.random.randint(0, 2, n)  # Target: 1 = High Value Customer, 0 = Low
})
 
print("=" * 50)
print("STEP 1: Dataset Loaded")
print("=" * 50)
print(df.head())
print(f"\nShape: {df.shape}")
 
# -------------------------------------------------------
# STEP 2: Handle Missing Values
# -------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 2: Missing Values")
print("=" * 50)
print(df.isnull().sum())
df.dropna(inplace=True)
print("Missing values handled.")
 
# -------------------------------------------------------
# STEP 3: Encode Categorical Columns
# -------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 3: Label Encoding - Country")
print("=" * 50)
le = LabelEncoder()
df['Country_Encoded'] = le.fit_transform(df['Country'])
print(df[['Country', 'Country_Encoded']].drop_duplicates().sort_values('Country_Encoded'))
 
# -------------------------------------------------------
# STEP 4: Visualize Customer Data
# -------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 4: Visualizing Customer Distribution")
print("=" * 50)
 
plt.figure(figsize=(12, 4))
 
plt.subplot(1, 3, 1)
df['Country'].value_counts().plot(kind='bar', color='steelblue', edgecolor='white')
plt.title('Customer Distribution by Country')
plt.xlabel('Country')
plt.ylabel('Count')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
 
plt.subplot(1, 3, 2)
plt.hist(df['Quantity'], bins=20, color='teal', edgecolor='white')
plt.title('Quantity Distribution')
plt.xlabel('Quantity')
plt.ylabel('Frequency')
 
plt.subplot(1, 3, 3)
plt.hist(df['UnitPrice'], bins=20, color='coral', edgecolor='white')
plt.title('UnitPrice Distribution')
plt.xlabel('Unit Price')
plt.ylabel('Frequency')
 
plt.tight_layout()
plt.savefig('customer_distribution.png', dpi=150)
plt.show()
print("Customer distribution plot saved.")
 
# -------------------------------------------------------
# STEP 5: Split Data into Train and Test Sets
# -------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 5: Train-Test Split (80/20)")
print("=" * 50)
 
features = ['Quantity', 'UnitPrice', 'Country_Encoded']
target = 'HighValue'
 
X = df[features]
y = df[target]
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
 
print(f"Training samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")
 
# -------------------------------------------------------
# STEP 6: Train Logistic Regression
# -------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 6: Logistic Regression")
print("=" * 50)
 
lr_model = LogisticRegression(max_iter=200, random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
print(f"Accuracy: {lr_acc * 100:.2f}%")
 
# -------------------------------------------------------
# STEP 7: Train Decision Tree
# -------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 7: Decision Tree Classifier")
print("=" * 50)
 
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
print(f"Accuracy: {dt_acc * 100:.2f}%")
 
# -------------------------------------------------------
# STEP 8: Train KNN
# -------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 8: K-Nearest Neighbors (k=5)")
print("=" * 50)
 
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)
print(f"Accuracy: {knn_acc * 100:.2f}%")
 
# -------------------------------------------------------
# STEP 9: Confusion Matrices
# -------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 9: Confusion Matrices")
print("=" * 50)
 
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
models = [
    ("Logistic Regression", lr_pred),
    ("Decision Tree", dt_pred),
    ("KNN (k=5)", knn_pred)
]
 
for ax, (name, pred) in zip(axes, models):
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'High'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(name)
 
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150)
plt.show()
print("Confusion matrices saved.")
 
# -------------------------------------------------------
# STEP 10: Model Accuracy Comparison Graph
# -------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 10: Model Accuracy Comparison")
print("=" * 50)
 
model_names = ['Logistic\nRegression', 'Decision\nTree', 'KNN\n(k=5)']
accuracies = [lr_acc * 100, dt_acc * 100, knn_acc * 100]
colors = ['#378ADD', '#639922', '#BA7517']
 
plt.figure(figsize=(8, 5))
bars = plt.bar(model_names, accuracies, color=colors, width=0.5, edgecolor='white')
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.5,
             f'{acc:.2f}%',
             ha='center', va='bottom', fontweight='bold', fontsize=12)
plt.ylim(0, 110)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=150)
plt.show()
 
# -------------------------------------------------------
# Summary
# -------------------------------------------------------
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"{'Model':<25} {'Accuracy':>10}")
print("-" * 36)
print(f"{'Logistic Regression':<25} {lr_acc*100:>9.2f}%")
print(f"{'Decision Tree':<25} {dt_acc*100:>9.2f}%")
print(f"{'KNN (k=5)':<25} {knn_acc*100:>9.2f}%")
best = max([('Logistic Regression', lr_acc), ('Decision Tree', dt_acc), ('KNN', knn_acc)], key=lambda x: x[1])
print(f"\nBest Model: {best[0]} ({best[1]*100:.2f}%)")
print("=" * 50)
 
