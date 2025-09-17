#A01751150 Luis Ubaldo Balderas Sanches

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ==============================
# 1. Cargar datos
# ==============================
df = pd.read_csv("heart.csv") 
target_col = "HeartDisease"            

X = df.drop(columns=[target_col])
y = df[target_col]

# ==============================
# 2. Separar Train/Val/Test
# ==============================
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

# ==============================
# 3. Preprocesamiento (OneHot para categóricas)
# ==============================
numeric_cols = X.select_dtypes(include=["int64","float64"]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# ==============================
# 4. Pipeline con Logistic Regression
# ==============================
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000, solver="lbfgs"))
])

# Entrenar modelo base
pipeline.fit(X_train, y_train)

# Validación
y_val_pred = pipeline.predict(X_val)
y_val_prob = pipeline.predict_proba(X_val)[:,1]

print("\n=== Desempeño en VALIDATION ===")
print(classification_report(y_val, y_val_pred))

# ==============================
# 5. Matriz de confusión
# ==============================
cm = confusion_matrix(y_val, y_val_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de confusión (Validation)")
plt.show()

# ==============================
# 6. Curva ROC
# ==============================
fpr, tpr, _ = roc_curve(y_val, y_val_prob, pos_label="p")
roc_auc = auc(fpr, tpr)


plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Validation)")
plt.legend()
plt.show()

# ==============================
# 7. Diagnóstico Bias/Varianza con Learning Curve
# ==============================
train_sizes, train_scores, val_scores = learning_curve(
    pipeline, X, y, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10), scoring="accuracy"
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.plot(train_sizes, train_mean, label="Train")
plt.plot(train_sizes, val_mean, label="Validation")
plt.xlabel("Tamaño del dataset")
plt.ylabel("Accuracy")
plt.title("Learning Curve")
plt.legend()
plt.show()

# ==============================
# 8. GridSearch para mejorar desempeño
# ==============================
param_grid = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__penalty": ["l1", "l2"],
    "model__solver": ["liblinear", "lbfgs"]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1_macro")
grid.fit(X_train, y_train)

print("\nMejores parámetros encontrados:", grid.best_params_)

# ==============================
# 9. Evaluar en TEST (modelo mejorado)
# ==============================
best_model = grid.best_estimator_
y_test_pred = best_model.predict(X_test)
y_test_prob = best_model.predict_proba(X_test)[:,1]

print("\n=== Desempeño en TEST (Modelo Mejorado) ===")
print(classification_report(y_test, y_test_pred))

# ROC Test
fpr, tpr, _ = roc_curve(y_test, y_test_prob, pos_label=y_test.unique()[1])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Test - Mejorado)")
plt.legend()
plt.show()
