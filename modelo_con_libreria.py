"""
Logistic Regression usando librería (scikit-learn) - Luis Balderas A01751150
""" 
import argparse, csv, math, random, sys
from typing import List


from sklearn.linear_model import LogisticRegression as SKLogisticRegression



def _can_float(s: str) -> bool:
    try:
        float(s)
        return True
    except:
        return False

def _build_column_types(rows, target_col):
    """
    Detecta columnas numericas vs categoricas
    Regresa dos listas: numeric_cols, categorical_cols
    """
    header = list(rows[0].keys())
    numeric_cols, categorical_cols = [], []
    for c in header:
        if c == target_col:
            continue
        # si todas las filas son convertibles a float => numerica; si alguna no, categórica
        all_float = True
        for r in rows:
            if not _can_float(r[c]):
                all_float = False
                break
        if all_float:
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)
    return numeric_cols, categorical_cols

def _fit_one_hot_maps(rows, categorical_cols):
    """
    Para cada columna categorica, junta el conjunto de categorias observadas
    y crea el mapeo de nombres de columnas one-hot resultantes
    """
    cat_values = {c: [] for c in categorical_cols}
    seen = {c: set() for c in categorical_cols}
    for r in rows:
        for c in categorical_cols:
            v = r[c]
            if v not in seen[c]:
                seen[c].add(v)
                cat_values[c].append(v)
    # genera nombres de columnas one-hot
    one_hot_cols = []
    for c in categorical_cols:
        for v in cat_values[c]:
            one_hot_cols.append(f"{c}__{v}")
    return cat_values, one_hot_cols

def _encode_row_numeric_and_one_hot(r, numeric_cols, categorical_cols, cat_values):
    xi = [float(r[c]) for c in numeric_cols]  # numéricos primero
    for c in categorical_cols:
        vals = cat_values[c]
        val = r[c]
        for v in vals:
            xi.append(1.0 if val == v else 0.0)
    return xi

def _parse_target_to_int(v):
    try:
        return int(v)
    except:
        s = str(v).strip().lower()
        if s in ("1", "true", "yes", "y", "passed", "pass", "positive", "pos","p"):
            return 1
        if s in ("0", "false", "no", "n", "failed", "fail", "negative", "neg","e"):
            return 0
        # ultimo recurso: si es floatable, umbral 0.5
        if _can_float(s):
            return 1 if float(s) >= 0.5 else 0
        raise ValueError(f"No puedo interpretar el target '{v}' como 0/1.")

def read_csv_numeric_features(path: str, target_col: str):
    with open(path, newline='', encoding='utf-8') as f:  # leemos el archivo
        reader = csv.DictReader(f)  # lo convertimos a diccionario (iterador)
        rows = list(reader)
    # detectar columnas numéricas vs categóricas (excluyendo target)
    numeric_cols, categorical_cols = _build_column_types(rows, target_col)


    if len(categorical_cols) == 0:
        feature_cols = [c for c in rows[0].keys() if c != target_col]
        X, y = [], []
        for r in rows:
            xi = [float(r[c]) for c in feature_cols]
            yi = _parse_target_to_int(r[target_col])
            X.append(xi); y.append(yi)
        return X, y, feature_cols

    # Si hay categóricas, construimos one-hot y concatenamos con numéricas
    cat_values, one_hot_cols = _fit_one_hot_maps(rows, categorical_cols)
    # Orden final: primero numéricas, luego una por categoría
    feature_cols = numeric_cols + one_hot_cols

    X, y = [], []
    for r in rows:
        xi = _encode_row_numeric_and_one_hot(r, numeric_cols, categorical_cols, cat_values)
        yi = _parse_target_to_int(r[target_col])
        X.append(xi); y.append(yi)
    return X, y, feature_cols



def confusion_matrix_(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt==1 and yp==1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt==0 and yp==0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt==0 and yp==1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt==1 and yp==0)
    return tp, tn, fp, fn

def metrics(tp, tn, fp, fn):
    total = tp+tn+fp+fn
    acc = (tp+tn)/total if total else 0
    prec = tp/(tp+fp) if tp+fp>0 else 0
    rec = tp/(tp+fn) if tp+fn>0 else 0
    f1 = 2*prec*rec/(prec+rec) if prec+rec>0 else 0
    return acc, prec, rec, f1

def save_dataset(path, X, y, feature_cols):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(feature_cols + ["target"])
        for xi, yi in zip(X, y):
            writer.writerow(list(xi) + [yi])

def train_test_split(X, y, test_size=0.3, seed=42):
    """Split simple, manteniendo tu estilo (sin numpy)."""
    n = len(X)
    idx = list(range(n))
    random.seed(seed)
    random.shuffle(idx)
    n_test = int(round(n*test_size))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test  = [X[i] for i in test_idx]
    y_test  = [y[i] for i in test_idx]
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# MAIN (solo cambie el modelo a scikit-learn)
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=0.1)       
    ap.add_argument("--epochs", type=int, default=1000)     # lo mapeamos a max_iter
    # argumentos para guardar los splits
    ap.add_argument("--train-out", default="train_dataset.csv")
    ap.add_argument("--test-out", default="test_dataset.csv")
    args = ap.parse_args()

    # Carga y features
    X, y, features = read_csv_numeric_features(args.data, args.target)
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=args.test_size)

    # guardamos datasets
    save_dataset(args.train_out, X_train, y_train, features)
    save_dataset(args.test_out,  X_test,  y_test,  features)


    model = SKLogisticRegression(max_iter=args.epochs, solver="lbfgs")
    model.fit(X_train, y_train)

    # Predicción
    y_pred = model.predict(X_test)

    # Métricas 
    tp, tn, fp, fn = confusion_matrix_(y_test, y_pred)
    acc, prec, rec, f1 = metrics(tp, tn, fp, fn)

    
    print("\n" + "="*50)
    print("      Logistic Regression (scikit-learn) - Resultados")
    print("="*50)

    coefs = model.coef_
    intercepts = model.intercept_

    if len(coefs.shape) == 2 and coefs.shape[0] == 1:
        # Binario: una fila de coeficientes
        print(f"Features usados ({len(features)}):")
        for f, w in zip(features, coefs[0]):
            print(f"  {f:<20} -> {w:.4f}")
        print(f"Bias: {intercepts[0]:.4f}")
    else:
        # Multiclase: imprimimos por clase (fila en coef_)
        for k, (w_row, b) in enumerate(zip(coefs, intercepts)):
            print(f"\nClase {k}:")
            for f, w in zip(features, w_row):
                print(f"  {f:<20} -> {w:.4f}")
            print(f"Bias: {b:.4f}")

    print("\nMatriz de confusión (y_true filas, y_pred columnas):")
    print(f"          Pred 0    Pred 1")
    print(f"True 0     {tn:5d}    {fp:5d}")
    print(f"True 1     {fn:5d}    {tp:5d}")

    print("\nMétricas de desempeño:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-score : {f1:.4f}")

    print("\nArchivos guardados:")
    print(f"  Train -> {args.train_out}")
    print(f"  Test  -> {args.test_out}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
