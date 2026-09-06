

echo "Creando estructura de carpetas..."

mkdir -p data
mkdir -p notebooks
mkdir -p src
mkdir -p docs

echo "Creando archivos base..."

# .gitignore
cat > .gitignore << 'EOF'
data/*.csv
data/*.zip
__pycache__/
*.pyc
.ipynb_checkpoints/
.DS_Store
EOF

# requirements.txt
cat > requirements.txt << 'EOF'
pandas
numpy
scikit-learn
tensorflow
imbalanced-learn
matplotlib
seaborn
EOF

# README.md
cat > README.md << 'EOF'
# Detección de Fraude con Tarjeta de Crédito

## Objetivo

Predecir si una transacción con tarjeta de crédito es legítima o fraudulenta,
comparando dos enfoques:

- **MLP con Backpropagation**: modelo supervisado que aprende a clasificar
  transacciones a partir de ejemplos etiquetados.
- **Deep Autoencoder**: modelo de detección de anomalías que aprende el
  comportamiento normal y detecta fraude por error de reconstrucción.

## Estructura del repo

```
notebooks/
  00_carga_datos.ipynb   -> EDA, limpieza y split train/test
  01_mlp.ipynb           -> MLP con Backpropagation
  02_autoencoder.ipynb   -> Deep Autoencoder
  03_metricas.ipynb      -> Comparación final de ambos modelos
src/
  preprocessing.py       -> funciones reusables de limpieza y escalado
  metrics.py             -> funciones de evaluación
data/
  dataset (ignorado por git, descargar de Kaggle)
docs/
  informe final de la materia
```

## Dataset

Credit Card Fraud Detection Dataset (Kaggle), ~284 mil transacciones
etiquetadas. Descargar y colocar en la carpeta `data/`.



## Instalación

```
pip install -r requirements.txt
```
EOF

# Notebooks vacíos (placeholders)
touch notebooks/00_carga_datos.ipynb
touch notebooks/01_mlp.ipynb
touch notebooks/02_autoencoder.ipynb
touch notebooks/03_metricas.ipynb

# Archivos fuente reusables (placeholders)
cat > src/preprocessing.py << 'EOF'
"""
Funciones reusables de limpieza y preprocesamiento de datos.
"""

def cargar_dataset(ruta_csv):
    """Carga el dataset desde un archivo CSV."""
    import pandas as pd
    return pd.read_csv(ruta_csv)


def dividir_train_test(df, target_col="Class", test_size=0.2, random_state=42):
    """Divide el dataset en conjuntos de entrenamiento y prueba."""
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
EOF

cat > src/metrics.py << 'EOF'
"""
Funciones de evaluación para problemas de clasificación desbalanceada.
"""

def calcular_metricas(y_true, y_pred, y_proba=None):
    """Calcula Precision, Recall, F1-Score y opcionalmente ROC-AUC y PR-AUC."""
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score
    )

    resultados = {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }

    if y_proba is not None:
        resultados["roc_auc"] = roc_auc_score(y_true, y_proba)
        resultados["pr_auc"] = average_precision_score(y_true, y_proba)

    return resultados
EOF

# Placeholder para el informe
touch docs/informe_final.md

echo ""
echo "Listo! Estructura creada:"
find . -not -path '*/\.git*' -type f | sort

echo ""
echo "Proximo paso: git add . && git commit -m 'estructura inicial del proyecto' && git push"
