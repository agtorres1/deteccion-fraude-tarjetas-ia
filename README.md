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
