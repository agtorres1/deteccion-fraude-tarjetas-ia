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
