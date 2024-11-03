import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport

# Cargar el archivo CSV
data = pd.read_csv('bank-full.csv', sep=';')

# Mostrar las primeras filas del dataset
print("Primeras filas de la base de datos:")
print(data.head())

# Información general sobre el dataset (tipos de datos y valores nulos)
print("\nInformación general del dataset:")
data.info()

# Descripción estadística de las columnas numéricas
print("\nDescripción estadística de las columnas numéricas:")
print(data.describe())

# Verificar valores nulos en cada columna
print("\nCantidad de valores nulos por columna:")
print(data.isnull().sum())

# Identificar columnas numéricas y de texto
caracteristicas_numericas = data.select_dtypes(include=[np.number]).columns.to_list()
caracteristicas_texto = data.select_dtypes(include=[object]).columns.to_list()

print("\nCaracterísticas numéricas:\n", caracteristicas_numericas)
print("\nCaracterísticas de texto:\n", caracteristicas_texto)

# Mostrar un análisis de los valores únicos en las columnas de texto
print("\nValores únicos en columnas de texto:")
for col in caracteristicas_texto:
    print(f"{col}: {data[col].nunique()} valores únicos")

# Mostrar los datos faltantes en porcentajes
print("\nPorcentaje de datos faltantes por columna:")
print((data.isnull().sum() / len(data)) * 100)

# Verificar duplicados
duplicados = data.duplicated().sum()
print(f"Duplicados encontrados: {duplicados}")

# Identificar valores atípicos usando IQR (Interquartile Range) para las columnas numéricas
Q1 = data[caracteristicas_numericas].quantile(0.25)
Q3 = data[caracteristicas_numericas].quantile(0.75)
IQR = Q3 - Q1
valores_atipicos = ((data[caracteristicas_numericas] < (Q1 - 1.5 * IQR)) | (data[caracteristicas_numericas] > (Q3 + 1.5 * IQR))).sum()

print("\nValores atípicos por columna:")
print(valores_atipicos)

# Verificar tipos de datos
print("\nTipos de datos incorrectos:")
for columna in data.columns:
    if data[columna].dtype == 'object':
        try:
            data[columna].astype(float)
        except:
            print(f"La columna '{columna}' tiene valores no numéricos.")

#Análisis de variables categóricas
cat_feats = data.select_dtypes(include=['object']).columns

# Revisar las categorías únicas y frecuencia de cada variable categórica
for col in cat_feats:
    print(f"\nCategorías únicas en '{col}': {data[col].unique()}")
    print(data[col].value_counts())

#Transformación de datos 
# Escalado de variables numéricas
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
num_feats = data.select_dtypes(include=['int64', 'float64']).columns
data[num_feats] = scaler.fit_transform(data[num_feats])

# Comprobamos el resultado de la normalización
print("\nDescripción estadística de variables numéricas normalizadas:\n", data[num_feats].describe())

# Verificación de consistencia en datos categóricos
# Eliminar posibles espacios en blanco y homogeneizar las categorías en minúsculas
for col in cat_feats:
    data[col] = data[col].str.strip().str.lower()

# Revisión de las categorías después del ajuste de consistencia
for col in cat_feats:
    print(f"\nCategorías en '{col}' tras ajuste:\n", data[col].unique())

# Verificar los resultados finales
print("Transformaciones completadas. Vista previa del dataset:")
print(data.head())

#Exploración de datos
import matplotlib.pyplot as plt
import seaborn as sns

num_feats = data.select_dtypes(include=['int64', 'float64']).columns
data[num_feats].hist(bins=20, figsize=(12, 10))
plt.suptitle("Distribución de variables numéricas")
plt.show()

plt.figure(figsize=(12, 8))
for i, col in enumerate(num_feats, 1):
    plt.subplot(2, 4, i)
    sns.boxplot(data=data, y=col)
    plt.title(f"Boxplot de {col}")
plt.tight_layout()
plt.show()

sns.countplot(data=data, x='y')
plt.title("Distribución de la variable objetivo 'y'")
plt.show()



