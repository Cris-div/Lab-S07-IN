# Ejercicio 6: preparación de datos según CRISP-DM, Fase 3

import pandas as pd

clientes = pd.DataFrame(
    [
        ["C01", 25, 2500, 4, "Lima"],
        ["C02", 32, 3200, 6, "Arequipa"],
        ["C03", None, 2800, 5, "Lima"],
        ["C04", 45, None, 8, "Cusco"],
        ["C05", 29, 3000, 4, "Lima"],
        ["C05", 29, 3000, 4, "Lima"],  # Registro duplicado
        ["C06", 38, 4100, None, "Arequipa"],
        ["C07", 41, 4500, 9, "Cusco"],
    ],
    columns=["Cliente", "Edad", "Ingresos", "Compras", "Ciudad"],
)

print("Valores faltantes por columna:")
print(clientes.isna().sum())
print("\nCantidad de filas duplicadas:", clientes.duplicated().sum())

# Elimina duplicados y reemplaza los faltantes numéricos por la media.
clientes = clientes.drop_duplicates().copy()
for columna in ["Edad", "Ingresos", "Compras"]:
    clientes[columna] = clientes[columna].fillna(clientes[columna].mean())

# Codifica Ciudad con columnas 0/1, sin asignar orden a las ciudades.
clientes = pd.get_dummies(clientes, columns=["Ciudad"], dtype=int)
print("\nBase final preparada:")
print(clientes.to_string(index=False))
