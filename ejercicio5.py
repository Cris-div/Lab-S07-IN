# Ejercicio 5: horas de capacitación y productividad
from pathlib import Path
import sys
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

ruta = Path(__file__).resolve().parent / "productividad.xlsx"
try:
    datos = pd.read_excel(ruta, engine="openpyxl")
except ImportError as error:
    if "openpyxl" in str(error).lower():
        raise SystemExit(
            "Falta openpyxl para leer el archivo Excel. Instálalo con este comando:\n"
            f'"{sys.executable}" -m pip install openpyxl'
        ) from error
    raise
print("Primeros registros:")
print(datos.head())

X = datos[["Horas_Capacitacion"]]
y = datos["Productividad"]
modelo = LinearRegression().fit(X, y)
r = datos["Horas_Capacitacion"].corr(y)
r2 = modelo.score(X, y)
print(f"Modelo: ŷ = {modelo.intercept_:.4f} + ({modelo.coef_[0]:.4f})X")
print(f"Pearson r = {r:.4f}")
print(f"R² = {r2:.4f}")

orden = X["Horas_Capacitacion"].argsort()
plt.scatter(X.iloc[:, 0], y, label="Datos")
plt.plot(X.iloc[orden, 0], modelo.predict(X.iloc[orden]), color="red", label="Regresión")
plt.xlabel("Horas de capacitación")
plt.ylabel("Productividad (unidades por semana)")
plt.title("Capacitación y productividad")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
