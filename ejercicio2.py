# Ejercicio 2: publicidad y ventas
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

def resolver(titulo, x, y, etiqueta_x, etiqueta_y):
    datos = pd.DataFrame({"X": x, "Y": y})
    modelo = LinearRegression().fit(datos[["X"]], datos["Y"])
    r = datos["X"].corr(datos["Y"])
    r2 = modelo.score(datos[["X"]], datos["Y"])
    print(f"Modelo: ŷ = {modelo.intercept_:.4f} + ({modelo.coef_[0]:.4f})X")
    print(f"Pearson r = {r:.4f}")
    print(f"R² = {r2:.4f}")
    orden = datos["X"].argsort()
    plt.scatter(datos["X"], datos["Y"], label="Datos")
    plt.plot(datos["X"].iloc[orden], modelo.predict(datos[["X"]].iloc[orden]), color="red", label="Regresión")
    plt.title(titulo)
    plt.xlabel(etiqueta_x)
    plt.ylabel(etiqueta_y)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

resolver('Ejercicio 2: publicidad y ventas', [2, 4, 6, 8, 10, 12, 14, 16], [18, 25, 31, 39, 44, 51, 57, 64], 'Publicidad (miles de S/)', 'Ventas (miles de S/)')
