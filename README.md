# FinancialDataLoader
FinancialDataLoader es un proyecto en Python que permite descargar y graficar datos financieros de múltiples fuentes (stooq y yahoo) de forma paralela y utilizando caché para mejorar la eficiencia computacional.

## Características

- Descarga de datos financieros desde **stooq** y **yahoo**.
- Implementación de caché para evitar descargas innecesarias.
- Procesamiento en paralelo utilizando **ThreadPoolExecutor** o **ProcessPoolExecutor**.
- Visualización de precios de cierre utilizando **matplotlib**.
