# Carpeta 3 — Optimización con Algoritmo Genético (AG)

## Contenido

| Archivo | Descripción |
|---|---|
| `Optimizacion_Algoritmo_Genetico.ipynb` | Libreta principal, paso a paso |
| `validacion_motor_vectorizado.png` | Validación del motor difuso rápido contra `skfuzzy` |
| `curva_convergencia_ag.png` | Curva de convergencia del AG (mejor individuo y promedio poblacional) |
| `comparacion_esquemas.png` | Comparación: Estático vs. Difuso base vs. Difuso + AG |
| `resultado_optimizacion_ag.json` | Umbrales optimizados y resultados numéricos de la comparación |

## Qué hace la libreta

1. Carga las reglas PRISM (Carpeta 1) y la configuración del sistema difuso (Carpeta 2).
2. Reimplementa el motor de inferencia difusa en una versión **vectorizada con NumPy** (mucho
 más rápida que `skfuzzy.control` para las miles de evaluaciones que exige un AG) y la
 **valida** contra el sistema `skfuzzy` original.
3. Entrena un **modelo sustituto** (Regresión Lineal, con Random Forest como contraste) que
 estima `emission_estimate + air_quality_index` a partir de las condiciones de tráfico y la
 decisión semafórica — ya que no se dispone de un simulador de tráfico real.
4. Define el **cromosoma** (8 genes: 2 umbrales por cada una de las 4 variables de entrada) y
 la **función de aptitud** (modelo sustituto + término físico de tiempo en ralentí).
5. Configura y ejecuta un **Algoritmo Genético completo con DEAP**: selección por torneo,
 cruce *blend*, mutación gaussiana y elitismo — mostrando el progreso generación por
 generación.
6. Compara, en un lote de validación *fuera de muestra*, tres esquemas: **Estático
 tradicional**, **Difuso base (PRISM)** y **Difuso + AG (optimizado)**.

## Alineación con la unidad "Optimización — Algoritmos Genéticos"

La libreta nombra explícitamente cada decisión de diseño del AG con la terminología de la
unidad: selección por **Tournament**, cruce **Intermediate/blend**, mutación **Gaussiana**,
**Elite count**, y dos criterios de detención combinados (**Generations** y **Stall
generations**). También se justifica la codificación **real** del cromosoma (en vez de binaria)
citando la aplicación de "Optimización de Funciones de Pertenencia" vista en clase.

## Librerías usadas

* `deap` — Algoritmo Genético (`base`, `creator`, `tools`)
* `scikit-learn` — modelo sustituto (`LinearRegression`, `RandomForestRegressor`)
* `numpy`, `pandas`, `matplotlib`
* `scikit-fuzzy` — solo para la validación cruzada del motor vectorizado (Paso 3)
