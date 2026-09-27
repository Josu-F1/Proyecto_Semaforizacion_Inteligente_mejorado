# Optimización Ambiental de la Semaforización Urbana
### Sistema de Inferencia Difusa Multivariable (MIMO) + Algoritmo Genético + Extracción de Reglas con PRISM

**Universidad Técnica de Ambato — Facultad de Ingeniería en Sistemas, Electrónica e Industrial**
**Carrera de Software — Inteligencia Artificial**
**Autores:** Guevara Panimboza Marlon Steven, Fiallos Yanza Jonathan Josué

---

## Estructura del proyecto

```
proyecto/
├── data/
│   └── smart_city_traffic_mobility.csv      (dataset compartido, 204 000 registros)
│
├── 01_Reglas_PRISM/
│   └── Extraccion_Reglas_PRISM.ipynb        (Carpeta 1: extracción de reglas)
│
├── 02_Sistema_Difuso_MIMO/
│   └── Sistema_Difuso_MIMO.ipynb            (Carpeta 2: sistema difuso Mamdani MIMO)
│
├── 03_Optimizacion_Genetica/
│   └── Optimizacion_Algoritmo_Genetico.ipynb (Carpeta 3: optimización con AG)
│
└── README.md   (este archivo)
```

Cada carpeta contiene, además de su libreta Jupyter (`.ipynb`), los archivos de resultados
(`.json`, `.csv`, `.png`) que produce esa etapa y que consume la siguiente. **El flujo de
trabajo es secuencial: ejecute las libretas en orden 1 → 2 → 3.**

## Orden de ejecución

1. **`01_Reglas_PRISM/Extraccion_Reglas_PRISM.ipynb`**
   Implementa el algoritmo **PRISM** desde cero (no existe una librería madura de PRISM en
   Python) para extraer reglas `SI...ENTONCES` a partir de las 4 variables de tráfico de
   entrada, para las 2 variables de salida (`green_light_duration` y la variable derivada
   `signal_cycle_adjustment`). Exporta las reglas en formato `.json`.

2. **`02_Sistema_Difuso_MIMO/Sistema_Difuso_MIMO.ipynb`**
   Construye un **Sistema de Inferencia Difusa Mamdani MIMO** con `scikit-fuzzy`
   (`skfuzzy.control`, alias `ctrl`), usando **únicamente** las reglas extraídas en el paso 1.
   Define las funciones de pertenencia, simula el sistema y lo compara contra los datos reales.

3. **`03_Optimizacion_Genetica/Optimizacion_Algoritmo_Genetico.ipynb`**
   Usa un **Algoritmo Genético** (librería `DEAP`) para calibrar la posición de las funciones
   de pertenencia del sistema difuso, minimizando un indicador de impacto ambiental
   (`emission_estimate + air_quality_index`) estimado mediante un modelo sustituto entrenado
   sobre el propio dataset.

## Requisitos (instalación)

```bash
pip install pandas numpy matplotlib scikit-fuzzy deap scikit-learn jupyter nbformat ipywidgets
```

## Cómo ejecutar

Desde la carpeta `proyecto/`:

```bash
jupyter notebook
```

y abra cada libreta en orden, o ejecútelas por línea de comandos:

```bash
jupyter nbconvert --to notebook --execute --inplace 01_Reglas_PRISM/Extraccion_Reglas_PRISM.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_Sistema_Difuso_MIMO/Sistema_Difuso_MIMO.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_Optimizacion_Genetica/Optimizacion_Algoritmo_Genetico.ipynb
```

> Las tres libretas ya vienen **ejecutadas** con todas sus salidas (tablas, gráficos e
> impresiones) guardadas, por lo que puede revisarlas directamente sin necesidad de
> volver a correrlas.
