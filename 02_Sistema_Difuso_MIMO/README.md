# Carpeta 2 — Sistema de Inferencia Difusa Multivariable (MIMO)

## Contenido

| Archivo | Descripción |
|---|---|
| `Sistema_Difuso_MIMO.ipynb` | Libreta principal, paso a paso |
| `config_sistema_difuso.json` | Configuración del sistema (rangos, umbrales, conteo de reglas) — usada por la Carpeta 3 |
| `funciones_pertenencia_entrada.png` | Gráfico de las funciones de pertenencia de las 4 entradas |
| `funciones_pertenencia_salida.png` | Gráfico de las funciones de pertenencia de las 2 salidas |
| `superficie_decision_gld.png` | Superficie de decisión difusa (vehicle_count × average_speed) |
| `dispersión_real_vs_difuso.png` | Comparación de la salida difusa contra los valores reales del dataset |
| `verificacion_4_fases_mamdani.png` | Fases de agregación (función de masa) y defuzzificación (centroide) para un escenario, en ambas salidas |

## Qué hace la libreta

1. Carga las reglas extraídas por **PRISM** en la Carpeta 1 (`../01_Reglas_PRISM/`).
2. Filtra las reglas con confianza ≥ 0.60 para usarlas como base de reglas difusas.
3. Define 4 **Antecedents** (`traffic_density`, `queue_length`, `vehicle_count`,
 `average_speed`) y 2 **Consequents** (`green_light_duration`, `signal_cycle_adjustment`)
 con `skfuzzy.control` (importado como `ctrl`), usando funciones de pertenencia triangulares
 ancladas en los umbrales calculados por PRISM.
4. Traduce **cada regla PRISM** directamente a un objeto `ctrl.Rule` (sin agregar las reglas
 cualitativas del documento original — solo se usan las reglas obtenidas por PRISM, tal
 como fue solicitado).
5. Construye el `ctrl.ControlSystem` y simula el sistema con `ctrl.ControlSystemSimulation`,
 calculando **ambas salidas de forma simultánea** (propiedad MIMO).
6. Prueba el sistema con escenarios ilustrativos y sobre una muestra real del dataset,
 reportando el error (MAE) frente a los valores históricos.
7. **Panel interactivo (`ipywidgets`):** permite mover sliders para las 4 entradas y ver, en
 vivo, todo el proceso de inferencia — las funciones de pertenencia de entrada con su grado
 de activación, las funciones de pertenencia de salida con el valor defuzzificado marcado, y
 los valores numéricos resultantes. Requiere abrir la libreta en un Jupyter con kernel activo
 para poder mover los controles (la versión ya ejecutada muestra el estado inicial).
8. **Verificación explícita de las 4 fases (Paso 7-ter):** reconstruye manualmente, para un
 escenario concreto, las 4 fases de un sistema de inferencia difusa Mamdani —
 **fuzzificación → inferencia → agregación → defuzzificación** — mostrando el grado de
 pertenencia de cada entrada, la fuerza de disparo de cada regla, la función de masa
 agregada (`np.fmax`) y el centroide final, y verifica que el resultado coincide con el que
 entrega `ctrl.ControlSystemSimulation`. Genera `verificacion_4_fases_mamdani.png`.
9. Exporta la configuración del sistema para que la Carpeta 3 pueda reconstruirlo y optimizarlo.

## Librerías usadas

* `scikit-fuzzy` (`skfuzzy`, `skfuzzy.control` / `ctrl`) — motor de inferencia difusa Mamdani
* `ipywidgets` — panel interactivo (sliders) para explorar el proceso de inferencia en vivo
* `pandas`, `numpy`, `matplotlib`
