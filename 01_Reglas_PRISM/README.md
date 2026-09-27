# Carpeta 1 — Extracción de Reglas con el Algoritmo PRISM

## Contenido

| Archivo | Descripción |
|---|---|
| `Extraccion_Reglas_PRISM.ipynb` | Libreta principal, paso a paso |
| `reglas_prism_green_light_duration.json` | Reglas extraídas para la salida `green_light_duration` (Corta/Media/Larga) |
| `reglas_prism_signal_cycle_adjustment.json` | Reglas extraídas para la variable derivada `signal_cycle_adjustment` (Reducir/Mantener/Extender) |
| `umbrales_discretizacion.json` | Umbrales (percentiles 33.3/66.7) usados para discretizar cada variable — reutilizados en la Carpeta 2 para ubicar las funciones de pertenencia |
| `resumen_reglas_prism.csv` | Tabla consolidada de todas las reglas, con confianza/cobertura/soporte/lift |

## Planteamiento del problema

Los semáforos con tiempos fijos generan ineficiencia (colas, esperas) e impacto ambiental
(emisiones, calidad del aire) al no adaptarse a las condiciones reales de tráfico. Definir
"a ojo" las reglas y los umbrales lingüísticos de un sistema de control difuso es subjetivo;
por eso esta libreta extrae esas reglas y umbrales **de forma objetiva a partir de los datos
reales** (204 000 registros) con el algoritmo PRISM, para que la Carpeta 2 construya el
sistema difuso únicamente con conocimiento respaldado por datos. El planteamiento completo
está desarrollado en la primera celda de la libreta.

## Qué hace la libreta

1. Carga el dataset `smart_city_traffic_mobility.csv` (204 000 registros).
2. Discretiza las 4 variables de entrada (`traffic_density`, `queue_length`, `vehicle_count`,
   `average_speed`) en 3 categorías cada una, usando percentiles.
3. Discretiza `green_light_duration` (igual que el documento del proyecto) y construye la
   variable derivada `signal_cycle_adjustment` (el dataset original solo trae
   `signal_cycle_seconds` con 2 valores posibles; se documenta el criterio de ingeniería
   aplicado dentro de la libreta).
4. Implementa el **algoritmo PRISM** (recubrimiento secuencial) desde cero.
5. Ejecuta PRISM en modo detallado (`verbose=True`) para la clase `Larga`, mostrando cada
   iteración — igual que las tablas del documento del proyecto.
6. Ejecuta PRISM para las 3 clases de ambas variables de salida, completando la base total de
   reglas.
7. Exporta las reglas en formato `.json`, listas para ser consumidas por la Carpeta 2.

**No se usan librerías externas de PRISM** (no existe ninguna madura y mantenida en Python);
todo el algoritmo está implementado con `pandas`/`numpy` puros, fiel a la definición de
Cendrowska (1987).

## Alineación con la unidad "Aprendizaje basado en reglas"

La libreta incluye explícitamente: las fórmulas de **Confianza**, **Soporte**, **Cobertura** y
**Lift** tal como se definieron en clase; el pseudocódigo `Recubrimiento_secuencial` /
`AprenderUnaRegla` / `mejorRestriccion`; y una **validación exacta** contra el ejemplo clásico
"Jugar al aire libre" (Quinlan, 1986) visto en la unidad, confirmando que la implementación
reproduce la misma regla (`SI Ambiente=soleado Y Humedad=alta ENTONCES Jugar=No`, confianza 1.0,
cobertura 3) antes de aplicarla al dataset real de 204 000 registros.
