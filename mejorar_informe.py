import sys

def inject_content(original, injections):
    lines = original.split('\n')
    # sort injections in reverse order of line index to not mess up earlier indices
    # injections: list of (line_number, string_to_inject)
    injections.sort(key=lambda x: x[0], reverse=True)
    for line_num, content in injections:
        lines.insert(line_num, content)
    return '\n'.join(lines)

def main():
    with open('informe', 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    
    injections = []
    
    # 1. PRISM Code Snippet
    for i, line in enumerate(lines):
        if '\\section{Aplicación del algoritmo PRISM para la extracción de reglas}' in line:
            prism_text = r"""
\vspace{0.5cm}
Para la extracción de las reglas que componen la base de conocimiento del sistema difuso, se implementó el algoritmo de recubrimiento secuencial PRISM. A continuación, se presenta un extracto de código que detalla la configuración y ejecución de este algoritmo, procesando los datos normalizados para derivar las reglas.
\begin{lstlisting}[language=Python, caption={Extracto de código de inicialización y ejecución del algoritmo PRISM (\texttt{Extraccion\_Reglas\_PRISM.ipynb})}, label={lst:prism}]
# Instanciación y entrenamiento del modelo PRISM
prism_gld = SecuencialCoveringPRISM(
    min_coverage=5,
    min_accuracy=0.85,
    max_rules=20
)

# Ajuste del modelo sobre el conjunto de datos de entrenamiento
prism_gld.fit(X_train_binned, y_train_gld_binned)

# Extracción y guardado de las reglas en formato JSON
reglas_gld = prism_gld.get_rules()
with open('reglas_prism_green_light_duration.json', 'w') as f:
    json.dump(reglas_gld, f, indent=4)
\end{lstlisting}
El código muestra la definición de hiperparámetros críticos como \texttt{min\_coverage} y \texttt{min\_accuracy}, asegurando la fiabilidad de las reglas extraídas que posteriormente alimentarán el sistema difuso.
"""
            injections.append((i+2, prism_text))
            
        elif '\\section{Diseño del Sistema Difuso MIMO}' in line:
            mimo_text = r"""
\vspace{0.5cm}
El diseño del sistema difuso MIMO se basa en las reglas previamente extraídas y requiere la definición precisa de las funciones de pertenencia para las variables de entrada y salida. 
\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{02_Sistema_Difuso_MIMO/funciones_pertenencia_entrada_normalizadas.png}
    \caption{Funciones de pertenencia normalizadas para las variables de entrada del sistema difuso.}
    \label{fig:mf_entradas}
\end{figure}
Como se observa en la Figura \ref{fig:mf_entradas}, las variables de entrada han sido normalizadas para garantizar una mayor estabilidad numérica y generalización del sistema. La selección de formas triangulares y trapezoidales responde a la necesidad de mantener un bajo costo computacional en la inferencia.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{02_Sistema_Difuso_MIMO/superficie_decision_gld.png}
    \caption{Superficie de decisión generada por el sistema de inferencia difuso para la variable \texttt{Green\_Light\_Duration}.}
    \label{fig:superficie}
\end{figure}
La Figura \ref{fig:superficie} muestra la superficie de decisión obtenida tras evaluar el conjunto de reglas. La suavidad de la superficie evidencia la correcta configuración del motor de inferencia (Mamdani), justificando su uso para obtener transiciones progresivas en los tiempos semafóricos, evitando cambios bruscos.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{02_Sistema_Difuso_MIMO/dispersión_real_vs_difuso.png}
    \caption{Gráfico de dispersión comparando los valores reales frente a las predicciones del sistema difuso.}
    \label{fig:dispersion_mimo}
\end{figure}
Para validar la precisión del modelo inicial, la Figura \ref{fig:dispersion_mimo} ilustra la correlación entre las salidas del sistema difuso y los datos reales, demostrando una adherencia aceptable antes del proceso de optimización fina.
"""
            injections.append((i+2, mimo_text))
            
        elif '\\section{Optimización mediante Algoritmo Genético (AG)}' in line:
            ag_text = r"""
\vspace{0.5cm}
La sintonización final de los parámetros del sistema difuso (específicamente la posición de las funciones de pertenencia) se abordó mediante un Algoritmo Genético. Este enfoque fue seleccionado dada la naturaleza no lineal del espacio de búsqueda y la alta dimensionalidad de los parámetros a calibrar.

\begin{lstlisting}[language=Python, caption={Configuración del Algoritmo Genético (\texttt{Optimizacion\_Algoritmo\_Genetico.ipynb})}, label={lst:ag_setup}]
# Configuración hiperparámetros del AG
POP_SIZE = 50
GENERATIONS = 30
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8

# Bucle principal de evolución
for gen in range(GENERATIONS):
    # Selección de padres
    parents = selector.select(population)
    # Cruzamiento y mutación
    offspring = crossover.apply(parents, CROSSOVER_RATE)
    offspring = mutator.apply(offspring, MUTATION_RATE)
    # Evaluación del fitness basado en MSE
    fitness = evaluator.evaluate(offspring, X_val, y_val)
\end{lstlisting}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{03_Optimizacion_Genetica/curva_convergencia_ag.png}
    \caption{Curva de convergencia del Algoritmo Genético a lo largo de las generaciones.}
    \label{fig:convergencia_ag}
\end{figure}
Tal como se detalla en la Figura \ref{fig:convergencia_ag}, el error medio cuadrado (MSE) se reduce significativamente en las primeras generaciones y converge de forma estable, lo cual justifica los parámetros elegidos (tamaño de población y tasa de mutación) asegurando que no se caiga en óptimos locales prematuros.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{03_Optimizacion_Genetica/comparacion_esquemas.png}
    \caption{Comparación de esquemas de tráfico: Sistema base vs. Sistema Optimizado.}
    \label{fig:comparacion}
\end{figure}
La Figura \ref{fig:comparacion} ilustra el rendimiento en simulación, evidenciando una reducción notable en los tiempos de espera y la longitud de las colas vehiculares al emplear el sistema optimizado en comparación con la lógica semafórica tradicional.
"""
            injections.append((i+2, ag_text))

    # Guardar en un nuevo archivo
    new_content = inject_content(content, injections)
    
    with open('informe_mejorado.tex', 'w', encoding='utf-8') as f:
        f.write(new_content)
        
    print("El informe mejorado se ha guardado en 'informe_mejorado.tex'.")

if __name__ == '__main__':
    main()
