
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from deap import base, creator, tools

warnings.filterwarnings('ignore')
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

print('Librerías cargadas correctamente (incluye DEAP para el Algoritmo Genético).')


DATA_PATH = Path('../data/smart_city_traffic_mobility.csv')
RUTA_PRISM = Path('../01_Reglas_PRISM')
RUTA_DIFUSO = Path('../02_Sistema_Difuso_MIMO')

df = pd.read_csv(DATA_PATH, sep=';')
df['costo_ambiental'] = df['emission_estimate'] + df['air_quality_index']

with open(RUTA_PRISM / 'umbrales_discretizacion.json', encoding='utf-8') as f:
    umbrales = json.load(f)
with open(RUTA_PRISM / 'reglas_prism_green_light_duration.json', encoding='utf-8') as f:
    reglas_gld = json.load(f)
with open(RUTA_PRISM / 'reglas_prism_signal_cycle_adjustment.json', encoding='utf-8') as f:
    reglas_sca = json.load(f)
with open(RUTA_DIFUSO / 'config_sistema_difuso.json', encoding='utf-8') as f:
    config_difuso = json.load(f)

UMBRAL_CONFIANZA = config_difuso['umbral_confianza_regla']
reglas_gld_f = [r for r in reglas_gld if r['confianza'] >= UMBRAL_CONFIANZA]
reglas_sca_f = [r for r in reglas_sca if r['confianza'] >= UMBRAL_CONFIANZA]

VARIABLES = ['traffic_density', 'queue_length', 'vehicle_count', 'average_speed']
ETIQUETAS = config_difuso['etiquetas']
RANGOS = {v: tuple(config_difuso['rangos_entrada'][v]) for v in VARIABLES}
UMBRALES_BASE = {v: tuple(umbrales['entradas'][v]) for v in VARIABLES}
GLD_INFO = config_difuso['green_light_duration']
SCA_INFO = config_difuso['signal_cycle_adjustment']

print(f"Registros: {len(df):,}  |  Reglas GLD: {len(reglas_gld_f)}  |  Reglas SCA: {len(reglas_sca_f)}")
print('Umbrales base (heredados de PRISM):', UMBRALES_BASE)


def trimf_vectorizado(x, a, b, c):
    '''Función de pertenencia triangular vectorizada. a <= b <= c.'''
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    if b > a:
        izquierda = (x > a) & (x <= b)
        y[izquierda] = (x[izquierda] - a) / (b - a)
    if c > b:
        derecha = (x > b) & (x < c)
        y[derecha] = (c - x[derecha]) / (c - b)
    y[np.isclose(x, b)] = 1.0
    return np.clip(y, 0.0, 1.0)


def calcular_membresias(datos: dict, umbrales_var: dict) -> dict:
    '''Calcula, para cada variable y cada una de sus 3 etiquetas, el vector de membresía.'''
    memb = {}
    for var in VARIABLES:
        mn, mx = RANGOS[var]
        th1, th2 = umbrales_var[var]
        punto_medio = (th1 + th2) / 2
        etq = ETIQUETAS[var]
        memb[(var, etq[0])] = trimf_vectorizado(datos[var], mn, mn, th1)
        memb[(var, etq[1])] = trimf_vectorizado(datos[var], th1, punto_medio, th2)
        memb[(var, etq[2])] = trimf_vectorizado(datos[var], th2, mx, mx)
    return memb


# Valores representativos (centro de cada conjunto de salida) usados en el promedio ponderado
REP_GLD = {
    'Corta': (GLD_INFO['rango_min'] + GLD_INFO['corte_1']) / 2,
    'Media': (GLD_INFO['corte_1'] + GLD_INFO['corte_2']) / 2,
    'Larga': (GLD_INFO['corte_2'] + GLD_INFO['rango_max']) / 2,
}
REP_SCA = {
    'Reducir':  (SCA_INFO['rango_delta_min'] - SCA_INFO['umbral_delta']) / 2,
    'Mantener': 0.0,
    'Extender': (SCA_INFO['umbral_delta'] + SCA_INFO['rango_delta_max']) / 2,
}


def inferir_lote(datos: dict, umbrales_var: dict, reglas: list, representativos: dict) -> np.ndarray:
    '''Evalúa un lote completo de escenarios (arreglos NumPy) contra la base de reglas PRISM.'''
    memb = calcular_membresias(datos, umbrales_var)
    n = len(next(iter(datos.values())))
    numerador = np.zeros(n)
    denominador = np.zeros(n)
    for r in reglas:
        disparo = None
        for attr, valor in r['antecedente']:
            var = attr.replace('_lbl', '')
            m = memb[(var, valor)]
            disparo = m if disparo is None else np.minimum(disparo, m)
        clase = r['clase_objetivo']
        numerador += disparo * representativos[clase]
        denominador += disparo
    valor_respaldo = float(np.mean(list(representativos.values())))
    salida = np.where(denominador > 1e-9, numerador / np.where(denominador > 1e-9, denominador, 1), valor_respaldo)
    return salida


print('Motor de inferencia difusa vectorizado definido.')


import skfuzzy as fuzz
from skfuzzy import control as ctrl


def puntos_triangulares(min_v, th1, th2, max_v):
    punto_medio = (th1 + th2) / 2
    return {'bajo': [min_v, min_v, th1], 'medio': [th1, punto_medio, th2], 'alto': [th2, max_v, max_v]}


def construir_sistema_skfuzzy(umbrales_var):
    antecedentes = {}
    for var in VARIABLES:
        mn, mx = RANGOS[var]
        th1, th2 = umbrales_var[var]
        universo = np.linspace(mn, mx, 200)
        a = ctrl.Antecedent(universo, var)
        pts = puntos_triangulares(mn, th1, th2, mx)
        etq = ETIQUETAS[var]
        a[etq[0]] = fuzz.trimf(universo, pts['bajo'])
        a[etq[1]] = fuzz.trimf(universo, pts['medio'])
        a[etq[2]] = fuzz.trimf(universo, pts['alto'])
        antecedentes[var] = a

    uni_gld = np.linspace(GLD_INFO['rango_min'], GLD_INFO['rango_max'], 200)
    gld_cons = ctrl.Consequent(uni_gld, 'green_light_duration')
    pts = puntos_triangulares(GLD_INFO['rango_min'], GLD_INFO['corte_1'], GLD_INFO['corte_2'], GLD_INFO['rango_max'])
    gld_cons['Corta'] = fuzz.trimf(uni_gld, pts['bajo'])
    gld_cons['Media'] = fuzz.trimf(uni_gld, pts['medio'])
    gld_cons['Larga'] = fuzz.trimf(uni_gld, pts['alto'])

    def regla(r, cons):
        term = None
        for attr, val in r['antecedente']:
            var = attr.replace('_lbl', '')
            t = antecedentes[var][val]
            term = t if term is None else (term & t)
        return ctrl.Rule(term, cons[r['clase_objetivo']])

    reglas_ctrl = [regla(r, gld_cons) for r in reglas_gld_f]
    return ctrl.ControlSystem(reglas_ctrl), antecedentes, gld_cons


sistema_val, _, _ = construir_sistema_skfuzzy(UMBRALES_BASE)

muestra_val = df.sample(30, random_state=1)
salidas_skfuzzy, salidas_vectorial = [], []
datos_val = {v: muestra_val[v].clip(upper=RANGOS[v][1]).values for v in VARIABLES}
pred_vec = inferir_lote(datos_val, UMBRALES_BASE, reglas_gld_f, REP_GLD)

for i, (_, fila) in enumerate(muestra_val.iterrows()):
    sim = ctrl.ControlSystemSimulation(sistema_val)
    for var in VARIABLES:
        sim.input[var] = float(min(fila[var], RANGOS[var][1]))
    try:
        sim.compute()
        salidas_skfuzzy.append(sim.output.get('green_light_duration', np.nan))
    except Exception:
        salidas_skfuzzy.append(np.nan)

comparacion = pd.DataFrame({'skfuzzy_centroide': salidas_skfuzzy, 'motor_vectorizado': pred_vec})
correlacion = comparacion.corr().iloc[0, 1]
print('Correlación entre skfuzzy (centroide) y el motor vectorizado (prom. ponderado):',
      round(correlacion, 4))
comparacion.head(10)


fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(comparacion['skfuzzy_centroide'], comparacion['motor_vectorizado'], s=30)
lims = [GLD_INFO['rango_min'], GLD_INFO['rango_max']]
ax.plot(lims, lims, 'r--', linewidth=1.3, label='Coincidencia perfecta')
ax.set_xlabel('skfuzzy (centroide completo)')
ax.set_ylabel('Motor vectorizado (promedio ponderado)')
ax.set_title(f'Validación del motor rápido (r = {correlacion:.3f})')
ax.legend()
plt.tight_layout()
plt.savefig('validacion_motor_vectorizado.png', dpi=110)
plt.show()
print('Ambos motores están altamente correlacionados: el motor vectorizado es una',
      'aproximación válida y muchísimo más rápida para las miles de evaluaciones que requiere el AG.')


FEATURES_SURROGATE = ['traffic_density', 'queue_length', 'vehicle_count', 'average_speed',
                       'green_light_duration', 'signal_cycle_seconds']

X = df[FEATURES_SURROGATE].values
y = df['costo_ambiental'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

modelo_lineal = LinearRegression().fit(X_train, y_train)
pred_lineal = modelo_lineal.predict(X_test)
print('--- Regresión Lineal (usada en el AG por su velocidad) ---')
print(f"  MAE = {mean_absolute_error(y_test, pred_lineal):.2f}   R² = {r2_score(y_test, pred_lineal):.4f}")
print('  Coeficientes:', dict(zip(FEATURES_SURROGATE, modelo_lineal.coef_.round(4))))

modelo_rf = RandomForestRegressor(n_estimators=120, max_depth=14, n_jobs=-1, random_state=RANDOM_STATE)
modelo_rf.fit(X_train, y_train)
pred_rf = modelo_rf.predict(X_test)
print('\n--- Random Forest (solo para contraste de calidad) ---')
print(f"  MAE = {mean_absolute_error(y_test, pred_rf):.2f}   R² = {r2_score(y_test, pred_rf):.4f}")
print('  Importancia de variables:', dict(zip(FEATURES_SURROGATE, modelo_rf.feature_importances_.round(3))))


# --- Cromosoma: 8 genes = (umbral_1, umbral_2) para cada una de las 4 variables de entrada ---
ORDEN_GENES = []
for v in VARIABLES:
    ORDEN_GENES += [f'{v}_th1', f'{v}_th2']

# Límites de búsqueda: percentiles 5% y 95% de cada variable (evita colapsar la partición difusa)
LIMITES = {}
for v in VARIABLES:
    p5, p95 = df[v].quantile([0.05, 0.95])
    LIMITES[v] = (float(p5), float(p95))
print('Límites de búsqueda por variable (percentiles 5/95):')
for v, l in LIMITES.items():
    print(f"  {v}: {l}   (umbral PRISM original: {UMBRALES_BASE[v]})")


def decodificar(genes) -> dict:
    '''Convierte el cromosoma plano en el diccionario {variable: (th1, th2)} con th1 < th2.'''
    umbrales_var = {}
    for i, v in enumerate(VARIABLES):
        a, b = genes[2 * i], genes[2 * i + 1]
        umbrales_var[v] = (min(a, b), max(a, b))
    return umbrales_var


# --- Lote fijo de escenarios reales usado para evaluar la aptitud (batch de entrenamiento del AG) ---
N_LOTE_AG = 900
lote_ag = df.sample(N_LOTE_AG, random_state=RANDOM_STATE).reset_index(drop=True)
datos_lote = {v: lote_ag[v].clip(upper=RANGOS[v][1]).values for v in VARIABLES}
ciclo_base_lote = lote_ag['signal_cycle_seconds'].values
vehicle_count_lote = lote_ag['vehicle_count'].values

PESO_FISICO = 0.5  # ponderación del término físico (ralentí) frente al término estadístico (surrogate)


def evaluar_aptitud(genes) -> float:
    umbrales_var = decodificar(genes)
    gld_pred = inferir_lote(datos_lote, umbrales_var, reglas_gld_f, REP_GLD)
    sca_pred = inferir_lote(datos_lote, umbrales_var, reglas_sca_f, REP_SCA)
    ciclo_pred = np.clip(ciclo_base_lote + sca_pred, 40, 150)

    X_pred = np.column_stack([
        datos_lote['traffic_density'], datos_lote['queue_length'],
        datos_lote['vehicle_count'], datos_lote['average_speed'],
        gld_pred, ciclo_pred,
    ])
    costo_estadistico = modelo_lineal.predict(X_pred)
    costo_ralenti = np.clip(ciclo_pred - gld_pred, 0, None) * (vehicle_count_lote / 1000)

    costo_total = costo_estadistico.mean() + PESO_FISICO * costo_ralenti.mean()
    return costo_total


genes_base = []
for v in VARIABLES:
    genes_base += list(UMBRALES_BASE[v])

aptitud_base = evaluar_aptitud(genes_base)
print(f"\nAptitud (costo ambiental estimado) del sistema BASE (umbrales de PRISM): {aptitud_base:.3f}")


creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

def gen_aleatorio(v):
    lo, hi = LIMITES[v]
    return random.uniform(lo, hi)

def crear_individuo():
    genes = []
    for v in VARIABLES:
        genes += [gen_aleatorio(v), gen_aleatorio(v)]
    return creator.Individual(genes)

def recortar(individuo):
    for i, v in enumerate(VARIABLES):
        lo, hi = LIMITES[v]
        individuo[2 * i] = float(np.clip(individuo[2 * i], lo, hi))
        individuo[2 * i + 1] = float(np.clip(individuo[2 * i + 1], lo, hi))
    return individuo

toolbox.register('individual', crear_individuo)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('evaluate', lambda ind: (evaluar_aptitud(ind),))
toolbox.register('mate', tools.cxBlend, alpha=0.4)
toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=15, indpb=0.3)
toolbox.register('select', tools.selTournament, tournsize=3)

print('Componentes de DEAP registrados: individual, population, evaluate, mate, mutate, select.')


TAMANO_POBLACION = 60
NUM_GENERACIONES = 80          # criterio de detención 1: Generations (máximo de generaciones)
STALL_GENERACIONES = 25        # criterio de detención 2: Stall generations (sin mejora)
PROB_CRUCE = 0.6
PROB_MUTACION = 0.3
PORCENTAJE_ELITE = 0.15

poblacion = toolbox.population(n=TAMANO_POBLACION)
for ind in poblacion:
    ind.fitness.values = toolbox.evaluate(ind)

historial_mejor, historial_promedio = [], []
mejor_global = tools.selBest(poblacion, 1)[0]
mejor_global = creator.Individual(mejor_global[:])
mejor_global.fitness.values = toolbox.evaluate(mejor_global)
generaciones_sin_mejora = 0

for generacion in range(1, NUM_GENERACIONES + 1):
    num_elite = int(TAMANO_POBLACION * PORCENTAJE_ELITE)
    elite = tools.selBest(poblacion, num_elite)
    resto = toolbox.select(poblacion, TAMANO_POBLACION - num_elite)
    descendencia = list(map(toolbox.clone, elite + resto))

    for hijo1, hijo2 in zip(descendencia[::2], descendencia[1::2]):
        if random.random() < PROB_CRUCE:
            toolbox.mate(hijo1, hijo2)
            recortar(hijo1); recortar(hijo2)
            del hijo1.fitness.values, hijo2.fitness.values

    for mutante in descendencia:
        if random.random() < PROB_MUTACION:
            toolbox.mutate(mutante)
            recortar(mutante)
            del mutante.fitness.values

    invalidos = [ind for ind in descendencia if not ind.fitness.valid]
    for ind in invalidos:
        ind.fitness.values = toolbox.evaluate(ind)

    # Elitismo: aseguramos que el mejor individuo histórico sobreviva
    peor_idx = int(np.argmax([ind.fitness.values[0] for ind in descendencia]))
    descendencia[peor_idx] = creator.Individual(mejor_global[:])
    descendencia[peor_idx].fitness.values = mejor_global.fitness.values

    poblacion[:] = descendencia

    aptitudes_gen = [ind.fitness.values[0] for ind in poblacion]
    mejor_gen = tools.selBest(poblacion, 1)[0]
    if mejor_gen.fitness.values[0] < mejor_global.fitness.values[0] - 1e-9:
        mejor_global = creator.Individual(mejor_gen[:])
        mejor_global.fitness.values = mejor_gen.fitness.values
        generaciones_sin_mejora = 0
    else:
        generaciones_sin_mejora += 1

    historial_mejor.append(mejor_global.fitness.values[0])
    historial_promedio.append(float(np.mean(aptitudes_gen)))

    if generacion % 10 == 0 or generacion == 1:
        print(f"Generación {generacion:3d}/{NUM_GENERACIONES}  |  "
              f"Mejor aptitud = {mejor_global.fitness.values[0]:.4f}  |  "
              f"Promedio poblacional = {historial_promedio[-1]:.4f}  |  "
              f"Sin mejora hace {generaciones_sin_mejora} generaciones")

    if generaciones_sin_mejora >= STALL_GENERACIONES:
        print(f"\n>> Detención anticipada en la generación {generacion}: "
              f"criterio 'Stall generations' alcanzado ({STALL_GENERACIONES} generaciones sin mejora).")
        break

print(f"\nOptimización completada. Mejor aptitud final: {mejor_global.fitness.values[0]:.4f}")
print(f"Aptitud del sistema BASE (sin optimizar): {aptitud_base:.4f}")
mejora_pct = 100 * (aptitud_base - mejor_global.fitness.values[0]) / aptitud_base
print(f"Mejora relativa (lote de entrenamiento del AG): {mejora_pct:.2f} %")


fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(historial_mejor, label='Mejor individuo', linewidth=2)
ax.plot(historial_promedio, label='Promedio poblacional', linewidth=1.5, alpha=0.7)
ax.axhline(aptitud_base, color='red', linestyle='--', linewidth=1.3, label='Sistema base (umbrales PRISM)')
ax.set_xlabel('Generación'); ax.set_ylabel('Aptitud (costo ambiental estimado)')
ax.set_title('Curva de convergencia del Algoritmo Genético')
ax.legend()
plt.tight_layout()
plt.savefig('curva_convergencia_ag.png', dpi=110)
plt.show()


umbrales_optimizados = decodificar(mejor_global)
tabla_comparativa = pd.DataFrame({
    'variable': VARIABLES,
    'umbral_1_PRISM (base)': [UMBRALES_BASE[v][0] for v in VARIABLES],
    'umbral_2_PRISM (base)': [UMBRALES_BASE[v][1] for v in VARIABLES],
    'umbral_1_AG (optimizado)': [round(umbrales_optimizados[v][0], 2) for v in VARIABLES],
    'umbral_2_AG (optimizado)': [round(umbrales_optimizados[v][1], 2) for v in VARIABLES],
})
tabla_comparativa


N_VALIDACION = 900
lote_val = df.drop(lote_ag.index, errors='ignore').sample(N_VALIDACION, random_state=99).reset_index(drop=True)
datos_val2 = {v: lote_val[v].clip(upper=RANGOS[v][1]).values for v in VARIABLES}
ciclo_base_val = lote_val['signal_cycle_seconds'].values
vehicle_count_val = lote_val['vehicle_count'].values


def costo_esquema(gld_pred, ciclo_pred):
    Xp = np.column_stack([datos_val2['traffic_density'], datos_val2['queue_length'],
                           datos_val2['vehicle_count'], datos_val2['average_speed'],
                           gld_pred, ciclo_pred])
    costo_stat = modelo_lineal.predict(Xp)
    costo_ral = np.clip(ciclo_pred - gld_pred, 0, None) * (vehicle_count_val / 1000)
    return costo_stat + PESO_FISICO * costo_ral


# 1) Estático tradicional (valores históricos reales, sin razonamiento difuso)
costo_estatico = costo_esquema(lote_val['green_light_duration'].values, ciclo_base_val)

# 2) Difuso base (umbrales originales de PRISM)
gld_base = inferir_lote(datos_val2, UMBRALES_BASE, reglas_gld_f, REP_GLD)
sca_base = inferir_lote(datos_val2, UMBRALES_BASE, reglas_sca_f, REP_SCA)
ciclo_base_pred = np.clip(ciclo_base_val + sca_base, 40, 150)
costo_difuso_base = costo_esquema(gld_base, ciclo_base_pred)

# 3) Difuso + AG (umbrales optimizados)
gld_ag = inferir_lote(datos_val2, umbrales_optimizados, reglas_gld_f, REP_GLD)
sca_ag = inferir_lote(datos_val2, umbrales_optimizados, reglas_sca_f, REP_SCA)
ciclo_ag_pred = np.clip(ciclo_base_val + sca_ag, 40, 150)
costo_difuso_ag = costo_esquema(gld_ag, ciclo_ag_pred)

resultados = pd.DataFrame({
    'Esquema': ['Estático tradicional', 'Difuso base (PRISM)', 'Difuso + AG (optimizado)'],
    'Costo ambiental promedio': [costo_estatico.mean(), costo_difuso_base.mean(), costo_difuso_ag.mean()],
    'Desviación estándar': [costo_estatico.std(), costo_difuso_base.std(), costo_difuso_ag.std()],
})
resultados['Mejora vs. estático (%)'] = 100 * (resultados['Costo ambiental promedio'][0] -
                                                 resultados['Costo ambiental promedio']) / resultados['Costo ambiental promedio'][0]
resultados


fig, ax = plt.subplots(figsize=(7, 4.5))
colores = ['#B0413E', '#3B6E8F', '#3C8D5F']
barras = ax.bar(resultados['Esquema'], resultados['Costo ambiental promedio'], color=colores)
ax.set_ylabel('Costo ambiental promedio estimado\n(emission_estimate + air_quality_index)')
ax.set_title('Comparación en lote de validación (fuera de muestra)')
for barra, valor in zip(barras, resultados['Costo ambiental promedio']):
    ax.text(barra.get_x() + barra.get_width() / 2, valor, f"{valor:.1f}",
            ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig('comparacion_esquemas.png', dpi=110)
plt.show()


resultado_final = {
    'umbrales_base_prism': UMBRALES_BASE,
    'umbrales_optimizados_ag': umbrales_optimizados,
    'parametros_ag': {
        'tamano_poblacion': TAMANO_POBLACION,
        'num_generaciones': NUM_GENERACIONES,
        'prob_cruce': PROB_CRUCE,
        'prob_mutacion': PROB_MUTACION,
        'peso_fisico_ralenti': PESO_FISICO,
    },
    'aptitud_base_entrenamiento': float(aptitud_base),
    'aptitud_optimizada_entrenamiento': float(mejor_global.fitness.values[0]),
    'comparacion_validacion': resultados.to_dict(orient='records'),
}

with open('resultado_optimizacion_ag.json', 'w', encoding='utf-8') as f:
    json.dump(resultado_final, f, ensure_ascii=False, indent=2)

print('Resultados exportados a resultado_optimizacion_ag.json')
resultado_final
