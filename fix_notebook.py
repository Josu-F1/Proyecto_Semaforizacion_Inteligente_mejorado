import json

nb_path = 'c:/Users/Marlon/Documents/Septimo/IA/Proyecto_Semaforizacion_Inteligente_mejorado/03_Optimizacion_Genetica/Optimizacion_Algoritmo_Genetico.ipynb'

with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        if 'resto = [random.choice(poblacion)' in src:
            # Replace random.choice with toolbox.select
            old_str = 'resto = [random.choice(poblacion) for _ in range(TAMANO_POBLACION - num_elite)]'
            new_str = 'resto = toolbox.select(poblacion, TAMANO_POBLACION - num_elite)'
            src = src.replace(old_str, new_str)
            
            # Put it back to list
            cell['source'] = [line + '\n' for line in src.split('\n')[:-1]] + [src.split('\n')[-1]]
            break

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    
print("Notebook fixed.")
