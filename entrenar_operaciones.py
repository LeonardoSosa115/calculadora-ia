import tensorflow as tf
import numpy as np
import os

# Función auxiliar para crear y guardar modelos
def entrenar_y_guardar(operacion, nombre_carpeta):
    print(f"--- Entrenando modelo de {operacion} ---")
    
    # 1. Datos
    X = np.random.randint(-1000, 1000, (10000, 2)).astype(float)
    if operacion == 'suma':
        y = np.sum(X, axis=1)
    else:
        y = X[:, 0] - X[:, 1] # Resta: Columna 0 - Columna 1

    # 2. Modelo (2 entradas -> 1 salida)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4, input_shape=[2], activation='linear'), # Pequeña capa oculta
        tf.keras.layers.Dense(1, activation='linear')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mse')
    model.fit(X, y, epochs=50, verbose=0)
    
    # 3. Guardar y Convertir
    keras_file = f'{nombre_carpeta}.h5'
    model.save(keras_file)
    
    output_path = f'public/{nombre_carpeta}'
    os.makedirs(output_path, exist_ok=True)
    
    os.system(f"tensorflowjs_converter --input_format keras {keras_file} {output_path}")
    print(f"Modelo guardado en {output_path}")

# Ejecutar para ambas operaciones
entrenar_y_guardar('suma', 'modelo_suma')
entrenar_y_guardar('resta', 'modelo_resta')