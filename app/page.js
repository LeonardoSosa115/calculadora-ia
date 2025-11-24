"use client";
import { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';

export default function Calculadora() {
  const [modeloActual, setModeloActual] = useState(null);
  const [operacion, setOperacion] = useState('suma'); // 'suma' o 'resta'
  const [valA, setValA] = useState('');
  const [valB, setValB] = useState('');
  const [resultado, setResultado] = useState(null);

  // Cargar el modelo cuando cambia la operación
  useEffect(() => {
    async function cambiarModelo() {
      setModeloActual(null); // Resetear mientras carga
      setResultado(null);
      
      const path = operacion === 'suma' 
        ? '/modelo_suma/model.json' 
        : '/modelo_resta/model.json';
      
      try {
        const m = await tf.loadLayersModel(path);
        setModeloActual(m);
        console.log(`Modelo de ${operacion} cargado`);
      } catch (err) {
        console.error(err);
      }
    }
    cambiarModelo();
  }, [operacion]); // Se ejecuta cada vez que 'operacion' cambia

  const calcular = async () => {
    if (!modeloActual || valA === '' || valB === '') return;

    // TENSORFLOW.JS: Crear tensor de entrada [1, 2]
    // 1 ejemplo, 2 características (número A y número B)
    const input = tf.tensor2d([[parseFloat(valA), parseFloat(valB)]]);
    
    const prediccion = modeloActual.predict(input);
    const data = await prediccion.data();
    
    setResultado(data[0].toFixed(2));
    
    input.dispose();
    prediccion.dispose();
  };

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
      <div className="bg-white p-8 rounded-xl shadow-lg max-w-md w-full">
        <h1 className="text-2xl font-bold text-center mb-6 text-purple-600">Calculadora Neuronal</h1>
        
        {/* Selector de Operación */}
        <div className="flex justify-center gap-4 mb-6">
          <button 
            onClick={() => setOperacion('suma')}
            className={`px-4 py-2 rounded ${operacion === 'suma' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
          >
            Suma (+)
          </button>
          <button 
            onClick={() => setOperacion('resta')}
            className={`px-4 py-2 rounded ${operacion === 'resta' ? 'bg-red-500 text-white' : 'bg-gray-200'}`}
          >
            Resta (-)
          </button>
        </div>

        {/* Inputs */}
        <div className="space-y-4">
          <input 
            type="number" 
            placeholder="Número A" 
            value={valA} 
            onChange={e => setValA(e.target.value)}
            className="w-full p-2 border rounded"
          />
          <input 
            type="number" 
            placeholder="Número B" 
            value={valB} 
            onChange={e => setValB(e.target.value)}
            className="w-full p-2 border rounded"
          />
          
          <button 
            onClick={calcular}
            disabled={!modeloActual}
            className="w-full bg-purple-600 text-white py-3 rounded hover:bg-purple-700 disabled:bg-gray-400"
          >
            {modeloActual ? 'Calcular con IA' : 'Cargando Modelo...'}
          </button>
        </div>

        {/* Resultado */}
        {resultado && (
          <div className="mt-6 text-center p-4 bg-gray-50 rounded border">
            <p className="text-gray-500 text-sm">Resultado Predicho:</p>
            <p className="text-4xl font-bold text-gray-800">{resultado}</p>
          </div>
        )}
      </div>
    </div>
  );
}