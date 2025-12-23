<h1 style="font-size: 3em;">TalentCLEF TaskA: Evaluación de similitudes</h1>

# Arquitectura

Me encuentro actualmente en busqueda del mejor modelo de embedding para la tarea de Retrieval, ya sea con finetunning propio o no.

Posteriormente habra que buscar un modelo de reranking.

# Embedding

## Notas sobre la evaluación previa a finetunning

### Hugging face

se han ejecutado todos los modelos de hugging face que atendian a los siguientes filtros:

- Libreria Transformers
- Contener idioma español
- Tarea Sentence similarity

estos estan guardados en la tabla hf_evaluation_results.csv

### MTEB

Ahora se estan ejecutando los modelos de la tabla de MTEB leaderboard (obtenida a dia 1 de diciembre de 2025). Se esta empezando con los modelos más pesados, para luego poder evaluarlos todos de manera continua con un bucle for. Se estan guardando los resultados en la carpeta models/results/mteb

Para obtener esta tabla se ha aplicado el siguiente filtro:

- Tarea Retrieval
- Modelo Open Source

Aqui no se ha aplicado filtro de idioma, ya que la gran mayoria son multilingues. La presencia de modelos monolingues para la resolucion de esta tarea es escaso.

He tomado el top 100 aplicado el filtro anterior, y de estos he intentado evaluarlos todos. Sin embargo, por escasez de recursos ha sido bastante complicado ejecutar aquellos que requirieran de más de 20GB de almacenamiento para ejecutarse. En concreto se ha descartado la ejecucción de los siguientes modelos:

- Puesto 16 GritLM/GritLM-8x7B Memoria necesaria: 89079 Mb
- Puesto 2 tencent/KaLM-Embedding-Gemma3-12B-2511 Memoria necesaria: 44884 Mb
- Puesto 7 Alibaba-NLP/gte-Qwen2-7B-instruct Memoria necesaria: 29040 Mb
- Puesto 51 McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised Memoria necesaria: 27126 Mb

### Español

Se ha buscado autores especificos de modelos españoles para tener una muestra de evaluación. Los mas destacados son "somosnlp", "PlanTL-GOB-ES" y "dccuchile". Además he investigado sobre algunos especificos, y he generado una lista con los modelos en español para ver que tal rinden. Estan guardados en su carpeta correspondiente

# Tutoria y next steps

La arquitectura de hacer fusion ranking de varios modelos y luego reranking tiene sentido. El fusion ranking existen librerias que mezclan y ponderan los rankings automaticamente. Debo justificar el uso de las librerias en concreto que use.

## Arquitectura de sistema

Aqui debo de hablar de:

- Arquitectura general
- Cada uno de los tres bloques de la pipeline
- Las tecnologias usadas (funciones de pesado, funciones de perdida, librerias, teconologias...)
- Describir el flujo de trabajo (que cosas voy a probar, el orden de las pruebas...)

## Resultados

Aqui debo mostrar todo el flujo de trabajo que me ha llevado a seleccionar el modelo final. Además debo definir que la medida MAP será la que determinará el rendimiento general de una solución.

### Bloque 1: Seleccion de modelos base para entrenar

- Todos los modelos que se han probado
- Comparacion de tamaños y rendimiento
- Comparación monolingue multilingue (comprobar que en los resultados de los participantes nadie uso monolingue. Puede ser que el tamaño de los modelos importe. Tambien puede ser que la terminología usada en los datos de entrenamiento o finetuning no se equiparen con el vocabulario usado en la realidad [por ejemplo, la palabra software o manager son palabras que hoy en dia se usan en el castellano como palabras traidas del ingles]). Enunciar posibles razones, pero NO una razon definitiva.
- Finetunning y seleccion final de candidatos

### Bloque 2: Prueba de varias tecnicas de fusion ranking

No usar demasiadas, ya que lo groso de la experimentacion esta en los pasos anteriores.
https://gemini.google.com/share/8dd4fd582e49

### Bloque 3: Prueba de varias tecnicas de reranking

Igual que el anterior

## Evaluacion

Mostrar los resultados

## Discusion

Entrar mas en detalle en el analisis de los resultados y de la evaluacion (seleccion de multilingue por ejemplo)

Incluir un analisis de errores (muy tipico en articulos cientificos): Que es lo que podria haber inducido a los resultados, que cosas han podido fallar... Investigar acerca de como hacer bien un analisis de errores.
