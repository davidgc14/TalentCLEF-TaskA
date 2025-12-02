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

Ahora se estan ejecutando los modelos de la tabla de MTEB leaderboard. Se esta empezando con los modelos más pesados, para luego poder evaluarlos todos de manera continua con un bucle for. Se estan guardando los resultados en la carpeta models/results/mteb

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
