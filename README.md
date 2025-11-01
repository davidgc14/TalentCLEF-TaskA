<h1 style="font-size: 3em;">TalentCLEF TaskA: Evaluación de similitudes</h1>

# Análisis previo

## Descripción del corpus

https://talentclef.github.io/talentclef/docs/talentclef-2025/data/description_corpus/

## Analisis de datos

En la evaluación, tenemos tres archivos:

corpus_elements: Conjunto nuevo de trabajos sobre el cual debemos encontrar la similitud frente a uno dado

queries: solicitudes que pretenden encontrar, contrastando con todos los de corpus_elements, si existe similitud (1) o no (0). Para eso habrá que definir un threshold a futuro para ver cuales colapsan a 1 y cuales a 0. (Updated: Creo que no es necesario, pues el propio archivo de evaluación construye el ranking y evalua cómo de buena es la similitud)

qrels: relaciones entre las queries y cada uno de los elementos del corpus. Solo se muestran los que si presentan relación (cuarta columna, indicando con un 1).

Los datos de las queries pueden o no pertenecer al corpus conocido (la idea es que no necesariamente, y se construya pensando que no va a haber una solicitud en nuestro corpus). He encontrado algunos casos en los que si aparece en ambos. Ejemplo: lawyer

Resulta que los datos DE VALIDACIÓN se han obtenido de applications a trabajos que ha hecho la gente, donde las queries son las ofertas, y los titulos del corpus elements son los títulos de la gente que ha aplicado al puesto en cuestión. El training set se ha obtenido del ESCO directamente.

# Primera aproximación: Embeddings preentrenados

Para la primera aproximación he optado por la forma más sencilla de abordarlo: no entrenar ningún modelo, sino usar un modelo ya entrenado en otros corpus de texto. Para ello me he servido del modelo _sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2_ que está preparado para ser multilingüe. Con este modelo obtenemos las representaciones vectoriales en el espacio de embedding preentrenado de 384 dimensiones. (Para monolingüe inglés también está disponible _sentence-transformers/all-MiniLM-L6-v2_).

Para medir su similitud se usa la similitud coseno, obteniendo una métrica definida en el intervalo [0,1].

## Búsqueda del threshold (deprecated?)

_Nota: Esta sección se desarrolló sin conocer el archivo de evaluación. Se mantiene por si resulta útil en el futuro, pero a priori esta sección es completamente prescindible._

Para decidir el threshold, he abordado su búsqueda gracias a los datos de entrenamiento. Usando los pares que me proporciona el entrenamiento, tengo seguro cuáles sí puedo considerar como parejas válidas. A partir de ellas puedo generar parejas que no tengan relación aparente (o por lo menos que no haya una conexión directa en los datos de entrenamiento) para conseguir un conjunto de datos que NO son parejas deseables (inferiores al threshold). Con ello consigo los dos conjuntos para poder aplicar clasificación binaria basada en un valor límite (threshold).

Para descubrir estas conexiones, he usado grafos. Tras generar el grafo resultante de los pares de conexiones (parejas) que hay en el entrenamiento, he obtenido las distintas componentes conexas, mediante las cuales he podido generar las parejas de términos que no están relacionados a priori.

Una vez obtenido el threshold que maximiza la F1, he aplicado el modelo mencionado anteriormente para generar un embedding y con él una similitud coseno entre los pares de titlejobs que se encuentran en el conjunto de validación. Tras ello, he obtenido una matriz de tamaño query x corpus_element con las predicciones del modelo base, colapsando a 0 o 1 según si superaba el umbral encontrado. Tras ello se ha aplicado el classification report.

**TO-DO**:

- Tengo que volver a comprobar si puedo separar por grupos, en lugar por componentes conexas, usando el family_id. Al parecer es el ID del ESCO, por lo que puede diferir de las componentes conexas, habiendo varias componentes conexas que puedan pertenecer al mismo family_id, o al revés.

## Evaluación

Para la evaluación me he servido del tutorial proporcionado por la organización (que lo he adaptado a mi caso en el archivo `evaluation_path.ipynb`). Este flujo de evaluación se sirve del archivo `talentclef_evaluate.py`, el cual contiene todas las métricas de evaluación para una correcta valoración del modelo. Ahora mismo se encuentra preparado con el modelo monolingüe y los datos de validación para inglés, aunque supuestamente sería necesario generar lo mismo para los cuatro idiomas.

**TO-DO**:

- Construir un archivo que procese la evaluación monolingue, a la cual se le pase como parámetro el lenguaje que se quiere evaluar, y trabajar a partir de ahí con rutas a los archivos de evaluación.
- COnstruir un archivo para el procesamiento multilingue, al que se le pasen como parametros los dos idiomas que se deben cruzar.
- Construir un archivo main.py que ejecute la evaluación para cada uno de los idiomas, asi como los casos cross-lingual.
