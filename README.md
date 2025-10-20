# TalentCLEF TaskA: Evaluación de similitudes

## Análisis previo

### Descripción del corpus

https://talentclef.github.io/talentclef/docs/talentclef-2025/data/description_corpus/

### Analisis de datos

En la evaluación, tenemos tres archivos:

corpus_elements: Conjunto nuevo de trabajos sobre el cual debemos encontrar la similitud frente a uno dado

queries: solicitudes que pretenden encontrar, contrastando con todos los de corpus_elements, indicar si existe similitud (1) o no (0). Para eso habrá que definir un threshold a futuro para ver cuales colapsan a 1 y cuales a 0.

qrels: relaciones entre las queries y cada uno de los elementos del corpus. Solo se muestran los que si presentan relacion (cuarta columna, indicando con un 1).

Los datos de las queries pueden o no pertenecer al corpus conocido (la idea es que no necesariamente, y se construya pensando que no va a haber una solicitud en nuestro corpus). He encontrado algunos casos en los que si aparece en ambos. Ejemplo: lawyer

## Primera aproximación: Embeddings preentrenados

Para la primera aproximación he optado por la forma más sencilla de abordarlo: no entrenar ningún modelo, sino usar un modelo ya entrenado en otros corpus de texto. Para ello me he servido del modelo _sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2_ que está preparado para ser multilingüe. Con este modelo obtenemos las representaciones vectoriales en el espacio de embedding preentrenado de 384 dimensiones. (Para monolingüe inglés también está disponible _sentence-transformers/all-MiniLM-L6-v2_).

Para medir su similitud he usado la similitud coseno, obteniendo una métrica definida en el intervalo [0,1].

Para decidir el threshold, me surge la duda de si puedo usar los datos de validación como herramienta para conseguir una métrica aproximada. Mi idea es: una vez obtenidas las medidas de similitud para cada par de títulos, dispongo en validación de una lista que sí se encuentran por encima del threshold (ya que hay que convertir esta medida de similitud continua en una discreta (0,1)). ¿Puedo usar estos datos para intentar hacer clustering y ver qué índice maximiza la cantidad de verdaderos positivos y minimiza los falsos?

Quizás puedo usar los datos de entrenamiento en este sentido, aunque igual el método es un poco enrevesado. Usando los pares que me proporciona el entrenamiento, tengo seguro cuáles sí puedo considerar como parejas válidas. A partir de ellas puedo generar parejas que no tengan relación aparente (o por lo menos que no haya una conexión directa) para conseguir un conjunto de datos que NO son parejas deseables (inferiores al threshold). Con ello conseguiría los dos conjuntos para poder aplicar clustering.

Para descubrir estas conexiones, puedo usar grafos. Tras generar el grafo resultante de los pares de conexiones (parejas) que hay en el entrenamiento, sólo tendría que unir pares de títulos que pertenecieran a componentes conexas distintas.

Por ahora voy a intentar abordarlo con los datos de validación, pero por lo menos sabemos que existe una manera de abordarlo usando únicamente los datos de entrenamiento.
