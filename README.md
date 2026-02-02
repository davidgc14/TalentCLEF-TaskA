# Emparejamiento Inteligente de Titulaciones Profesionales (TFM - UNED)

**Trabajo Fin de Máster en Lenguajes y Sistemas Informáticos**

- **Universidad:** Universidad Nacional a Distancia (UNED)
- **Fecha:** Febrero 2026

Este repositorio contiene el código fuente, los datos y la documentación asociada al TFM titulado _"Emparejamiento inteligente de titulaciones profesionales: una propuesta basada en Transformers optimizada para el español"_.

El proyecto aborda la **Tarea A de TalentCLEF 2025** (Multilingual Job Title Matching), proponiendo una arquitectura en cascada diseñada para maximizar la precisión en la recuperación de candidatos, con especial énfasis en el idioma español.

## Resumen del Proyecto

El objetivo del sistema es, dada una titulación profesional (_query_), identificar y ordenar las titulaciones más similares de una base de conocimiento (_corpus_).

La solución propuesta implementa una **Arquitectura en Cascada** de tres etapas:

1.  **Retrieval (Recuperación Densa):** Uso de un _ensemble_ de tres modelos Bi-Encoders (basados en Sentence Transformers) ajustados (_fine-tuned_) para recuperar candidatos eficientemente.
2.  **Ranking Fusion:** Combinación de los rankings preliminares utilizando la técnica de Suma Ponderada (**WSum**) para mitigar sesgos individuales y mejorar la robustez.
3.  **Re-ranking:** Refinamiento de los mejores candidatos (_top-k_) mediante un modelo **Cross-Encoder**, que evalúa la interacción semántica profunda entre la query y los documentos.

### Resultados Destacados

El sistema alcanzó la **3ª posición global por equipos** en la competición TalentCLEF 2025.

| Métrica |     Idioma      | Dataset | Puntuación (MAP) |
| :------ | :-------------: | :-----: | :--------------: |
| MAP     | Español (es-es) |  Test   |    **0.497**     |
| MAP     | Inglés (en-en)  |  Test   |    **0.551**     |
| MAP     | Alemán (de-de)  |  Test   |    **0.508**     |

---

## Estructura del Repositorio

El proyecto sigue una estructura modular organizada de la siguiente manera:

```text
.
├── data/                   # Datos de la competición (queries, corpus, qrels)
├── doc/                    # Documentación y papers
├── models/                 # Directorio para guardar modelos (NO incluidos en repo por tamaño)
├── notebooks/              # Notebooks utilizados para Fine-Tuning, Fusión y Análisis
├── output/                 # Resultados de las ejecuciones (formato TREC y JSON)
├── src/                    # Código fuente del sistema
│   ├── main.py             # Pipeline principal de ejecución
│   └── evaluation_file.py  # Script de evaluación (ranx)
├── latex/                  # Código fuente LaTeX de la memoria
└── requirements.txt        # Dependencias del proyecto

```

---

## Instalación y Requisitos

El proyecto se ha desarrollado utilizando **Python 3.12**. Se recomienda el uso de un entorno virtual y, preferiblemente, ejecución en una máquina con GPU (CUDA) para la inferencia de los modelos.

1. **Clonar el repositorio:**

```bash
git clone https://github.com/davidgc14/TalentCLEF-TaskA.git
cd TalentCLEF-TaskA

```

2. **Instalar dependencias:**

```bash
pip install -r requirements.txt

```

Las principales librerías utilizadas son:

- `sentence-transformers`
- `ranx` (para evaluación y fusión)
- `pandas` y `numpy`
- `torch`

---

## Modelos y Pesos

**NOTA IMPORTANTE:** Debido al gran tamaño de los archivos (varios GB), los pesos de los modelos entrenados **no están incluidos en este repositorio**.

El sistema está diseñado para buscar modelos en la carpeta `models/`. Tienes dos opciones para ejecutar el proyecto:

### Opción A: Usar modelos base de HuggingFace

Puedes ejecutar el script pasando los nombres de los modelos originales de HuggingFace como argumentos.

### Opción B: Replicar el entrenamiento

Puedes generar tus propios modelos ajustados utilizando el notebook proporcionado:

1. Abrir `notebooks/colab_FineTuning.ipynb`.
2. Ejecutar el entrenamiento con los datos de la carpeta `data/`.
3. Guardar los modelos resultantes en la estructura `models/finetuned models/`.

**NOTA**: Probablemente haya que ajustar rutas a directorios previo a su ejecución.

Los modelos base utilizados en la arquitectura final fueron:

- `Lajavaness/bilingual-embedding-large`
- `intfloat/multilingual-e5-large-instruct`
- `Alibaba-NLP/gte-multilingual-base`
- Reranker: `Jsevisal/CrossEncoder-ModernBERT-base-qnli`

---

## Ejecución

El punto de entrada principal es el script `src/main.py`. Este script carga los datos, genera embeddings, realiza la fusión y aplica el re-ranking.

### Uso básico

Para ejecutar el pipeline completo (Retrieval + Fusión + Rerank) en el conjunto de validación para español:

```bash
python src/main.py \
    --lang es \
    --sources validation \
    --device cuda \
    --run-name "mi_ejecucion_test"

```

### Argumentos Disponibles

| Argumento          | Descripción                                 | Valor por defecto     |
| ------------------ | ------------------------------------------- | --------------------- |
| `--model1`         | Ruta o nombre del primer modelo bi-encoder  | `models/finetuned...` |
| `--model2`         | Ruta o nombre del segundo modelo bi-encoder | `models/finetuned...` |
| `--model3`         | Ruta o nombre del tercer modelo bi-encoder  | `models/finetuned...` |
| `--sources`        | Datasets a evaluar (`validation`, `test`)   | `validation test`     |
| `--lang`           | Idioma objetivo (`en`, `es`, `de`, `zh`)    | `es`                  |
| `--device`         | Dispositivo de inferencia (`cpu`, `cuda`)   | `cpu`                 |
| `--top-k`          | Número de documentos para re-ranking        | `5`                   |
| `--skip-reranking` | Flag para saltar la etapa de cross-encoder  | `False`               |

**NOTA**: El orden de los modelos es relevante, ya que cada modelo tiene un peso asociado para la fusión de ranking.

### Ejemplo: Ejecución con modelos base

Si no dispones de los modelos entrenados localmente, utiliza este comando para probar el sistema:

```bash
python src/main.py \
    --model1 "Lajavaness/bilingual-embedding-large" \
    --model2 "intfloat/multilingual-e5-large-instruct" \
    --model3 "Alibaba-NLP/gte-multilingual-base" \
    --lang es \
    --device cuda \
    --top-k 5

```

### Salida

Los resultados se generarán automáticamente en la carpeta `output/<FECHA>/<ID>/`. Se generan:

1. **Archivos .trec:** Rankings compatibles con la herramienta de evaluación oficial.
2. **Archivos .json:** Métricas de evaluación (MAP, P@5, etc.) calculadas sobre el set de validación (si aplica).

---

## Documentación

Para una explicación detallada de la metodología, los experimentos y el análisis de resultados, consulta la memoria del proyecto:

- **Memoria TFM:** [TFM.pdf](https://github.com/davidgc14/TalentCLEF-TaskA/blob/master/latex/TFM.pdf) (o en la carpeta `latex/TFM.pdf`).
- **Diagramas:** Disponibles en `latex/imagenes/`.
