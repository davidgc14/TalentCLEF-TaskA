# TalentCLEF Task A 2025 - Multilingual Job Title Matching

Sistema de evaluación para modelos en tareas de matching de títulos de trabajo multilingües y cross-lingües.

## Descripción

Este proyecto implementa un framework de evaluación para la **TalentCLEF Task A 2025**, que consiste en la búsqueda y emparejamiento de títulos de trabajo en múltiples idiomas utilizando modelos de embeddings semánticos.

El sistema evalúa:

- **Evaluación Monolingüe**: Queries y corpus en el mismo idioma (en-en, es-es, de-de, zh-zh)
- **Evaluación Cross-lingüe**: Queries en un idioma y corpus en otro (en-es, en-de, en-zh)

## Estructura del Proyecto

```
.
├── data/                    # Conjunto de datos de entrenamiento, testeo y validación
├── src/
│   ├── main.py              # Pipeline principal de evaluación
│   ├── evaluation_file.py   # Script de evaluación con ranx
│   └── output/              # Resultados de las ejecuciones
│       ├── YYYY-MM-DD/
│       │   └── XXX/         # Índice de la ejecución
│       └── ranking_validation.csv
├── doc/                     # Documentación y papers de referencia
├── models/                  # Listado de modelos preentrenados
└── workspace/               # Documentación LaTeX del TFM
```

## Requisitos

Versión python 3.12.3

```bash
pandas
numpy
sentence-transformers
ranx
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Uso

```bash
python src/main.py \
  --model "paraphrase-multilingual-MiniLM-L12-v2" \
  --source validation \
  --nickname "mini-lm" \
  --main-lang en \
  --device cuda
```

## Parámetros

| Parámetro      | Descripción                                   | Valores                | Por defecto                             |
| -------------- | --------------------------------------------- | ---------------------- | --------------------------------------- |
| `--model`      | Nombre del modelo en HuggingFace o ruta local | string                 | `paraphrase-multilingual-MiniLM-L12-v2` |
| `--source`     | Dataset a utilizar                            | `validation`, `test`   | `validation`                            |
| `--nickname`   | Alias del modelo en resultados                | string                 | `default`                               |
| `--main-lang`  | Idioma principal para evaluación cross-lingüe | `en`, `es`, `de`, `zh` | `en`                                    |
| `--device`     | Dispositivo de ejecución                      | `cpu`, `cuda`          | `cpu`                                   |
| `--mono-only`  | Solo evaluación monolingüe                    | flag                   | -                                       |
| `--multi-only` | Solo evaluación cross-lingüe                  | flag                   | -                                       |

## Formato de Datos

### Queries

```
q_id	jobtitle
1	    nanny
2	    food technologist
3	    broadcast engineer
4	    automation engineer
...
```

### Corpus Elements

```
c_id	jobtitle
1	    recording engineer
2	    director of taxation
3	    technical support representative
4	    hr manager
...
```

### QRels (Ground Truth)

```
q_id    iter    doc_id    rel
1       0       100       1
1       0       105       1
```

## Métricas de Evaluación

El sistema calcula las siguientes métricas usando la librería **ranx**:

- **MAP** (Mean Average Precision): Precisión promedio sobre todas las queries
- **MRR** (Mean Reciprocal Rank): Reciprocidad del primer resultado relevante
- **NDCG** (Normalized Discounted Cumulative Gain): Ganancia acumulativa normalizada
- **Precision@5**: Precisión en los primeros 5 resultados
- **Precision@10**: Precisión en los primeros 10 resultados
- **Precision@100**: Precisión en los primeros 100 resultados

Notar que las evaluaciones cross-lingües **no tienen qrels disponibles**, sólo se evalúan en Codabench, por lo que retornan `NaN` por defecto.

## Resultados

### Archivos Generados

Cada ejecución crea una carpeta incremental en `src/output/YYYY-MM-DD/XXX/` con:

1. **Archivos TREC**: Resultados de ranking en formato TREC

   - `run_{lang1}-{lang2}_{model_name}.trec`

2. **results_multilingual.json**: Resumen de métricas de evaluación monolingüe

3. **results_crosslingual.json**: Resumen de métricas de evaluación cross-lingüe

### Ranking Global

El archivo `ranking_validation.csv` mantiene un ranking histórico de todos los modelos evaluados, evitando la repetición de registros y ordenando los resultados de mayor a menor MAP medio.

- Los valores cross-lingües aparecen como `NaN` cuando no hay qrels disponibles para evaluación
- El `avg_map` se calcula **solo con los casos monolingües en, es, de** (excluyendo zh)
- Si alguno de los tres valores (en-en, es-es, de-de) es NaN, entonces avg_map será NaN

## Pipeline de Ejecución

1. **Carga de datos**: Lee queries y corpus elements de los directorios correspondientes
2. **Encoding**: Genera embeddings usando el modelo de sentence transformers
3. **Cálculo de similitudes**: Calcula similitud coseno entre queries y corpus
4. **Formato TREC**: Genera archivos de resultados en formato TREC estándar
5. **Evaluación**: Calcula métricas usando qrels (solo para validation)
6. **Almacenamiento**: Guarda resultados en JSON y actualiza ranking CSV

## Documentación

- Documentación oficial en `doc/lab-overview.pdf`
- Tutoriales en `doc/tutorials/`
- Papers de resultados de la competición en `doc/aproximations/`

---
