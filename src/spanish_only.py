import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
from evaluation_file import evaluate_run
import time
import json
import tempfile
import os
import argparse
import time


today = time.strftime("%Y-%m-%d")

project_dir = Path(__file__).resolve().parents[1]
date_dir = project_dir / 'src' / 'output' / today
date_dir.mkdir(parents=True, exist_ok=True)

# Crear subdirectorio incremental por ejecución (001, 002, ...)
exec_dirs = sorted([d for d in date_dir.iterdir() if d.is_dir() and d.name.isdigit() and len(d.name) == 3])
if exec_dirs and not any(exec_dirs[-1].iterdir()):
    output_dir = exec_dirs[-1]
else:
    next_id = int(exec_dirs[-1].name) + 1 if exec_dirs else 1
    output_dir = date_dir / f"{next_id:03d}"
    output_dir.mkdir(exist_ok=True)


# ========================
# DATA LOADING AND ENCODING
# ========================

def load_spanish_data(data_dir):
    """Load queries and corpus elements from Spanish data directory."""
    queries_path = data_dir / "queries"
    corpus_elements_path = data_dir / "corpus_elements"
    
    queries = pd.read_csv(queries_path, sep="\t")
    corpus_elements = pd.read_csv(corpus_elements_path, sep="\t")
    
    return (
        queries.q_id.to_list(),
        queries.jobtitle.to_list(),
        corpus_elements.c_id.to_list(),
        corpus_elements.jobtitle.to_list(),
    )


def encode_data(model, queries_texts, corpus_texts, model_name, device):
    """Encode queries and corpus using the model."""
    print('Encoding data:', model_name, 'on device:', device)
    query_embeddings = model.encode(queries_texts, convert_to_tensor=True, show_progress_bar=True)
    corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True, show_progress_bar=True)
    return query_embeddings, corpus_embeddings


def calculate_similarities_and_format_results(query_embeddings, corpus_embeddings, queries_ids, corpus_ids, model_name):
    """Calculate cosine similarities and format results in TREC format."""
    print('Calculating similarities and preparing results...')
    similarities = util.cos_sim(query_embeddings, corpus_embeddings).cpu().numpy()
    
    results = []
    for q_idx, q_id in enumerate(queries_ids):
        sorted_indices = np.argsort(-similarities[q_idx])  # Orden descendente
        for rank, c_idx in enumerate(sorted_indices):  
            doc_id = corpus_ids[c_idx]
            score = similarities[q_idx, c_idx]
            results.append(f"{str(q_id)} Q0 {str(doc_id)} {rank+1} {score:.4f} {model_name}")
    return results


def run_evaluation_temp(qrels_path, results, model_name):
    """Run evaluation using temporary file without saving."""
    print('Evaluating Spanish monolingual performance...')
    
    # Crear archivo temporal
    with tempfile.NamedTemporaryFile(mode='w', suffix='.trec', delete=False, encoding='utf-8') as tmp_file:
        tmp_file.write("\n".join(results))
        tmp_path = tmp_file.name
    
    try:
        evaluation_results = evaluate_run(qrels_path, tmp_path)
    finally:
        # Eliminar archivo temporal
        os.unlink(tmp_path)
    
    return evaluation_results


def get_model_name(model):
    """Extract model name from the model object."""
    model_name = model[0].auto_model.config._name_or_path 
    return model_name.split("/")[-1]


# ========================
# EVALUATION FUNCTION
# ========================

def spanish_monolingual_evaluation(model, device, source):
    """Evaluate model performance on Spanish monolingual data."""
    data_dir = project_dir / 'data' / source / 'spanish'
    qrels_path = data_dir / "qrels.tsv"
    
    print('Loading Spanish data...')
    queries_ids, queries_texts, corpus_ids, corpus_texts = load_spanish_data(data_dir)
    model_name = get_model_name(model)
    
    query_embeddings, corpus_embeddings = encode_data(model, queries_texts, corpus_texts, model_name, device)
    results = calculate_similarities_and_format_results(query_embeddings, corpus_embeddings, queries_ids, corpus_ids, model_name)
    
    evaluation_results = run_evaluation_temp(qrels_path, results, model_name)
    print('Spanish evaluation completed')
    return evaluation_results


# ========================
# SAVING RESULTS
# ========================

def save_spanish_results(evaluation_results, model_name, nickname, source):
    """Save Spanish monolingual evaluation results to JSON file."""
    print("Saving Spanish evaluation results...")

    json_path = output_dir / "results_spanish_monolingual.json"

    results_data = {
        "metadata": {
            "type": "spanish_monolingual",
            "model_name": model_name,
            "nickname": nickname,
            "source": source,
            "timestamp": today
        },
        "results": evaluation_results
    }

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(results_data, jf, indent=2, ensure_ascii=False)

    print(f"Saved Spanish results to {json_path}")


# ========================
# RANKING
# ========================

def update_spanish_ranking(map_score, model_name, nickname, source):
    """Update Spanish-specific ranking CSV file with current execution results."""
    print("Updating Spanish ranking file...")
    
    ranking_file = project_dir / 'src' / 'output' / f"ranking_spanish_{source}.csv"
    execution_id = output_dir.name
    
    new_record = {
        'timestamp': today,
        'execution_id': execution_id,
        'model_name': model_name,
        'model_alias': nickname,
        'map_es_es': map_score
    }
    
    if ranking_file.exists():
        df = pd.read_csv(ranking_file)
    else:
        df = pd.DataFrame(columns=['timestamp', 'execution_id', 'model_name', 'model_alias', 'map_es_es'])
    
    # Double check for existing identical record
    comparison_cols = ['model_name', 'model_alias', 'map_es_es']
    
    if not df.empty:
        new_record_comparison = {k: new_record.get(k, np.nan) for k in comparison_cols}
        existing_records = df[comparison_cols].to_dict('records')
        
        for existing in existing_records:
            if all(abs(existing.get(k, np.nan) - new_record_comparison.get(k, np.nan)) < 1e-6 
                   if isinstance(new_record_comparison.get(k), (float, int)) and not np.isnan(new_record_comparison.get(k, np.nan))
                   else existing.get(k) == new_record_comparison.get(k) 
                   for k in comparison_cols):
                print("Ya existe un registro idéntico en el ranking español. No se agregará el nuevo registro.")
                return
    
    # Evitar warning de pandas con DataFrame vacío
    if df.empty:
        df = pd.DataFrame([new_record])
    else:
        df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
    
    df = df.sort_values('map_es_es', ascending=False).reset_index(drop=True)
    df.to_csv(ranking_file, index=False)
    
    print(f"Ranking español actualizado en {ranking_file}")


# ========================
# MAIN PIPELINE
# ========================

def run_spanish_evaluation(model_name, nickname, device, source):
    """Run complete Spanish monolingual evaluation pipeline."""
    model = SentenceTransformer(model_name, device=device)
    
    evaluation_results = spanish_monolingual_evaluation(model, device, source)
    save_spanish_results(evaluation_results, model_name, nickname, source)
    
    map_score = evaluation_results.get('map', np.nan)
    update_spanish_ranking(map_score, model_name, nickname, source)
    
    print(f"\n{'='*50}")
    print(f"Evaluación completada para {model_name}")
    print(f"MAP español-español: {map_score:.4f}")
    print(f"{'='*50}\n\n")


# ========================
# MAIN
# ========================

def main():
    parser = argparse.ArgumentParser(description='Evaluate sentence transformer models on Spanish monolingual job title matching')
    parser.add_argument('--model', type=str, default='paraphrase-multilingual-MiniLM-L12-v2',
                        help='Model name from HuggingFace or local path')
    parser.add_argument('--source', type=str, default='validation', choices=['validation', 'test'],
                        help='Dataset source to use for evaluation')
    parser.add_argument('--nickname', type=str, default='default', help='Alias for the model in results')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to run the model on')
    args = parser.parse_args()

    start_time = time.time()
    run_spanish_evaluation(args.model, args.nickname, args.device, args.source)
    end_time = time.time()

    print(f"Tiempo de ejecución: {round(end_time - start_time)} segundos.\n")


if __name__ == "__main__":
    main()