import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
import evaluation_file
import time
import json


lang_dict = {
    'en': 'english',
    'es': 'spanish',
    'de': 'german',
    'zh': 'chinese',
}

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

def load_data(data_dir):
    """Load queries and corpus elements from a data directory."""
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


def write_run_file(results, run_file):
    """Write results to TREC format file."""
    print('Writing results to:', run_file.name)
    with open(run_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results))


def run_evaluation(qrels_path, run_file):
    """Run the evaluation script and return results as dict."""
    print('Evaluating with evaluation_file.py...')
    results = evaluation_file.evaluate_run(qrels_path, run_file)
    return results


def get_model_name(model):
    """Extract model name from the model object."""
    model_name = model[0].auto_model.config._name_or_path 
    return model_name.split("/")[-1]


# ========================
# EVALUATION FUNCTIONS
# ========================

def multilingual_evaluation(lang, model, device, source):
    """Evaluate model performance on multilingual data."""
    data_dir = project_dir / 'data' / source / lang_dict[lang]
    qrels_path = data_dir / "qrels.tsv"
    
    print('Loading data for language:', lang)
    queries_ids, queries_texts, corpus_ids, corpus_texts = load_data(data_dir)
    model_name = get_model_name(model)
    
    query_embeddings, corpus_embeddings = encode_data(model, queries_texts, corpus_texts, model_name, device)
    results = calculate_similarities_and_format_results(query_embeddings, corpus_embeddings, queries_ids, corpus_ids, model_name)
    
    run_file = output_dir / f"run_{lang}-{lang}_{model_name}.trec"
    write_run_file(results, run_file)
    
    evaluation_results = run_evaluation(qrels_path, run_file)
    print('Evaluation completed for language:', lang)
    return evaluation_results


def crosslingual_evaluation(lang, main_lang, model, device, source):
    """Evaluate model performance on cross-lingual data."""
    data_dir_main = project_dir / 'data' / source / lang_dict[main_lang]
    data_dir_target = project_dir / 'data' / source / lang_dict[lang]
    
    print('Loading data for crosslingual evaluation:', main_lang, '->', lang)
    
    # Load queries from main language
    queries = pd.read_csv(data_dir_main / "queries", sep="\t")
    queries_ids = queries.q_id.to_list()
    queries_texts = queries.jobtitle.to_list()
    
    # Load corpus from target language
    corpus_elements = pd.read_csv(data_dir_target / "corpus_elements", sep="\t")
    corpus_ids = corpus_elements.c_id.to_list()
    corpus_texts = corpus_elements.jobtitle.to_list()
    
    model_name = get_model_name(model)
    query_embeddings, corpus_embeddings = encode_data(model, queries_texts, corpus_texts, model_name, device)
    results = calculate_similarities_and_format_results(query_embeddings, corpus_embeddings, queries_ids, corpus_ids, model_name)
    
    run_file = output_dir / f"run_{main_lang}-{lang}_{model_name}.trec"
    write_run_file(results, run_file)
    
    print('Still not able to evaluate crosslingual runs. Skipping evaluation step.')
    return {m: np.nan for m in ["map", "mrr", "ndcg", "precision@5", "precision@10", "precision@100"]}


# ========================
# SAVING RESULTS
# ========================

def save_multilingual_results(multilingual_results, model_name, nickname, source):
    """Save multilingual evaluation results to a structured JSON file."""
    print("multilingual evaluations completed. Saving results...")

    json_path = output_dir / "results_multilingual.json"

    results_data = {
        "metadata": {
            "type": "multilingual",
            "model_name": model_name,
            "nickname": nickname,
            "source": source,
            "timestamp": today
        },
        "results": multilingual_results
    }

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(results_data, jf, indent=2, ensure_ascii=False)

    print(f"Saved multilingual results to {json_path}")


def save_crosslingual_results(crosslingual_results, model_name, nickname, source):
    """Save crosslingual evaluation results to a structured JSON file."""
    print("crosslingual evaluations completed. Saving results...")

    json_path = output_dir / "results_crosslingual.json"

    results_data = {
        "metadata": {
            "type": "crosslingual",
            "model_name": model_name,
            "nickname": nickname,
            "source": source,
            "timestamp": today
        },
        "results": crosslingual_results
    }

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(results_data, jf, indent=2, ensure_ascii=False)

    print(f"Saved crosslingual results to {json_path}")






# ========================
# RANKING
# ========================

def update_ranking(multilingual_results, crosslingual_results, model_name, nickname, source):
    """Update ranking CSV file with current execution results."""
    print("Updating ranking file...")
    
    ranking_file = project_dir / 'src' / 'output' / f"ranking_{source}.csv"
    execution_id = output_dir.name
    
    new_record = {
        'timestamp': today,
        'execution_id': execution_id,
        'model_name': model_name,
        'model_alias': nickname,
    }
    
    # MAP
    for lang in ['en', 'es', 'de', 'zh']:
        if lang in multilingual_results:
            new_record[f'map_{lang}_{lang}'] = multilingual_results[lang].get('map', np.nan)
        else:
            new_record[f'map_{lang}_{lang}'] = np.nan

    for key in crosslingual_results:
        lang_pair = key.replace('-', '_')
        new_record[f'map_{lang_pair}'] = crosslingual_results[key].get('map', np.nan)
    
    # avg_map - solo casos monolingües en, es, de (todos deben estar presentes)
    map_values = [
        new_record.get('map_en_en', np.nan),
        new_record.get('map_es_es', np.nan),
        new_record.get('map_de_de', np.nan)
    ]
    if any(np.isnan(v) for v in map_values):
        new_record['avg_map'] = np.nan
    else:
        new_record['avg_map'] = sum(map_values) / len(map_values)
    

    if ranking_file.exists():
        df = pd.read_csv(ranking_file)
    else:
        df = pd.DataFrame(columns=[
            'timestamp', 'execution_id', 'model_name', 'model_alias', 'avg_map',
            'map_en_en', 'map_es_es', 'map_de_de', 'map_zh_zh',
            'map_en_es', 'map_en_de', 'map_en_zh'
        ])
    
    # Double check for existing identical record
    comparison_cols = ['model_name', 'model_alias', 'avg_map',
                       'map_en_en', 'map_es_es', 'map_de_de', 'map_zh_zh',
                       'map_en_es', 'map_en_de', 'map_en_zh']
    
    if not df.empty:
        new_record_comparison = {k: new_record[k] for k in comparison_cols}
        existing_records = df[comparison_cols].to_dict('records')
        
        for existing in existing_records:
            if all(abs(existing.get(k, 0) - new_record_comparison.get(k, 0)) < 1e-6 
                   if isinstance(new_record_comparison.get(k), float) 
                   else existing.get(k) == new_record_comparison.get(k) 
                   for k in comparison_cols):
                print("Ya existe un registro idéntico en el ranking. No se agregará el nuevo registro.")
                return
    

    df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
    df = df.sort_values('avg_map', ascending=False).reset_index(drop=True)
    df.to_csv(ranking_file, index=False)
    
    print(f"Ranking actualizado en {ranking_file}")
    


# ========================
# EVALUATION PIPELINES
# ========================

def all_lang_evaluation(model_name, nickname, main_lang, device, source):
    """Run complete evaluation pipeline for all languages."""
    model = SentenceTransformer(model_name, device=device)

    multilingual_results = {
        lang: multilingual_evaluation(lang, model, device, source)
        for lang in lang_dict.keys()
    }
    save_multilingual_results(multilingual_results, model_name, nickname, source)

    crosslingual_results = {
        f"{main_lang}-{lang}": crosslingual_evaluation(lang, main_lang, model, device, source)
        for lang in lang_dict.keys() if lang != main_lang
    }
    save_crosslingual_results(crosslingual_results, model_name, nickname, source)

    update_ranking(multilingual_results, crosslingual_results, model_name, nickname, source)



def run_multilingual_only(model_name, nickname, device, source):
    model = SentenceTransformer(model_name, device=device)
    multilingual_results = {
        lang: multilingual_evaluation(lang, model, device, source)
        for lang in lang_dict.keys()
    }
    save_multilingual_results(multilingual_results, model_name, nickname, source)
    
    crosslingual_results = {}
    update_ranking(multilingual_results, crosslingual_results, model_name, nickname, source)



def run_crosslingual_only(model_name, nickname, main_lang, device, source):
    model = SentenceTransformer(model_name, device=device)
    crosslingual_results = {
        f"{main_lang}-{lang}": crosslingual_evaluation(lang, main_lang, model, device, source)
        for lang in lang_dict.keys() if lang != main_lang
    }
    save_crosslingual_results(crosslingual_results, model_name, nickname, source)

    multilingual_results = {}
    update_ranking(multilingual_results, crosslingual_results, model_name, nickname, source)



# ========================
# MAIN
# ========================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate sentence transformer models on crosslingual job title matching')
    parser.add_argument('--model', type=str, default='paraphrase-multilingual-MiniLM-L12-v2',
                        help='Model name from HuggingFace or local path')
    parser.add_argument('--source', type=str, default='validation', choices=['validation', 'test'],
                        help='Dataset source to use for evaluation')
    parser.add_argument('--nickname', type=str, default='default', help='Alias for the model in results')
    parser.add_argument('--main-lang', type=str, default='en', choices=['en', 'es', 'de', 'zh'],
                        help='Main language for crosslingual evaluation')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to run the model on')
    parser.add_argument('--mono-only', action='store_true', help='Run only multilingual evaluations')
    parser.add_argument('--multi-only', action='store_true', help='Run only crosslingual evaluations')
    args = parser.parse_args()

    if args.mono_only and args.multi_only:
        parser.error("Cannot use --mono-only and --multi-only together")

    if args.mono_only:
        run_multilingual_only(args.model, args.nickname, args.device, args.source)
    elif args.multi_only:
        run_crosslingual_only(args.model, args.nickname, args.main_lang, args.device, args.source)
    else:
        all_lang_evaluation(args.model, args.nickname, args.main_lang, args.device, args.source)


if __name__ == "__main__":
    main()
