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

def monolingual_evaluation(lang, model, device, source):
    """Evaluate model performance on monolingual data."""
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


def multilingual_evaluation(lang, main_lang, model, device, source):
    """Evaluate model performance on cross-lingual data."""
    data_dir_main = project_dir / 'data' / source / lang_dict[main_lang]
    data_dir_target = project_dir / 'data' / source / lang_dict[lang]
    
    print('Loading data for multilingual evaluation:', main_lang, '->', lang)
    
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
    
    print('Still not able to evaluate multilingual runs. Skipping evaluation step.')
    return {m: 0.0 for m in ["map", "mrr", "ndcg", "precision@5", "precision@10", "precision@100"]}


# ========================
# SAVING RESULTS
# ========================

def save_monolingual_results(monolingual_results, model_name, nickname):
    print("Monolingual evaluations completed. Saving results...")
    txt_path = output_dir / "results.txt"
    json_path = output_dir / "results.json"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== Evaluation Results for monolingual ===\n")
        f.write(f"Model name: {model_name}\nModel alias: {nickname}\n\n")
        for lang, res in monolingual_results.items():
            f.write(f"=== Language: {lang} ===\n")
            for metric, score in res.items():
                f.write(f"{metric}: {score:.4f}\n")
            f.write("\n")

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(monolingual_results, jf, indent=2, ensure_ascii=False)


def save_multilingual_results(multilingual_results, model_name, nickname):
    print("Multilingual evaluations completed. Saving results...")
    txt_path = output_dir / "results_multilingual.txt"
    json_path = output_dir / "results_multilingual.json"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== Evaluation Results for multilingual ===\n")
        f.write(f"Model name: {model_name}\nModel alias: {nickname}\n\n")
        for lang_pair, res in multilingual_results.items():
            f.write(f"=== Language Pair: {lang_pair} ===\n")
            for metric, score in res.items():
                f.write(f"{metric}: {score:.4f}\n")
            f.write("\n")

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(multilingual_results, jf, indent=2, ensure_ascii=False)


# ========================
# EVALUATION PIPELINES
# ========================

def all_lang_evaluation(model_name, nickname, main_lang, device, source):
    """Run complete evaluation pipeline for all languages."""
    model = SentenceTransformer(model_name, device=device)

    monolingual_results = {
        lang: monolingual_evaluation(lang, model, device, source)
        for lang in lang_dict.keys()
    }
    save_monolingual_results(monolingual_results, model_name, nickname)

    multilingual_results = {
        f"{main_lang}-{lang}": multilingual_evaluation(lang, main_lang, model, device, source)
        for lang in lang_dict.keys() if lang != main_lang
    }
    save_multilingual_results(multilingual_results, model_name, nickname)


def run_monolingual_only(model_name, nickname, device, source):
    model = SentenceTransformer(model_name, device=device)
    monolingual_results = {
        lang: monolingual_evaluation(lang, model, device, source)
        for lang in lang_dict.keys()
    }
    save_monolingual_results(monolingual_results, model_name, nickname)


def run_multilingual_only(model_name, nickname, main_lang, device, source):
    model = SentenceTransformer(model_name, device=device)
    multilingual_results = {
        f"{main_lang}-{lang}": multilingual_evaluation(lang, main_lang, model, device, source)
        for lang in lang_dict.keys() if lang != main_lang
    }
    save_multilingual_results(multilingual_results, model_name, nickname)


# ========================
# MAIN
# ========================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate sentence transformer models on multilingual job title matching')
    parser.add_argument('--model', type=str, default='paraphrase-multilingual-MiniLM-L12-v2',
                        help='Model name from HuggingFace or local path')
    parser.add_argument('--source', type=str, default='validation', choices=['validation', 'test'],
                        help='Dataset source to use for evaluation')
    parser.add_argument('--nickname', type=str, default='default', help='Alias for the model in results')
    parser.add_argument('--main-lang', type=str, default='en', choices=['en', 'es', 'de', 'zh'],
                        help='Main language for multilingual evaluation')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to run the model on')
    parser.add_argument('--mono-only', action='store_true', help='Run only monolingual evaluations')
    parser.add_argument('--multi-only', action='store_true', help='Run only multilingual evaluations')
    args = parser.parse_args()

    if args.mono_only and args.multi_only:
        parser.error("Cannot use --mono-only and --multi-only together")

    if args.mono_only:
        run_monolingual_only(args.model, args.nickname, args.device, args.source)
    elif args.multi_only:
        run_multilingual_only(args.model, args.nickname, args.main_lang, args.device, args.source)
    else:
        all_lang_evaluation(args.model, args.nickname, args.main_lang, args.device, args.source)


if __name__ == "__main__":
    main()
