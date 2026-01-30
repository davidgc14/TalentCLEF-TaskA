import pandas as pd
import numpy as np
from ranx import Run, fuse
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from pathlib import Path
from evaluation_file import evaluate_run
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
# DATA LOADING
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


# ========================
# MODEL INFERENCE
# ========================

def generate_predictions(model, queries_texts, corpus_texts, queries_ids, corpus_ids, device, model_name):
    """
    Generate predictions as a dictionary mapping query_id -> {doc_id: score}.
    
    Args:
        model: SentenceTransformer model
        queries_texts: List of query texts
        corpus_texts: List of corpus texts
        queries_ids: List of query IDs
        corpus_ids: List of corpus document IDs
        device: Device to run inference on
        model_name: Name of the model for logging
    
    Returns:
        Dictionary with structure {query_id: {doc_id: score, ...}, ...}
    """
    print(f'\n[{model_name}] Encoding queries and corpus...')
    query_embeddings = model.encode(queries_texts, convert_to_tensor=True, 
                                    show_progress_bar=True, device=device)
    corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True, 
                                     show_progress_bar=True, device=device)
    
    print(f'[{model_name}] Calculating similarities...')
    similarities = util.cos_sim(query_embeddings, corpus_embeddings).cpu().numpy()
    
    # Build dictionary
    predictions_dict = {}
    for q_idx, q_id in enumerate(queries_ids):
        predictions_dict[str(q_id)] = {
            str(corpus_ids[c_idx]): float(similarities[q_idx, c_idx])
            for c_idx in range(len(corpus_ids))
        }
    
    return predictions_dict


# ========================
# FUSION Y RERANKING
# ========================

def logits_normalization(head, min_value, max_value=1):
    values = [v for _, v in head]

    v_min = min(values)
    v_max = max(values)

    normalized = [
        (key, min_value + (max_value - min_value) * (value - v_min) / (v_max - v_min))
        for key, value in head
    ]

    return normalized


def apply_reranking_top_k(model, k, run_dict, queries_dict, corpus_dict):
    """
    Apply reranking to top-k documents for each query.
    
    Args:
        model: CrossEncoder model
        k: Number of top documents to rerank
        run_dict: Dictionary {query_id: {doc_id: score}}
        queries_dict: Dictionary {query_id: query_text}
        corpus_dict: Dictionary {doc_id: doc_text}
    
    Returns:
        Reranked dictionary with same structure
    """
    print(f'\nApplying reranking to top-{k} documents...')
    reranked_run = {}

    for q_id, docs_scores in run_dict.items():
        if q_id not in queries_dict:
            continue

        query_text = queries_dict[q_id]

        sorted_items = sorted(docs_scores.items(), key=lambda x: x[1], reverse=True)

        head_items = sorted_items[:k]
        tail_items = sorted_items[k:]

        if not head_items:
            reranked_run[q_id] = docs_scores
            continue

        min_similarity = head_items[-1][1]

        pairs_to_predict = []
        valid_head_ids = []

        for doc_id, original_score in head_items:
            if doc_id in corpus_dict:
                doc_text = corpus_dict[doc_id]
                pairs_to_predict.append([query_text, doc_text])
                valid_head_ids.append(doc_id)

        if pairs_to_predict:
            new_scores = model.predict(
                pairs_to_predict,
                convert_to_numpy=True
            )

            if len(new_scores.shape) > 1:
                new_scores = new_scores.flatten()

            new_head_items = []
            for i, doc_id in enumerate(valid_head_ids):
                new_head_items.append((doc_id, float(new_scores[i])))

            new_head_items.sort(key=lambda x: x[1], reverse=True)
            new_head_items = logits_normalization(new_head_items, min_similarity)

            final_list = new_head_items + tail_items
            reranked_run[q_id] = dict(final_list)
        else:
            reranked_run[q_id] = dict(sorted_items)

    return reranked_run


def fusion_ranking(predictions_list):
    """Combine multiple prediction dictionaries using weighted fusion."""
    weights = [.35, .38, .27]

    runs = [Run.from_dict(preds) for preds in predictions_list]
    fused_run = fuse(
        runs=runs,
        method="wsum",
        params={"weights": weights},
        norm="min-max"
    )

    return fused_run.to_dict()


def reranking(fused_dict, queries_dict, corpus_dict, device, top_k=5, 
              rr_model_name='Jsevisal/CrossEncoder-ModernBERT-base-qnli'):
    """
    Apply cross-encoder reranking to fused results.
    
    Args:
        fused_dict: Dictionary from fusion step
        queries_dict: Dictionary {query_id: query_text}
        corpus_dict: Dictionary {doc_id: doc_text}
        device: Device to run the model on
        top_k: Number of top documents to rerank per query
        rr_model_name: Name/path of the cross-encoder model
    
    Returns:
        Reranked dictionary
    """
    print(f'\nLoading cross-encoder model: {rr_model_name}')
    model = CrossEncoder(
        rr_model_name,
        device=device,
        trust_remote_code=True,
        cache_dir=project_dir / 'models' / rr_model_name.split('/')[-1],
    )

    reranked_dict = apply_reranking_top_k(model, top_k, fused_dict, queries_dict, corpus_dict)

    return reranked_dict

# ========================
# TREC FORMAT CONVERSION
# ========================

def dict_to_trec_format(predictions_dict, run_name):
    """
    Convert predictions dictionary to TREC format (strict format).
    
    Args:
        predictions_dict: Dictionary with structure {query_id: {doc_id: score, ...}, ...}
        run_name: Name/tag for the run
    
    Returns:
        List of strings in TREC format
    """
    print('\nConverting to TREC format...')
    results = []
    
    for q_id, doc_scores in predictions_dict.items():
        # Ensure q_id is string and clean
        q_id_str = str(q_id).strip()
        
        # Sort documents by score (descending)
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (doc_id, score) in enumerate(sorted_docs, start=1):
            # Ensure doc_id is string and clean
            doc_id_str = str(doc_id).strip()
            
            # Ensure score is float
            score_float = float(score)
            
            # TREC format: q_id Q0 doc_id rank score tag
            # Sin espacios extra, con formato estricto
            results.append(f"{q_id_str} Q0 {doc_id_str} {rank} {score_float:.4f} {run_name}")
    
    return results


def write_run_file(results, run_file):
    """Write results to TREC format file."""
    print('Writing results to:', run_file)
    with open(run_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results))


# ========================
# MAIN PIPELINE
# ========================

def main():
    start_time = time.time()

    import argparse
    parser = argparse.ArgumentParser(description='Evaluate ensemble of models on TalentCLEF task')
    parser.add_argument('--model1', type=str, default='models/finetuned models/bilingual-embedding-large',
                        help='Path to first model')
    parser.add_argument('--model2', type=str, default='models/finetuned models/multilingual-e5-large-instruct',
                        help='Path to second model')
    parser.add_argument('--model3', type=str, default='models/finetuned models/gte-multilingual-base',
                        help='Path to third model')
    parser.add_argument('--source', type=str, default='validation', choices=['validation', 'test'],
                        help='Dataset source to use')
    parser.add_argument('--lang', type=str, default='es', choices=['en', 'es', 'de', 'zh'],
                        help='Language to evaluate')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to run the models on')
    parser.add_argument('--run-name', type=str, default='ensemble',
                        help='Name for the ensemble run')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of top documents to rerank')
    parser.add_argument('--skip-reranking', action='store_true',
                        help='Skip reranking step')
    args = parser.parse_args()

    model_paths = [args.model1, args.model2, args.model3]
    
    # Load data
    data_dir = project_dir / 'data' / args.source / lang_dict[args.lang]
    print(f'Loading data from {data_dir}...')
    queries_ids, queries_texts, corpus_ids, corpus_texts = load_data(data_dir)
    
    # Create dictionaries for reranking
    queries_dict = {str(q_id): q_text for q_id, q_text in zip(queries_ids, queries_texts)}
    corpus_dict = {str(c_id): c_text for c_id, c_text in zip(corpus_ids, corpus_texts)}
    
    # Generate predictions for each model
    predictions_list = []
    
    for i, model_path in enumerate(model_paths, 1):
        print(f'\n{"="*60}')
        print(f'Processing Model {i}/{len(model_paths)}: {model_path}')
        print(f'{"="*60}')
        
        # Load model
        print(f'Loading model from {model_path}...')
        model = SentenceTransformer(model_path, device=args.device, cache_folder=model_path, trust_remote_code=True)
        model_name = Path(model_path).stem
        
        # Generate predictions
        predictions_dict = generate_predictions(
            model, queries_texts, corpus_texts, 
            queries_ids, corpus_ids, args.device, model_name
        )
        
        predictions_list.append(predictions_dict)
        
        # Free memory
        del model
    
    print(f'\n{"="*60}')
    print(f'All {len(predictions_list)} models processed')
    print(f'{"="*60}')
    
    # Fusion
    print('\nApplying fusion...')
    fused_dict = fusion_ranking(predictions_list)
    
    # Reranking (optional)
    if args.skip_reranking:
        print('\nSkipping reranking step (--skip-reranking flag set)')
        final_dict = fused_dict
    else:
        print('\nApplying Reranking...')
        final_dict = reranking(fused_dict, queries_dict, corpus_dict, 
                              args.device, top_k=args.top_k)
    
    # Convert to TREC format
    trec_results = dict_to_trec_format(final_dict, args.run_name)
    
    # Write run file
    run_file = output_dir / f"run_{args.lang}-{args.lang}.trec"
    write_run_file(trec_results, run_file)
    
    # Evaluate if source is validation
    if args.source == 'validation':
        qrels_path = data_dir / "qrels.tsv"
        print(f'\nEvaluating against qrels: {qrels_path}')
        evaluation_results = evaluate_run(qrels_path, run_file)
        
        print("\n=== Evaluation Results ===")
        for metric, score in evaluation_results.items():
            print(f"{metric}: {score:.4f}")
        
        # Save evaluation results
        results_file = output_dir / f"results_{args.lang}_{args.run_name}.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump({
                'models': model_paths,
                'run_name': args.run_name,
                'lang': args.lang,
                'source': args.source,
                'reranking': not args.skip_reranking,
                'top_k': args.top_k if not args.skip_reranking else None,
                'timestamp': today,
                'metrics': evaluation_results
            }, f, indent=2)
        print(f'\nResults saved to {results_file}')
    else:
        print('\nSource is "test" - skipping evaluation (no qrels available)')
    
    print(f'\nRun file saved to: {run_file}')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'\nTotal execution time: {elapsed_time/60:.2f} minutes')


if __name__ == "__main__":
    main()