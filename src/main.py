import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import subprocess
import time
import os


def monolingual_evaluation(lang, source='validation', model_name="all-MiniLM-L6-v2", nickname="default", device='cpu'):

    today = time.strftime("%Y-%m-%d")
    runtime = time.strftime("%Y%m%d_%H%M%S")

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_dir, 'data', source, lang)
    output_dir = os.path.join(project_dir, 'src', 'output', today)

    os.makedirs(output_dir, exist_ok=True)

    queries_path = os.path.join(data_dir, "queries")
    corpus_elements_path = os.path.join(data_dir, "corpus_elements")
    qrels_path = os.path.join(data_dir, "qrels.tsv")


    print('Loading data for language:', lang)

    queries = pd.read_csv(queries_path, sep="\t")
    corpus_elements = pd.read_csv(corpus_elements_path, sep="\t")

    queries_ids = queries.q_id.to_list()
    queries_texts = queries.jobtitle.to_list()

    # map_queries = dict(zip(queries_ids,queries_texts))

    corpus_ids = corpus_elements.c_id.to_list()
    corpus_texts = corpus_elements.jobtitle.to_list()
    # map_corpus = dict(zip(corpus_ids,corpus_texts))

    print('Loading model:', model_name, 'on device:', device)

    model = SentenceTransformer(model_name, device=device)
    query_embeddings = model.encode(queries_texts, convert_to_tensor=True, show_progress_bar=True)
    corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True, show_progress_bar=True)


    print('Calculating similarities and preparing results...')

    similarities = util.cos_sim(query_embeddings, corpus_embeddings).cpu().numpy()

    results = []
    
    for q_idx, q_id in enumerate(queries_ids):
        sorted_indices = np.argsort(-similarities[q_idx])  # Decrease order
        for rank, c_idx in enumerate(sorted_indices):  
            doc_id = corpus_ids[c_idx]
            score = similarities[q_idx, c_idx]
            results.append(f"{str(q_id)} Q0 {str(doc_id)} {rank+1} {score:.4f} {model_name}")


    run_file = os.path.join(output_dir, f"evaluation_{lang}_{nickname}_{runtime}.trec")

    print('Writing results to:', os.path.basename(run_file))

    with open(run_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results))


    print('Evaluating with evaluation_file.py...')

    command = ["python", "evaluation_file.py", "--qrels", qrels_path, "--run", run_file]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.stderr:
        print("Error:", result.stderr)
    print(result.stdout)


    print('Saving evaluation results...')

    with open(os.path.join(output_dir, f"results_{lang}_{runtime}.txt"), "w", encoding="utf-8") as f:
        f.write(f"Model name: {model_name}\n\n")
        f.write(f"Model alias: {nickname}\n\n")
        f.write("\n".join(result.stdout.splitlines()[-7:]))


    print('Evaluation completed for language:', lang)
    return result.stdout


if __name__ == "__main__":
    monolingual_evaluation(lang='english')