import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import subprocess
import time
import os


lang_dict = {
    'en': 'english',
    'es': 'spanish',
    'de': 'german',
    'zh': 'chinese',
}

today = time.strftime("%Y-%m-%d")

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
date_dir = os.path.join(project_dir, 'src', 'output', today)
os.makedirs(date_dir, exist_ok=True)

# comprobar si la ultima ejecucion esta vacia

exec_dirs = sorted([d for d in os.listdir(date_dir) if d.isdigit() and len(d) == 3])

if exec_dirs and not os.listdir(os.path.join(date_dir, exec_dirs[-1])):
    output_dir = os.path.join(date_dir, exec_dirs[-1])
else:
    next_id = int(exec_dirs[-1]) + 1 if exec_dirs else 1
    output_dir = os.path.join(date_dir, f"{next_id:03d}")
    os.makedirs(output_dir, exist_ok=True)



def monolingual_evaluation(lang, model, source='validation', nickname="default", device='cpu'):


    data_dir = os.path.join(project_dir, 'data', source, lang_dict[lang])


    queries_path = os.path.join(data_dir, "queries")
    corpus_elements_path = os.path.join(data_dir, "corpus_elements")
    qrels_path = os.path.join(data_dir, "qrels.tsv")


    print('Loading data for language:', lang)

    queries = pd.read_csv(queries_path, sep="\t")
    corpus_elements = pd.read_csv(corpus_elements_path, sep="\t")

    queries_ids = queries.q_id.to_list()
    queries_texts = queries.jobtitle.to_list()


    corpus_ids = corpus_elements.c_id.to_list()
    corpus_texts = corpus_elements.jobtitle.to_list()


    model_name = model[0].auto_model.config._name_or_path 
    model_name = model_name.split("/")[-1]
    print('Encoding data:', model_name, 'on device:', device)

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


    run_file = os.path.join(output_dir, f"run_{lang}-{lang}_{model_name}.trec")

    print('Writing results to:', os.path.basename(run_file))

    with open(run_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results))


    print('Evaluating with evaluation_file.py...')

    command = ["python", os.path.join(project_dir, "src", "evaluation_file.py"), "--qrels", qrels_path, "--run", run_file]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.stderr:
        print("Error:", result.stderr)
    print(result.stdout)



    print('Evaluation completed for language:', lang)
    return result.stdout





def all_lang_evaluation(model_name="all-MiniLM-L6-v2", nickname="default", device='cpu'):

    model = SentenceTransformer(model_name, device=device)

    results = {}
    for lang in lang_dict.keys():
        print(f"Starting evaluation for language: {lang}")
        results[lang] = monolingual_evaluation(lang=lang, model=model, nickname=nickname, device=device)

    print("All evaluations completed. Saving all evaluation results...")

    with open(os.path.join(output_dir, f"results.txt"), "w", encoding="utf-8") as f:
        f.write("=== Evaluation Results for multilingual ===\n")
        f.write(f"Model name: {model_name}\n\n")
        f.write(f"Model alias: {nickname}\n\n")
        for lang in results:
            f.write(f"=== Results for language: {lang} ===\n")
            f.write(results[lang])
            f.write("\n\n")




def main():
    # Example usage
    all_lang_evaluation(model_name="all-MiniLM-L6-v2", nickname="MiniLM_v2", device='cpu')





if __name__ == "__main__":
    main()