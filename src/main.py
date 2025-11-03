import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import subprocess
import time
import os

source = "validation"

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



def monolingual_evaluation(lang, model, device):


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



def multilingual_evaluation(lang, main_lang, model, device):

    data_dir_main = os.path.join(project_dir, 'data', source, lang_dict[main_lang])
    data_dir_target = os.path.join(project_dir, 'data', source, lang_dict[lang])


    queries_path = os.path.join(data_dir_main, "queries")
    corpus_elements_path = os.path.join(data_dir_target, "corpus_elements")
    # qrels_path = os.path.join(data_dir_main, "qrels.tsv")


    print('Loading data for multilingual evaluation:', main_lang, '->', lang)

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


    run_file = os.path.join(output_dir, f"run_{main_lang}-{lang}_{model_name}.trec")

    print('Writing results to:', os.path.basename(run_file))

    with open(run_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results))


    print('Still not able to evaluate multilingual runs. Skipping evaluation step.')
    return "Multilingual evaluation not implemented yet."











def all_lang_evaluation(model_name="paraphrase-multilingual-MiniLM-L12-v2", nickname="default", main_lang='en', device='cpu'):

    model = SentenceTransformer(model_name, device=device)

    monolingual_results = {}
    for lang in lang_dict.keys():
        print(f"Starting evaluation for language: {lang}")
        monolingual_results[lang] = monolingual_evaluation(lang=lang, model=model, device=device)

    print("Monolingual evaluations completed. Saving results...")

    with open(os.path.join(output_dir, f"results.txt"), "w", encoding="utf-8") as f:
        f.write("=== Evaluation Results for monolingual ===\n")
        f.write(f"Model name: {model_name}\n")
        f.write(f"Model alias: {nickname}\n\n")
        for lang in monolingual_results:
            f.write(f"=== Language: {lang} ===\n")
            f.write("\n".join(monolingual_results[lang].splitlines()[-6:]))
            f.write("\n\n\n")

    multilingual_results = {}
    for lang in lang_dict.keys():
        if lang == main_lang:
            continue
        print(f"Starting multilingual evaluation for language pair: {main_lang} -> {lang}")
        multilingual_results[f"{main_lang}-{lang}"] = multilingual_evaluation(lang=lang, main_lang=main_lang, model=model, device=device)

    print("Multilingual evaluations completed. Saving results...")

    with open(os.path.join(output_dir, f"results.txt"), "a", encoding="utf-8") as f:
        f.write("=== Evaluation Results for multilingual ===\n")
        f.write(f"Model name: {model_name}\n")
        f.write(f"Model alias: {nickname}\n\n")
        for lang_pair in multilingual_results:
            f.write(f"=== Language Pair: {lang_pair} ===\n")
            f.write(multilingual_results[lang_pair])
            f.write("\n\n\n")



def main():
    all_lang_evaluation()





if __name__ == "__main__":
    main()