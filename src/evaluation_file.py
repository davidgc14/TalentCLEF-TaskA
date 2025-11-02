import argparse
import pandas as pd
from ranx import Qrels, Run, evaluate

def load_qrels(qrels_path):
    """
    Loads the qrels file (TREC format: q_id, iter, doc_id, rel)
    and converts it to a Qrels object.
    """
    qrels_df = pd.read_csv(qrels_path, sep="\t", header=None,
                           names=["q_id", "iter", "doc_id", "rel"],
                           dtype={"q_id": str, "doc_id": str, "rel":int})

    return Qrels.from_df(qrels_df, q_id_col="q_id", doc_id_col="doc_id", score_col="rel")

def load_run(run_path):
    """
    Loads the run file (TREC format: q_id, Q0, doc_id, rank, score, [tag])
    and converts it to a Run object.
    """
    run_df = pd.read_csv(run_path, sep=r"\s+", header=None)
    
    # Assign column names based on the number of columns
    if run_df.shape[1] == 5:
        run_df.columns = ["q_id", "Q0", "doc_id", "rank", "score"]
    elif run_df.shape[1] >= 6:
        run_df.columns = ["q_id", "Q0", "doc_id", "rank", "score", "tag"]
    else:
        raise ValueError("The run file does not have the expected format.")

    run_df["q_id"] = run_df.q_id.astype(str)
    run_df["doc_id"] = run_df.doc_id.astype(str)
    return Run.from_df(run_df, q_id_col="q_id", doc_id_col="doc_id", score_col="score")

def main():
    parser = argparse.ArgumentParser(
        description="Simplified evaluation script for TalentCLEF2025. "
                    "Requires qrels, run, query_lang, and corpuselements_lang as input."
    )
    parser.add_argument("--qrels", required=True, help="Path to the qrels file (TREC format)")
    parser.add_argument("--run", required=True, help="Path to the run file (TREC format)")
    args = parser.parse_args()

    # Display received parameters
    print("Received parameters:")
    print(f"  qrels: {args.qrels}")
    print(f"  run: {args.run}")

    print("Loading qrels...")
    qrels = load_qrels(args.qrels)
    print("Loading run...")
    run = load_run(args.run)

    # Define the evaluation metrics
    metrics = ["map", "mrr", "ndcg", "precision@5", "precision@10", "precision@100"]

    print("Running evaluation...")
    results = evaluate(qrels, run, metrics)

    print("\n=== Evaluation Results ===")
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")

if __name__ == "__main__":
    main()
