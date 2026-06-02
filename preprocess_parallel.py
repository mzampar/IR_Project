
# **********************************************************
# preprocess_parallel.py
#
# This script preprocesses the corpus, assuming it is a
# beir dataset.
# A simple parallelisation is performed with the multiprocessing module.
# The preprocessing is performed by the function preprocess_text
# defined in utils.py
#
# **********************************************************


# --- MODULES ---

import sys
import os
import json
from beir.datasets.data_loader import GenericDataLoader

import multiprocessing as mp
from tqdm import tqdm

from utils import *

import sys

# --- FUNCTIONS ---

# Function to process a single document
def process_document(doc_id, corpus):
    return {
        doc_id: {
            "text": preprocess_text(corpus[doc_id]["text"]),
            "title": corpus[doc_id].get("title", "")
        }
    }

corpus = None

def init_worker(shared_corpus):
    global corpus
    corpus = shared_corpus

def worker(doc_id):
    return process_document(doc_id, corpus)

# --- INPUT ARGS ---

data_path, out_dir = sys.argv[1:]

# --- MAIN ---
def main():

    # (Some tests were performed but at the end these features were not used)
    # Whether to remove the documents not referring to covid19
    remove_non_covid19 = False
    # Wheter to consider the word covid19 a stopword
    remove_covid_str = False
    
    # Load the dataset
    corpus, queries, qrels = GenericDataLoader(
        data_folder=data_path
    ).load(split="test")
    
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    
    print("SLURM_CPUS_PER_TASK:", n_jobs)
    
    # Parallelize the preprocessing of the corpus
    with mp.Pool(n_jobs, initializer=init_worker, initargs=(corpus,)) as pool:
        corpus_processed_list = list(tqdm(pool.imap(worker, corpus), total=len(corpus)))
   
    # Merge the results into a single dictionary
    corpus_processed = {}
    for item in corpus_processed_list:
        corpus_processed.update(item)
    
    empty_doc_ids = []
    non_cov_doc_ids = []
        
    # Write the processed docs to file
    with open(os.path.join(out_dir, "corpus.jsonl"), "w") as f:
        for doc_id, doc in corpus_processed.items():
            text = doc.get("text", "").strip()

            if remove_non_covid19 and "covid19" not in text.lower():
                non_cov_doc_ids.append(doc_id)
                continue

            # Noticed empty docs, discarded
            # It was not a problem of processing the documents, the original corpus contains empty docs
            if not text:
                empty_doc_ids.append(doc_id)
                continue

            if remove_covid_str : 
                text = re.sub(r'\bcovid19\b', '', text)
                text = re.sub(r'\s+', ' ', text).strip()

            f.write(json.dumps({
                "_id": doc_id,
                "text": text,
                "title": doc.get("title", "")
            }) + "\n")

    # save empty doc ids
    with open(os.path.join(out_dir, "empty_docs.txt"), "w") as f:
        for doc_id in empty_doc_ids:
            f.write(f"{doc_id}\n")

    if remove_non_covid19:
        # save non cov doc ids
        with open(os.path.join(out_dir, "non_cov_docs.txt"), "w") as f:
            for doc_id in non_cov_doc_ids:
                f.write(f"{doc_id}\n")

if __name__ == "__main__":
    main()

