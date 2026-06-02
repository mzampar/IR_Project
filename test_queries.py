
# **********************************************************
# test_queries.py
#
# This script computes the ndcg
# (normalised discounted cumulative gain) 
# (the function to compute it is defined in utils.py)
# on the test queries of the dataset
# It produces 2 csv's: 1 with the performance for each query 
# and one with the mean performance
#
# **********************************************************

# --- MODULES ---

import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import preprocess_text, compute_ndcg
import sys

# --- INPUT ARGS ---

qrels_, queries_, doc_titles, svd_dir, min_rel_docs, n_components, doc_ids_, tf_idf_vocab, tf_idf_values, out_csv = sys.argv[1:]

if min_rel_docs.lower() == 'none':
    min_rel_docs = None
else:
    min_rel_docs = int(min_rel_docs)

# --- MAIN ---

if doc_titles.lower() == 'none':
    titles_df = None
else:
    titles_df = pd.read_csv(doc_titles, sep=';')


# Read the csv with query-id corpus-id and relevance score
qrels = pd.read_csv(qrels_, sep='\t')

# Read the queries and parse them in a pandas Df
rows = []
with open(queries_, "r") as f:
    for line in f:
        obj = json.loads(line)

        rows.append({
            "id": obj["_id"],
            "text": obj["text"],
            "query": obj["metadata"]["query"],
            "narrative": obj["metadata"]["narrative"]
        })

query_df = pd.DataFrame(rows)

# Load the document IDs
with open(
    doc_ids_,
    "r"
) as f:
    doc_ids = json.load(f)

# Load the SVD matrix
svd_matrix = np.load(f"{svd_dir}/trec-covid-svd_matrix-{n_components}.npy")

# Load the singular values (optional, not needed for similarity)
singular_values = np.load(f"{svd_dir}/trec-covid-singular_values-{n_components}.npy")

# Reconstract the sigma matrix
sigma = np.diag(singular_values)
sigma_inv = np.diag(1.0 / singular_values)

# Re-normalise the svd-matrix to get U
U = svd_matrix @ sigma_inv   
n_terms=svd_matrix.shape[0]

# Load the SVD components (V^T)
V_t = np.load(f"{svd_dir}/trec-covid-svd_components-{n_components}.npy")

# Load vocabulary
with open(tf_idf_vocab) as f:
    vocab = json.load(f)
# Recreate vectorizer
vectorizer = TfidfVectorizer(vocabulary=vocab)
# Initialize internal state
vectorizer.fit(["covid19"])
# Load IDF
idf = np.load(tf_idf_values)
# Assign IDF 
vectorizer.idf_ = idf

results_ = []
k_values = [1,10,20,30,40,50,60,70,80,90,100]

# Test on each query
for _, row in query_df.iterrows():
    
    qrels_ = qrels[qrels['query-id'].astype(int) == int(row["id"])]
    n_rel_docs1 = len(qrels_[qrels_['score']==1]) # partially relevant
    n_rel_docs2 = len(qrels_[qrels_['score']==2]) # relevant

    # Use relevant and partially relevant doc-ids to compute performance
    qrels_ = qrels_[qrels_['score']>0]
    n_rel_docs = len(qrels_)

    # Skip queries with less than min_rel_docs relevant documents
    if n_rel_docs2 < min_rel_docs: 
        continue

    rel_ids = list(qrels_['corpus-id'])
    rel_scores = list(qrels_['score'])  

    query_id = row["id"]
    query_text = row["text"]
    query = row["query"]
    query_n = row["narrative"]

    print(f"Query ID: {query_id}\n")
    print(f"Query Text: {query_text}\n")
    print(f"Query Narrative: {query_n}\n")
    print(f"Query: {query}\n")

    #text = query + ' ' + query_n
    #text = query + query_text
    text = query_text + ' ' + query + ' ' + query_n
    query_vec = vectorizer.transform([preprocess_text(text,remove_covid=False)])

    # Map query to latent concept space
    query_svd = sigma_inv @ svd_matrix.T @ query_vec.T

    # Compute cosine similarity with all documents
    similarities = cosine_similarity(query_svd.T, V_t.T).flatten()

    # Compute for each k in k_values
    ndcg_results = []  
    
    for k in k_values:
        top_doc_indices = np.argsort(similarities)[-k:][::-1]
        retrieved_doc_ids = [doc_ids[i] for i in top_doc_indices]
    
        # nDCG
        ndcg, dcg, idcg = compute_ndcg(
            retrieved_doc_ids, rel_ids, rel_scores, n=k
        )
        ndcg_results.append(ndcg)    
    
    results_.append({
        "query_id": query_id,
        **{f"ndcg_at_{k}": f"{ndcg:.2f}" for k, ndcg in zip(k_values, ndcg_results)},
        "n_rel_docs": n_rel_docs,
        "n_rel_docs1": n_rel_docs1,
        "n_rel_docs2": n_rel_docs2,
        "query_text": query_text,
    })
   
    # Get the ID of the closest document
    closest_doc_idx = top_doc_indices[0]
    closest_doc_id = doc_ids[closest_doc_idx]
    
    print(f"Closest document ID: {closest_doc_id}\n")
    if titles_df is not None:
        title = titles_df[titles_df['corpus-id']==closest_doc_id]['title'].values
        print(title)
    
    # Check if the closest document is in qrels for this query
    is_relevant = (closest_doc_id in rel_ids)
    print(f"\nIs the closest document relevant? {is_relevant}\n")

    print(' ----- ')

# Convert the list of dictionaries to a pandas DataFrame
df_results = pd.DataFrame(results_)
df_results.to_csv(out_csv, index=False, sep='\t')
    
# List of ndcg columns
precision_cols = [col for col in df_results.columns if col.startswith("ndcg_at_")]

# Initialize a dictionary to store summary statistics
summary_stats = {}

# Compute statistics for each ndcg column
for col in precision_cols:
    values = df_results[col].astype(float).values
    summary_stats[col] = {
        "mean": np.mean(values),
        "std": np.std(values),
        "median": np.median(values),
        "min": np.min(values),
        "p5": np.percentile(values,5),
        "p95": np.percentile(values,95),
        "max": np.max(values),
    }

# Convert to DataFrame
summary_df = pd.DataFrame(summary_stats).T
summary_df = summary_df.reset_index()  
summary_df = summary_df.rename(columns={'index': 'metric'})

# Round numeric columns
numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
summary_df[numeric_cols] = np.round(summary_df[numeric_cols], 2)
# Add information about number of singular components and terms in the vocabulary
summary_df['n_components'] = n_components
summary_df['n_terms'] = n_terms
# Save csv
summary_df.to_csv(out_csv.replace('.tsv','-summary.csv'),index=False)

