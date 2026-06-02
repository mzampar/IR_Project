
# **********************************************************
# tf_idf.py
#
# This script gets in input a corpus and computes the tf-idf matrix.
# In input are also passed:
# - n_terms: maximum number of terms to keep (keeps the most frequent terms)
#            if set to -1, this option is not used
# - ngrams: if 1 use unigrams, if 2 unigrams and bigrams, if 3 uni bi and trigrams, ...
# - min_df: skip terms that appear in less than min_df documents 
# - max_df: skip temrms that appear in more that the fraction (in [0,1]) specified by max_df
#
# It assumes a beir dataset is passed.
# It computes the tf-idf matrix, builds a vocabulary with the extracted terms,
# adds the term 'covid19' to the vocabulary and recomputes the tf-idf.
# This is because using a low max_df would remove the term covid19, but, since
# in the corpus there are many non relevant documents without this term, 
# this fact would harm performance.
#
# **********************************************************


# --- MODULES ---

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from beir.datasets.data_loader import GenericDataLoader
import sys
from scipy.sparse import save_npz
import json
import numpy as np


# --- INPUT ARGS ---

data_path, n_terms, ngrams, min_df, max_df, out_matrix, out_vocabulary, out_doc_ids, out_tf_idf = sys.argv[1:]

n_terms=int(n_terms)
ngrams = int(ngrams)
min_df = int(min_df)
max_df = float(max_df)

if n_terms == -1 :
   n_terms = None

# --- MAIN --- 

corpus, queries, qrels = GenericDataLoader(
    data_folder=data_path
).load(split="test")

# Extract the text from the corpus
corpus_texts = [doc["text"] for doc in corpus.values()]

# Learn vocabulary 
base_vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=n_terms,
    min_df=min_df,
    max_df=max_df,
    ngram_range=(1, ngrams),
)

base_vectorizer.fit(corpus_texts)

# Extract learned vocabulary
vocab = set(base_vectorizer.get_feature_names_out())

# Add covid19 to the vocabulary
forced_terms = {"covid19"}
vocab.update(forced_terms)

# Fit again with new vocabulary 
tfidf_vectorizer = TfidfVectorizer(
    vocabulary=sorted(vocab),
    ngram_range=(1, ngrams),
)

tfidf_matrix = tfidf_vectorizer.fit_transform(corpus_texts)

print(tfidf_matrix.shape)

# Save the sparse TF-IDF matrix
save_npz(
    out_matrix,
    tfidf_matrix.T
)

# Save the document IDs
with open(
    out_doc_ids,
    "w"
) as f:
    json.dump(list(corpus.keys()), f)

# Save vocabulary
with open(out_vocabulary, "w") as f:
    json.dump(tfidf_vectorizer.vocabulary_, f)

# Save IDF values
np.save(out_tf_idf, tfidf_vectorizer.idf_)
