# **********************************************************
# svd.py
#
# This script gets in input a tf-idf matrix, the desired number of 
# components (singular values) to keep in the SVD decomposition
# of the tf-idf matrix, performs the truncated SVD and saves the
# U, Sigma and V matrices.
#
# **********************************************************

# --- MODULES ---

from scipy.sparse import load_npz
from scipy.linalg import norm
import sys
import os
import json
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

# --- INPUT ARGS ---

tf_idf_matrix_path, n_components, out_dir, outfig = sys.argv[1:]

n_components = int(n_components)

# --- MAIN ---

# Load the sparse TF-IDF matrix
tfidf_matrix = load_npz(tf_idf_matrix_path)
print(tfidf_matrix.shape)

# Initialize TruncatedSVD
svd = TruncatedSVD(n_components=n_components, random_state=0)

# U * Sigma
svd_matrix = svd.fit_transform(tfidf_matrix)
print(svd_matrix.shape)

singular_values = svd.singular_values_
print(singular_values.shape)

# V^T
print(svd.components_.shape)

# To reconstruct the approximate matrix: U * S * V^T
# approximate_matrix = svd_matrix @ svd.components_

os.makedirs(out_dir, exist_ok=True)

# Reduce precision to reduce disk usage
svd_matrix16 = svd_matrix.astype(np.float16)
singular_values = singular_values.astype(np.float16)
svd_components = svd.components_.astype(np.float16)

# Save the reduced matrix
np.save(
    f"{out_dir}/trec-covid-svd_matrix-{n_components}.npy",
    svd_matrix
)

# Save the singular values
np.save(
    f"{out_dir}/trec-covid-singular_values-{n_components}.npy",
    singular_values
)

# Save the components (V^T)
np.save(
    f"{out_dir}/trec-covid-svd_components-{n_components}.npy",
    svd_components
)

# --------

try:
    print('plot singular values')
    # show first 500 singular values
    svd = TruncatedSVD(n_components=500, random_state=0)
    svd.fit(tfidf_matrix)
    
    s = svd.singular_values_  
    
    plt.figure()
    plt.plot(range(1, len(s) + 1), s)
    plt.yscale('log')
    plt.xlabel('Rank')
    plt.ylabel('Singular value (log scale)')
    plt.title('Singular value spectrum (top-k)')
    plt.savefig(outfig)
    print(f'Plot saved to {outfig}')
    
except Exception as e:
    print(e)
