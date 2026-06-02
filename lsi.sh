#!/bin/bash
#SBATCH --job-name=lsi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=THIN
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=logs/lsi_%j.out

# **********************************************************
# lsi.sh
#
# This script runs the entire Latent Semantic Indexing 
# construction and assessment.
#
# Steps performed:
# - download of the data
# - preprocessing of the corpus
# - tf-idf matrix computation
# - SVD of the tf-idf matrix
# - evaluation of the system on test queries
#
# Run this script with:
#
# sbatch lsi.sh "$(pwd)/lsi.cfg"
# or
# bash lsi.sh "$(pwd)/lsi.cfg"
# 
# **********************************************************

set -e

# --- MODULES ---

eval "$(/u/dssc/mzampar/miniconda3/bin/conda shell.bash hook)"
conda activate base

# --- CONFIG ---

config=$1
source $config

echo "n_terms $n_terms"
echo "n_grams $n_grams"
echo "n_components $n_components"
echo "min_df $min_df"
echo "max_df $max_df"

# --- MAIN ---

mkdir -p $(dirname $sv_plot)
mkdir -p $csv_dir

# Download data
if [ ! -d "$data_path" ]; then
    echo "$data_path does not exist. Downloading data..."
    python download.py "$data_link" "$data_path_"
else
    echo "$data_path already exists. Skipping download."
fi

# Preprocess data
if [ ! -d "$data_path_processed" ]; then
    mkdir -p "$data_path_processed"
    echo "$data_path_processed does not exist. Submitting preprocessing job..."
    JOB_ID=$(sbatch --parsable preprocess_parallel.sh "$config")

    # Wait for the specific job to finish
    echo "Waiting for job $JOB_ID to finish..."
    while squeue -j "$JOB_ID" | grep -q "$JOB_ID"; do
        sleep 10
    done
    echo "Job $JOB_ID has finished."
    cp -r $data_path/qrels $data_path_processed
    cp -r $data_path/queries.jsonl $data_path_processed
else
    echo "$data_path_processed already exists. Skipping preprocessing."
fi

# Compute TF-IDF matrix
if [ ! -f "$tf_out_matrix" ]; then
    echo "$tf_out_matrix does not exist. Running TF-IDF computation..."
    mkdir -p $(dirname $tf_out_matrix )
    python tf_idf.py "$data_path_processed" "$n_terms" "$n_grams" $min_df $max_df "$tf_out_matrix" "$tf_out_vocabulary" "$tf_out_doc_ids" "$tfidf_values"

else
    echo "$tf_out_matrix already exists. Skipping TF-IDF computation."
fi

# Check if svd_out_dir does not exist
if [ ! -d "$svd_out_dir" ]; then
    echo "Folder $svd_out_dir does not exist. Running SVD decomposition..."
    mkdir -p "$svd_out_dir"
    python svd.py "$tf_out_matrix" "$n_components" "$svd_out_dir" "$sv_plot" 

    if [ -z "$(ls -A "$svd_out_dir")" ]; then
        echo "ERROR: output directory is empty"
        rm -r $svd_out_dir
        exit 1
    fi

else
    echo "Folder $svd_out_dir already exists. Skipping SVD decomposition."
fi

# Test precision and recall on test queries
python test_queries.py $qrels $queries $doc_titles $svd_out_dir $min_rel_docs $n_components $tf_out_doc_ids "$tf_out_vocabulary" "$tfidf_values" "$out_csv"

# Scatter plot of precision vs number of relevant documents
python plot_eff_vs_nrel.py "$out_csv" 'n_rel_docs2' 'ndcg_at_20' 'none' "$out_fig"

