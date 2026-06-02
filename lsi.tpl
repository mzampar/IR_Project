# config for script lsi.sh

data_link="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip"

wd="/u/dssc/mzampar/scratch/IR_Project/"
fig_dir="$wd/fig"
csv_dir="$wd/csv"

data_path="$wd/datasets/trec-covid"
data_path_processed="$wd/datasets/trec-covid-processed"
qrels="$wd/datasets/trec-covid-processed/qrels/test.tsv"
queries="$wd/datasets/trec-covid-processed/queries.jsonl"
doc_titles="$wd/datasets/trec-covid-processed/doc_titles.csv"


n_terms=N_TERMS
n_grams=N_GRAMS
min_df=MIN_DF
max_df=MAX_DF

tf_out_matrix="$wd/tf_idf-${n_terms}_terms-${n_grams}_grams-${min_df}_min_df-${max_df}_max_df/trec-covid-tfidf_sparse.npz"
tf_out_features="$wd/tf_idf-${n_terms}_terms-${n_grams}_grams-${min_df}_min_df-${max_df}_max_df/trec-covid-tfidf_features.json"
tf_out_doc_ids="$wd/tf_idf-${n_terms}_terms-${n_grams}_grams-${min_df}_min_df-${max_df}_max_df/trec-covid-tfidf_docids.json"
tfidf_transformer="$wd/tf_idf-${n_terms}_terms-${n_grams}_grams-${min_df}_min_df-${max_df}_max_df/trec-covid-tfidf_transformer.json"

n_components=N_COMP

min_rel_docs=MIN_REL_DOCS

svd_out_dir="$wd/svd-${n_terms}_terms-${n_grams}_grams-${n_components}_components-${min_df}_min_df-${max_df}_max_df"
sv_plot="$fig_dir/svd-${n_terms}_terms-${n_grams}_grams-500_values.png"

out_csv="$csv_dir/efficiency-${n_terms}_terms-${n_grams}_grams-${n_components}_components-${min_df}_min_df-${max_df}_max_df-${min_rel_docs}_min_rel_docs.tsv"
out_fig="$fig_dir/efficiency-${n_terms}_terms-${n_grams}_grams-${n_components}_components-${min_df}_min_df-${max_df}_max_df-${min_rel_docs}_min_rel_docs.png"

