#!/bin/bash

# **********************************************************
# grid_search_lsi.sh
#
# This script loops on multiple hyper-parameters and submits
# each job.
# A config template is needed to generate temporary config files.
#
# Run this script with:
#
# bash grid_search_lsi.sh lsi.sh lsi.tpl
# 
# **********************************************************

# --- MAIN ---


sh=$1          # sbatch script
tpl=$2         # template config file

set -e

count=0

min_df_vals="15"
max_df_vals="0.5"
n_term_vals='-1'
n_comp_vals="10 25 50 100 150 200 250 300"
grams_vals="1"
min_rel_docs_vals="100"

mkdir -p scratch

for min_df in $min_df_vals; do
  for max_df in $max_df_vals; do
    for n_terms in $n_term_vals; do 
      for n_comp in $n_comp_vals; do
        for n_grams in $grams_vals; do
          for min_rel_docs in $min_rel_docs_vals; do

            temp_config=$(mktemp)
      
            # Substitute placeholders in template
            sed -e "s/N_TERMS/${n_terms}/g" \
                -e "s/N_GRAMS/${n_grams}/g" \
                -e "s/MIN_DF/${min_df}/g" \
                -e "s/MAX_DF/${max_df}/g" \
                -e "s/N_COMP/${n_comp}/g" \
                -e "s/MIN_REL_DOCS/${min_rel_docs}/g" \
                "$tpl" > "$temp_config"

            fname=$(basename "$temp_config")
            mv "$temp_config" "./scratch/$fname"
            
            # Submit job
            sbatch "$sh" "./scratch/$fname"
            count=$((count + 1))

          done 
        done
      done
    done
  done
done

echo $count


