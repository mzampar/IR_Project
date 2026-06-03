# **********************************************************
# plot_eff_vs_nrel.py
#
# This script plots, for each query, the number of 
# relevant documents vs the performance of the system on that query
#
# **********************************************************

# --- MODULES ---

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr
import sys

# --- INPUT ARGS ---

infile, x_col, y_col, title, outfile = sys.argv[1:]

# --- MAIN --- 

# Load the data from the CSV file
data=pd.read_csv(infile, sep='\t')

# Extract the relevant columns
nrel_values = data[x_col]
efficiency_values = data[y_col]
names = data['query_id']

# Compute mean efficiency
mean_efficiency = efficiency_values.mean()

# Compute correlations
pearson_corr, p1 = pearsonr(nrel_values, efficiency_values)
spearman_corr, p2 = spearmanr(nrel_values, efficiency_values)

plt.figure(figsize=(10, 6))

for i, name in enumerate(names):
    plt.annotate(name, (nrel_values[i], efficiency_values[i]),
                 textcoords="offset points", xytext=(0,10), ha='center')

plt.scatter(nrel_values, efficiency_values, alpha=0.7)

#plt.xlabel('# relevant documents')
plt.ylabel(y_col.replace('_', ' ').replace('at','@').upper())
#plt.grid(True)

if title.lower() != 'none':
    plt.title(title)

# Legend 
"""
legend_text = (f"Spearman: {spearman_corr:.2f} - {p2:.2e}\n"
               f"Mean performance: {mean_efficiency:.2f}")
legend_text = (f"Mean performance: {mean_efficiency:.2f}\n"
               f"Spearman coefficient: {spearman_corr:.2f}")

dummy = Line2D([], [], linestyle='none')
plt.legend([dummy], [legend_text], loc='lower right', frameon=False)
"""
print(f"Pearson correlation: {pearson_corr:.2f} (p={p1:.2e})")
print(f"Spearman correlation: {spearman_corr:.2f} (p={p2:.2e})")
print(f"Mean performance: {mean_efficiency:.2f}")

# Remove top and right edges of the plot 
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel("relevant documents")
# Move label toward right end
ax.xaxis.set_label_coords(0.95, -0.08)
ax.xaxis.set_label_coords(1.01, -0.02)

if efficiency_values.max() <= 1.0:
    ax.set_ylim(-0.05, 1.0) 
else:
    ax.set_ylim(-0.05, 1.05)


plt.savefig(outfile.replace('.png',f'-{y_col}.png'))
