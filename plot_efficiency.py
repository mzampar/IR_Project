import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the CSV file
files = [
"csv/efficiency--1_terms-1_grams-10_components-15_min_df-0.5_max_df-100_min_rel_docs-summary.csv",
"csv/efficiency--1_terms-1_grams-25_components-15_min_df-0.5_max_df-100_min_rel_docs-summary.csv",
"csv/efficiency--1_terms-1_grams-50_components-15_min_df-0.5_max_df-100_min_rel_docs-summary.csv",
#"csv/efficiency--1_terms-1_grams-100_components-15_min_df-0.5_max_df-100_min_rel_docs-summary.csv",
"csv/efficiency--1_terms-1_grams-150_components-15_min_df-0.5_max_df-100_min_rel_docs-summary.csv",
"csv/efficiency--1_terms-1_grams-200_components-15_min_df-0.5_max_df-100_min_rel_docs-summary.csv",
#"csv/efficiency--1_terms-1_grams-250_components-15_min_df-0.5_max_df-100_min_rel_docs-summary.csv",
"csv/efficiency--1_terms-1_grams-300_components-15_min_df-0.5_max_df-100_min_rel_docs-summary.csv",
]

data_list = []
for file in files:
     data = pd.read_csv(file, sep=',')
     data_list.append(data)
    
df = pd.concat(data_list, ignore_index=True)

plt.figure(figsize=(10, 6))

stat = 'median'
stat = 'mean'
# Select where metric string column starts with "ndcg"
metric = "ndcg"
df = df[df['metric'].str.startswith(metric)]

# Generate blue palette
n_vals = sorted(df['n_components'].unique())
cmap = plt.get_cmap('Blues')
colors = cmap(np.linspace(0.4, 0.9, len(n_vals)))

"""
# Grey + 1 blue palette to highlight the line of interest
colors = np.array([
    [0.49803922, 0.49803922, 0.49803922, 1.0],  # tab:gray
    [0.49803922, 0.49803922, 0.49803922, 1.0],  # tab:gray
    [0.49803922, 0.49803922, 0.49803922, 1.0],  # tab:gray
    [0.17914648, 0.49287197, 0.73542484, 1.0],  # original 4th blue
    [0.49803922, 0.49803922, 0.49803922, 1.0],  # tab:gray
    [0.49803922, 0.49803922, 0.49803922, 1.0],  # tab:gray
])
"""

# Loop on components and plot line for each
for i, n_components in enumerate(sorted(df['n_components'].unique())):

    df_subset = df[df['n_components'] == n_components]

    x_ticks = [m.split('_')[2] for m in df_subset['metric']]
    x_ticks[0] = f'{metric.upper()} @ ' + x_ticks[0]

    y = df_subset[stat].values
    x = np.arange(len(x_ticks))

    color = colors[i % len(colors)]

    # plot line
    plt.plot(
        x,
        y,
        marker='o',
        linewidth=2,
        color=color,
    )

    # label near last point
    if i == 0:
        label = f'{n_components} components'
    else:
        label = f'{n_components}'

    base_offset = (10, 0)
    xytext = (
        base_offset[0],
        base_offset[1] 
    )

    plt.annotate(
        label,
        xy=(x[-1], y[-1]),
        xytext=xytext,
        textcoords='offset points',
        va='center',
        color=color,
        fontsize=10,
        arrowprops=dict(
            arrowstyle='-',
            color=color,
            lw=1.5
        )
    )

plt.xticks(np.arange(len(x_ticks)), x_ticks)

plt.ylabel(f'{stat} NDCG across queries')

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#ax.set_ylim(-0.05, 1.0)
ax.set_ylim(0.0, 0.5)

plt.tight_layout()
#plt.show()
plt.savefig('./test/test-components.png')
#plt.savefig('./test/test-components-grey.png')
