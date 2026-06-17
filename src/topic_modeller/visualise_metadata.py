#!/usr/bin/env python3
"""
Metadata-aware visualisations for topic model results.

Requires the documents_with_topics.csv produced by topic_model.py alongside
the standard topic_model_results.json. Uses any metadata column in that CSV
(e.g. year, url, domain) to break down topic distributions.

Visualisations Generated:
1. Topic share over time        (--group-by year  → stacked area / bar chart)
2. Topic distribution per group (--group-by <col> → stacked horizontal bar)
3. Heatmap: topics × group values

Input:
    data/{PREFIX_}topic_model_results.json
    data/{PREFIX_}documents_with_topics.csv

Output:
    visualisations/metadata/
    - {PREFIX_}{GROUP_BY}_topic_share.png
    - {PREFIX_}{GROUP_BY}_stacked_bar.png
    - {PREFIX_}{GROUP_BY}_heatmap.png

Usage:
    python visualise_metadata.py --group-by year
    python visualise_metadata.py --group-by domain --domain-prefix my_corpus
    python visualise_metadata.py --group-by url --top-topics 8
"""
import json
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import save_figure, get_topic_results_path

parser = argparse.ArgumentParser(description="Metadata-aware topic visualisations")
parser.add_argument("--group-by", required=True, help="Metadata column to group by (e.g. year, url, domain)")
parser.add_argument("--domain-prefix", type=str, default=None, help="Prefix used when running topic_model.py")
parser.add_argument("--top-topics", type=int, default=10, help="Number of top topics to show (default: 10)")
args = parser.parse_args()

group_col = args.group_by
domain_prefix = args.domain_prefix
top_n = args.top_topics

prefix = f"{domain_prefix}_" if domain_prefix else ""
results_json = get_topic_results_path(domain_prefix)
assignments_csv = f"data/{prefix}documents_with_topics.csv"
output_dir = "visualisations/metadata"
os.makedirs(output_dir, exist_ok=True)

# Load data
print(f"Loading topic results from {results_json}...")
with open(results_json, 'r', encoding='utf-8') as f:
    topic_data = json.load(f)

print(f"Loading document assignments from {assignments_csv}...")
df = pd.read_csv(assignments_csv)

if group_col not in df.columns:
    available = [c for c in df.columns if c not in ('content', 'topic_id')]
    print(f"Error: column '{group_col}' not found in {assignments_csv}")
    print(f"Available metadata columns: {available}")
    raise SystemExit(1)

# Drop outliers (topic_id == -1)
df = df[df['topic_id'] != -1].copy()

# Build label map: topic_id → short label (top 3 keywords)
topic_labels = {}
for t in topic_data:
    kws = ', '.join(t['keywords'][:3])
    topic_labels[t['topic_id']] = f"T{t['topic_id']}: {kws}"

# Identify top N topics by document count
top_topic_ids = [t['topic_id'] for t in topic_data[:top_n]]

# Pivot: rows = group values, columns = topic_ids
df['group'] = df[group_col].astype(str)
pivot = df[df['topic_id'].isin(top_topic_ids)].groupby(['group', 'topic_id']).size().unstack(fill_value=0)
pivot = pivot.reindex(columns=top_topic_ids, fill_value=0)
pivot.columns = [topic_labels.get(tid, f"T{tid}") for tid in pivot.columns]

# Sort groups: try numeric sort for year-like columns, else alphabetic
try:
    pivot = pivot.loc[sorted(pivot.index, key=lambda x: float(x))]
except ValueError:
    pivot = pivot.sort_index()

colors = plt.cm.tab20(np.linspace(0, 1, len(pivot.columns)))

# 1. Stacked area chart (good for time series) or bar (for categorical)
is_temporal = group_col.lower() in ('year', 'date', 'month', 'decade')

print(f"Generating topic share chart grouped by '{group_col}'...")
fig, ax = plt.subplots(figsize=(14, 6))

if is_temporal:
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pivot_pct.plot.area(ax=ax, colormap='tab20', alpha=0.85)
    ax.set_ylabel("Topic share (%)", fontsize=10)
else:
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pivot_pct.plot.bar(ax=ax, stacked=True, colormap='tab20', alpha=0.9, width=0.85)
    ax.set_ylabel("Topic share (%)", fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)

ax.set_xlabel(group_col.capitalize(), fontsize=10)
ax.set_title(f"Topic Share by {group_col.capitalize()}", fontsize=13, fontweight='bold')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=7, title="Topics")
plt.tight_layout()
out_path = f"{output_dir}/{prefix}{group_col}_topic_share.png"
save_figure(out_path, fig, dpi=150, bbox_inches='tight')
print(f"✓ Saved {out_path}\n")
plt.close()

# 2. Stacked horizontal bar (absolute counts)
print(f"Generating stacked bar chart grouped by '{group_col}'...")
fig, ax = plt.subplots(figsize=(14, max(6, len(pivot) * 0.4)))
pivot.plot.barh(ax=ax, stacked=True, colormap='tab20', alpha=0.9)
ax.set_xlabel("Number of Documents", fontsize=10)
ax.set_ylabel(group_col.capitalize(), fontsize=10)
ax.set_title(f"Topic Distribution by {group_col.capitalize()}", fontsize=13, fontweight='bold')
ax.legend(loc='upper left', bbox_to_anchor=(1, 0), fontsize=7, title="Topics")
plt.tight_layout()
out_path = f"{output_dir}/{prefix}{group_col}_stacked_bar.png"
save_figure(out_path, fig, dpi=150, bbox_inches='tight')
print(f"✓ Saved {out_path}\n")
plt.close()

# 3. Heatmap: groups × topics (normalised per group)
print(f"Generating heatmap grouped by '{group_col}'...")
heat_data = pivot.div(pivot.sum(axis=1), axis=0).fillna(0)

fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 1.2), max(6, len(pivot) * 0.35)))
im = ax.imshow(heat_data.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

ax.set_xticks(np.arange(len(heat_data.columns)))
ax.set_yticks(np.arange(len(heat_data.index)))
ax.set_xticklabels(heat_data.columns, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(heat_data.index, fontsize=8)
ax.set_xlabel("Topic", fontsize=10)
ax.set_ylabel(group_col.capitalize(), fontsize=10)
ax.set_title(f"Topic Share Heatmap by {group_col.capitalize()}", fontsize=13, fontweight='bold')

cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
cbar.ax.set_ylabel("Share of group docs", rotation=-90, va="bottom", fontsize=9)
plt.tight_layout()
out_path = f"{output_dir}/{prefix}{group_col}_heatmap.png"
save_figure(out_path, fig, dpi=150, bbox_inches='tight')
print(f"✓ Saved {out_path}\n")
plt.close()

print("=" * 70)
print("Done! Generated:")
print(f"  - {output_dir}/{prefix}{group_col}_topic_share.png")
print(f"  - {output_dir}/{prefix}{group_col}_stacked_bar.png")
print(f"  - {output_dir}/{prefix}{group_col}_heatmap.png")
print("=" * 70)
