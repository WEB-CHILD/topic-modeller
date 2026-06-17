#!/usr/bin/env python3
"""
Visualisations for BERTopic model results.

Visualizations Generated:
1. Word clouds for all topics (grid layout)
2. Topic overview bar chart (document counts + top keywords)
3. Topic distribution pie chart (top 10 topics)
4. Individual high-resolution word clouds per topic
5. Keyword importance heatmap (top 10 topics × top 10 keywords)

Input:
    data/topic_model_results.json  (or data/{PREFIX}_topic_model_results.json)

Output:
    visualisations/topics/
    - {PREFIX_}wordclouds_all_topics.png
    - {PREFIX_}topic_overview.png
    - {PREFIX_}topic_distribution.png
    - {PREFIX_}keyword_heatmap.png
    - wordclouds/{PREFIX_}topic_XX.png

Usage:
    python visualise_topics.py
    python visualise_topics.py --domain-prefix my_corpus
"""
import json
import argparse
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import os
from utils import save_figure, get_topic_results_path, find_cjk_font

parser = argparse.ArgumentParser(description="Generate topic model visualizations")
parser.add_argument("--domain-prefix", type=str, default=None, help="Optional prefix matching the topic model output files (e.g., 'my_corpus')")
args = parser.parse_args()

domain_prefix = args.domain_prefix

display_domain = None
if domain_prefix:
    m = re.match(r'^(?:att|p)\d+_(.+)$', domain_prefix)
    if m:
        domain_only = m.group(1)
    else:
        domain_only = domain_prefix.split("_", 1)[1] if "_" in domain_prefix else domain_prefix
    display_domain = domain_only.replace("-", ".")

input_json = get_topic_results_path(domain_prefix)
output_prefix = f"{domain_prefix}_" if domain_prefix else ""

print("Loading topic model results...")
with open(input_json, 'r', encoding='utf-8') as f:
    topic_data = json.load(f)
print(f"✓ Loaded {len(topic_data)} topics from {input_json}\n")

cjk_font = find_cjk_font()
if cjk_font:
    print(f"✓ Using CJK-compatible font: {cjk_font}\n")
    plt.rcParams['font.sans-serif'] = [cjk_font]
    plt.rcParams['axes.unicode_minus'] = False
else:
    print("⚠ No CJK font found - wordclouds may not display non-Latin characters properly\n")

os.makedirs('visualisations/topics/wordclouds', exist_ok=True)

# 1. Word cloud grid
print("Generating topic wordclouds...")
num_topics = len(topic_data)
cols = 4
rows = (num_topics + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
axes = axes.flatten() if num_topics > 1 else [axes]

for i, topic in enumerate(topic_data):
    if topic.get('keyword_scores'):
        word_freq = topic['keyword_scores']
    else:
        keywords = topic['keywords']
        word_freq = {word: len(keywords) - idx for idx, word in enumerate(keywords)}

    wc = WordCloud(width=400, height=300, background_color='white',
                   colormap='plasma', relative_scaling=0.5,
                   font_path=cjk_font).generate_from_frequencies(word_freq)

    axes[i].imshow(wc, interpolation='bilinear')
    topic_name = topic.get('name', f"Topic {topic['topic_id']}")
    title_lines = [f"{topic_name}", f"({topic['num_docs']} docs)"]
    if display_domain:
        title_lines.append(display_domain)
    axes[i].set_title("\n".join(title_lines), fontsize=9, fontweight='bold')
    axes[i].axis('off')

for i in range(num_topics, len(axes)):
    axes[i].axis('off')

plt.tight_layout()
save_figure(f'visualisations/topics/{output_prefix}wordclouds_all_topics.png', fig, dpi=150, bbox_inches='tight')
print(f"✓ Saved visualisations/topics/{output_prefix}wordclouds_all_topics.png\n")
plt.close()

# 2. Topic sizes bar chart
print("Generating topic overview...")
fig, ax = plt.subplots(figsize=(14, max(8, num_topics * 0.4)))

topic_ids = [t['topic_id'] for t in topic_data]
top_keywords = [', '.join(t['keywords'][:5]) for t in topic_data]

y_pos = np.arange(len(topic_ids))
colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(topic_ids)))

ax.barh(y_pos, [t['num_docs'] for t in topic_data], color=colors)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"T{tid}: {kw}" for tid, kw in zip(topic_ids, top_keywords)], fontsize=8)
ax.set_xlabel('Number of Documents', fontsize=10)
title = 'Topic Sizes and Top 5 Keywords'
if display_domain:
    title = f"{title} — {display_domain}"
ax.set_title(title, fontsize=12, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
save_figure(f'visualisations/topics/{output_prefix}topic_overview.png', fig, dpi=150, bbox_inches='tight')
print(f"✓ Saved visualisations/topics/{output_prefix}topic_overview.png\n")
plt.close()

# 3. Distribution pie chart
print("Generating topic distribution chart...")
top_n = min(10, len(topic_data))
top_topics = topic_data[:top_n]
other_docs = sum(t['num_docs'] for t in topic_data[top_n:])

labels = [f"T{t['topic_id']}: {', '.join(t['keywords'][:2])}" for t in top_topics]
sizes = [t['num_docs'] for t in top_topics]

if other_docs > 0:
    labels.append(f"Other ({len(topic_data) - top_n} topics)")
    sizes.append(other_docs)

fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))
wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                   colors=colors, startangle=90)

for text in texts:
    text.set_fontsize(9)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(9)

title = 'Topic Distribution (Top 10 Topics)'
if display_domain:
    title = f"{title} — {display_domain}"
ax.set_title(title, fontsize=14, fontweight='bold')
plt.tight_layout()
save_figure(f'visualisations/topics/{output_prefix}topic_distribution.png', fig, dpi=150, bbox_inches='tight')
print(f"✓ Saved visualisations/topics/{output_prefix}topic_distribution.png\n")
plt.close()

# 4. Individual wordclouds
print("Generating individual topic wordclouds...")
for topic in topic_data:
    if topic.get('keyword_scores'):
        word_freq = topic['keyword_scores']
    else:
        keywords = topic['keywords']
        word_freq = {word: len(keywords) - idx for idx, word in enumerate(keywords)}

    wc = WordCloud(width=800, height=600, background_color='white',
                   colormap='plasma', relative_scaling=0.5,
                   font_path=cjk_font).generate_from_frequencies(word_freq)

    plt.figure(figsize=(10, 7))
    plt.imshow(wc, interpolation='bilinear')
    topic_name = topic.get('name', f"Topic {topic['topic_id']}")
    title = f"{topic_name} - {topic['num_docs']} documents"
    if display_domain:
        title = f"{title} — {display_domain}"
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    fig = plt.gcf()
    save_figure(f"visualisations/topics/wordclouds/{output_prefix}topic_{topic['topic_id']:02d}.png",
                fig, dpi=150, bbox_inches='tight')
    plt.close()

print(f"✓ Saved individual wordclouds to visualisations/topics/wordclouds/\n")

# 5. Keyword importance heatmap
print("Generating keyword importance heatmap...")
top_n_topics = min(10, len(topic_data))
top_topics = topic_data[:top_n_topics]
max_keywords = 10
topic_labels = []
keyword_matrix = []

for topic in top_topics:
    topic_name = topic.get('name', f"T{topic['topic_id']}")
    topic_labels.append(topic_name[:30])

    if topic.get('keyword_scores'):
        scores = list(topic['keyword_scores'].values())[:max_keywords]
        scores.extend([0] * (max_keywords - len(scores)))
    else:
        num_kw = min(len(topic['keywords']), max_keywords)
        scores = list(range(num_kw, 0, -1))
        scores.extend([0] * (max_keywords - num_kw))

    keyword_matrix.append(scores)

keyword_matrix = np.array(keyword_matrix)

all_keywords = []
for topic in top_topics:
    all_keywords.extend(topic['keywords'][:max_keywords])
unique_keywords = list(dict.fromkeys(all_keywords))[:max_keywords]

fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(keyword_matrix, cmap='YlOrRd', aspect='auto')

ax.set_xticks(np.arange(max_keywords))
ax.set_yticks(np.arange(len(topic_labels)))
ax.set_xticklabels([f"KW{i+1}" for i in range(max_keywords)], fontsize=9)
ax.set_yticklabels(topic_labels, fontsize=9)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Keyword Importance", rotation=-90, va="bottom")

title = "Top 10 Topics — Keyword Importance Heatmap"
if display_domain:
    title = f"{title} — {display_domain}"
ax.set_title(title, fontsize=12, fontweight='bold')
fig.tight_layout()
save_figure(f'visualisations/topics/{output_prefix}keyword_heatmap.png', fig, dpi=150, bbox_inches='tight')
print(f"✓ Saved visualisations/topics/{output_prefix}keyword_heatmap.png\n")
plt.close()

print("=" * 70)
print("Done! Generated:")
print(f"  - visualisations/topics/{output_prefix}wordclouds_all_topics.png")
print(f"  - visualisations/topics/{output_prefix}topic_overview.png")
print(f"  - visualisations/topics/{output_prefix}topic_distribution.png")
print(f"  - visualisations/topics/{output_prefix}keyword_heatmap.png")
print(f"  - visualisations/topics/wordclouds/{output_prefix}topic_XX.png (per topic)")
print("=" * 70)
