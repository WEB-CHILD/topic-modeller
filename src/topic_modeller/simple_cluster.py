#!/usr/bin/env python3
"""
Simple K-means clustering (lightweight alternative to BERTopic).

Usage:
    python simple_cluster.py [CSV_PATH] [--clusters N]

Examples:
    python simple_cluster.py data/docs.csv
    python simple_cluster.py data/docs.csv --clusters 30
"""
import argparse
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import load_csv, load_custom_stopwords, remove_stopwords, save_array_to_json

parser = argparse.ArgumentParser(description="K-means topic clustering")
parser.add_argument("csv_path", nargs="?", default="data/docs.csv", help="Path to input CSV (default: data/docs.csv)")
parser.add_argument("--clusters", type=int, default=20, help="Number of clusters (default: 20)")
args = parser.parse_args()

if torch.backends.mps.is_available():
    device = "mps"
    print("✓ Using Apple Silicon GPU (MPS) for acceleration\n")
elif torch.cuda.is_available():
    device = "cuda"
    print(f"✓ Using NVIDIA GPU\n")
else:
    device = "cpu"
    print("⚠ No GPU available, using CPU\n")

# 1. Load CSV
documents, doc_ids, _ = load_csv(args.csv_path)

# 2. Remove stopwords
print("Step 2: Removing stopwords from documents...")
combined_stopwords = load_custom_stopwords()
documents_filtered = [remove_stopwords(doc, combined_stopwords) for doc in documents]
print(f"✓ Stopwords removed\n")

# 3. Create embeddings
print("Step 3: Creating embeddings with SentenceTransformer...")
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
embeddings = model.encode(documents_filtered, show_progress_bar=True, batch_size=64, convert_to_numpy=True)
print(f"✓ Created embeddings with shape {embeddings.shape}\n")

# 4. Cluster
num_clusters = args.clusters
print(f"Step 4: Clustering documents into {num_clusters} clusters...")
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(embeddings)
print(f"✓ Clustering complete\n")

# 5. Extract top keywords per cluster using TF-IDF
print("Step 5: Extracting keywords for each cluster...")
df_clustered = pd.DataFrame({"document": documents, "cluster": labels})

def get_keywords(docs, n=10):
    if not docs:
        return []
    vectorizer = TfidfVectorizer(max_features=5000, stop_words=list(combined_stopwords))
    X = vectorizer.fit_transform(docs)
    if X.shape[1] == 0:
        return []
    scores = X.sum(axis=0).A1
    indices = np.argsort(scores)[::-1][:n]
    return np.array(vectorizer.get_feature_names_out())[indices].tolist()

# 6. Print and collect results
cluster_data = []
print("\n" + "="*70)

for cluster_id in range(num_clusters):
    cluster_docs = df_clustered[df_clustered.cluster == cluster_id]['document'].tolist()
    keywords = get_keywords(cluster_docs, n=20)

    cluster_info = {
        'cluster_id': cluster_id,
        'num_docs': len(cluster_docs),
        'keywords': keywords,
        'sample': str(cluster_docs[0])[:150].replace('\n', ' ') + "..." if cluster_docs else ""
    }
    cluster_data.append(cluster_info)

    print(f"\nCluster {cluster_id} ({len(cluster_docs)} docs)")
    print(f"Keywords: {', '.join(keywords)}")
    if cluster_docs:
        print(f"Sample: {cluster_info['sample']}")
    print("-" * 70)

output_file = "data/cluster_keywords.json"
save_array_to_json(cluster_data, output_file)
