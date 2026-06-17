#!/usr/bin/env python3
"""
Multilingual topic modeling using BERTopic

This model is multilingual and aims at producing topics across languages.

Usage:
    python topic_model.py [CSV_PATH]

    CSV_PATH: Path to the CSV file (default: data/docs.csv)

Required CSV columns:
- `content`: The text content of each document (string). Rows with empty `content` are dropped.
- `id` (optional): A unique identifier for each document. If absent, the script uses the row index.

Optional metadata columns (any additional columns are preserved and exported):
- `year`, `url`, `domain`, or any other field — carried through to the
  documents_with_topics.csv output for use in metadata visualisations.

Example CSV header:
    id,content,year,url

Examples:
    python topic_model.py
    python topic_model.py data/docs.csv
    python topic_model.py data/docs.csv --domain-prefix my_corpus
"""
import os
import sys
import shutil
import argparse
import pandas as pd
import numpy as np
import torch
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from utils import load_csv, load_custom_stopwords, remove_stopwords, save_array_to_json, get_device, get_output_paths

MIN_DOCUMENTS_PR_TOPIC = 80
AMOUNT_OF_KEYWORDS_PR_TOPIC = 50

device = get_device()

parser = argparse.ArgumentParser(description="Multilingual topic modeling using BERTopic")
parser.add_argument("csv_path", nargs="?", default="data/docs.csv", help="Path to the CSV file (default: data/docs.csv)")
parser.add_argument("--domain-prefix", type=str, default=None, help="Optional prefix for output filenames (e.g., 'my_corpus')")
args = parser.parse_args()

csv_path = args.csv_path
domain_prefix = args.domain_prefix

# 1. Load CSV
documents, doc_ids, metadata_df = load_csv(csv_path)

# 2. Remove stopwords (multilingual)
print("Step 2: Removing stopwords from documents...")
combined_stopwords = load_custom_stopwords()
documents_filtered = [remove_stopwords(doc, combined_stopwords) for doc in documents]
print(f"✓ Stopwords removed\n")

# Calculate adaptive min_df based on dataset size
num_docs = len(documents_filtered)
min_df_percentage = max(0.001, min(0.01, 10 / num_docs)) if num_docs > 0 else 0.001
print(f"Dataset size: {num_docs} documents")
print(f"Adaptive min_df: {min_df_percentage*100:.2f}% (words must appear in this % of documents)\n")

# 3. Create multilingual embedding model
print("Step 3: Loading multilingual embedding model...")
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device.value)
print(f"✓ Model loaded on {device.value}\n")

# 4. Configure BERTopic
print("Step 4: Configuring BERTopic model...")
vectorizer_model = CountVectorizer(
    stop_words=list(combined_stopwords),
    max_features=5000,
    ngram_range=(1, 2),
    min_df=min_df_percentage,
    max_df=0.95
)

topic_model = BERTopic(
    embedding_model=embedding_model,
    vectorizer_model=vectorizer_model,
    min_topic_size=MIN_DOCUMENTS_PR_TOPIC,
    nr_topics="auto",
    calculate_probabilities=False,
    top_n_words=AMOUNT_OF_KEYWORDS_PR_TOPIC,
    verbose=True
)
print(f"✓ BERTopic model configured\n")

# 5. Fit topic model
print("Step 5: Fitting topic model (this may take a while)...")
topics, probabilities = topic_model.fit_transform(documents_filtered)
print(f"✓ Topic modeling complete\n")

# 6. Extract topic information
print("Step 6: Extracting topic information...")
topic_info = topic_model.get_topic_info()
print(f"✓ Found {len(topic_info) - 1} topics (excluding outliers)\n")

# 7. Build topic data
print("Step 7: Preparing topic data for export...")
topic_data = []

for idx, row in topic_info.iterrows():
    topic_id = row['Topic']
    if topic_id == -1:
        continue

    topic_words = topic_model.get_topic(topic_id)
    if topic_words:
        keywords = [word for word, score in topic_words[:AMOUNT_OF_KEYWORDS_PR_TOPIC]]
        keyword_scores = {word: float(score) for word, score in topic_words[:AMOUNT_OF_KEYWORDS_PR_TOPIC]}
    else:
        keywords = []
        keyword_scores = {}

    topic_docs_indices = np.where(np.array(topics) == topic_id)[0]
    num_docs_in_topic = len(topic_docs_indices)

    if num_docs_in_topic > 0:
        sample_doc = documents[topic_docs_indices[0]]
        sample = str(sample_doc)[:200].replace('\n', ' ') + "..."
    else:
        sample = ""

    document_ids = [doc_ids[i] for i in topic_docs_indices[:80]]

    topic_entry = {
        'topic_id': int(topic_id),
        'num_docs': int(num_docs_in_topic),
        'keywords': keywords,
        'keyword_scores': keyword_scores,
        'name': row.get('Name', f"Topic {topic_id}"),
        'sample': sample,
        'document_ids': document_ids
    }
    topic_data.append(topic_entry)

topic_data.sort(key=lambda x: x['num_docs'], reverse=True)
print(f"✓ Prepared {len(topic_data)} topics\n")

# 8. Print results
print("\n" + "="*70)
print("TOPIC MODEL RESULTS")
print("="*70)

for topic in topic_data[:10]:
    print(f"\nTopic {topic['topic_id']}: {topic['name']}")
    print(f"Documents: {topic['num_docs']}")
    print(f"Keywords: {', '.join(topic['keywords'][:AMOUNT_OF_KEYWORDS_PR_TOPIC])}")
    print(f"Sample: {topic['sample']}")
    print("-" * 70)

# 9. Save topic results JSON
output_file, model_path, assignments_path = get_output_paths(domain_prefix)
save_array_to_json(topic_data, output_file)
print(f"\n✓ Full results saved to {output_file}")

# 10. Save documents with topic assignments (preserves all metadata columns)
print("\nStep 9: Saving document-topic assignments...")
assignments_df = metadata_df.copy()
assignments_df['topic_id'] = topics
assignments_df['content'] = documents
os.makedirs(os.path.dirname(assignments_path) or ".", exist_ok=True)
assignments_df.to_csv(assignments_path, index=False)
print(f"✓ Document assignments saved to {assignments_path}")

# 11. Save the model
print("\nStep 10: Saving topic model...")
os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
if os.path.exists(model_path):
    if os.path.isdir(model_path):
        shutil.rmtree(model_path)
    else:
        os.remove(model_path)
    print(f"⚠ Removed existing model: {model_path}")
topic_model.save(model_path)
print(f"✓ Model saved to {model_path}/")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Total documents: {len(documents)}")
print(f"Topics found: {len(topic_data)}")
print(f"Outliers: {len([t for t in topics if t == -1])}")
print(f"\nOutput files:")
print(f"  - {output_file} (topic data)")
print(f"  - {assignments_path} (all documents with topic assignments)")
print(f"  - {model_path}/ (saved model)")
