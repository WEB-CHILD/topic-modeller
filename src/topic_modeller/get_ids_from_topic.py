#!/usr/bin/env python3
"""
Extract all document IDs for a specific topic from the topic model results.

Usage:
    python get_ids_from_topic.py <topic_number> [json_path]

Examples:
    python get_ids_from_topic.py 0
    python get_ids_from_topic.py 5 data/topic_model_results.json

Default JSON file: data/topic_model_results.json
"""

import json
import sys
from pathlib import Path


def load_topic_data(json_path: str) -> list:
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_topic_ids(topic_number: int, json_path: str = None):
    if json_path is None:
        project_root = Path(__file__).parent.parent.parent
        json_path = project_root / "data" / "topic_model_results.json"

    topics = load_topic_data(json_path)

    topic_found = None
    for topic in topics:
        if topic.get("topic_id") == topic_number:
            topic_found = topic
            break

    if topic_found is None:
        print(f"Error: Topic {topic_number} not found.")
        print(f"Available topics: {sorted([t.get('topic_id') for t in topics if 'topic_id' in t])}")
        return

    print(f"\n{'='*80}")
    print(f"TOPIC {topic_number}")
    print(f"{'='*80}")
    print(f"Number of documents: {topic_found.get('num_docs', 'N/A')}")
    print(f"Top keywords: {', '.join(topic_found.get('keywords', [])[:10])}")
    print(f"\n{'Document IDs:':-<80}")

    document_ids = topic_found.get("document_ids", [])
    if document_ids:
        print(f"Showing {len(document_ids)} document ID(s) (max 80 per topic):")
        print()
        ids_per_row = 5
        for i in range(0, len(document_ids), ids_per_row):
            row_ids = document_ids[i:i+ids_per_row]
            print("  " + "  ".join(f"{doc_id}" for doc_id in row_ids))
    else:
        print("No document IDs found for this topic.")

    sample = topic_found.get("sample", "")
    if sample:
        print(f"\n{'Sample Document:':-<80}")
        sample_preview = sample[:300] + "..." if len(sample) > 300 else sample
        print(sample_preview)

    print(f"{'='*80}\n")
    return document_ids


def main():
    if len(sys.argv) < 2:
        print("Usage: python get_ids_from_topic.py <topic_number> [json_path]")
        print("\nExample: python get_ids_from_topic.py 5")
        sys.exit(1)

    try:
        topic_number = int(sys.argv[1])
    except ValueError:
        print(f"Error: Topic number must be an integer, got '{sys.argv[1]}'")
        sys.exit(1)

    json_path = sys.argv[2] if len(sys.argv) > 2 else None
    print_topic_ids(topic_number, json_path)


if __name__ == "__main__":
    main()
