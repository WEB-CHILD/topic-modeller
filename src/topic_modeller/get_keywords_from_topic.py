#!/usr/bin/env python3
"""
Print all keywords for a specific topic from the topic model results.

Usage:
    python get_keywords_from_topic.py <topic_number> [json_path]

Examples:
    python get_keywords_from_topic.py 0
    python get_keywords_from_topic.py 5 data/topic_model_results.json

Default JSON file: data/topic_model_results.json
"""

import json
import sys
from pathlib import Path


def load_topic_data(json_path: str) -> list:
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_topic_keywords(topic_number: int, json_path: str = None):
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
    print(f"\n{'Keywords:':-<80}")

    keywords = topic_found.get("keywords", [])
    if keywords:
        for i, keyword in enumerate(keywords, 1):
            print(f"{i:3d}. {keyword}")
    else:
        print("No keywords found for this topic.")

    sample = topic_found.get("sample", "")
    if sample:
        print(f"\n{'Sample Document:':-<80}")
        sample_preview = sample[:300] + "..." if len(sample) > 300 else sample
        print(sample_preview)

    print(f"{'='*80}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python get_keywords_from_topic.py <topic_number> [json_path]")
        print("\nExample: python get_keywords_from_topic.py 5")
        sys.exit(1)

    try:
        topic_number = int(sys.argv[1])
    except ValueError:
        print(f"Error: Topic number must be an integer, got '{sys.argv[1]}'")
        sys.exit(1)

    json_path = sys.argv[2] if len(sys.argv) > 2 else None
    print_topic_keywords(topic_number, json_path)


if __name__ == "__main__":
    main()
