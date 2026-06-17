#!/usr/bin/env python3
"""
Search topics by keyword.

Searches topic keywords and sample text for one or more terms and prints
matching topics ranked by number of keyword hits.

Usage:
    python search_topics.py <keyword> [keyword2 ...] [--json data/topic_model_results.json]

Examples:
    python search_topics.py climate
    python search_topics.py war conflict violence
    python search_topics.py education school --json data/my_corpus_topic_model_results.json
"""

import json
import sys
import argparse
from pathlib import Path


def load_topic_data(json_path: str) -> list:
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def search_topics(terms: list[str], json_path: str):
    topics = load_topic_data(json_path)
    terms_lower = [t.lower() for t in terms]

    results = []
    for topic in topics:
        keywords_text = ' '.join(topic.get('keywords', [])).lower()
        sample_text = topic.get('sample', '').lower()
        combined = keywords_text + ' ' + sample_text

        matches = [term for term in terms_lower if term in combined]
        if matches:
            results.append((len(matches), topic, matches))

    results.sort(key=lambda x: x[0], reverse=True)

    if not results:
        print(f"\nNo topics found containing: {', '.join(terms)}")
        return

    print(f"\nFound {len(results)} topic(s) matching: {', '.join(terms)}\n")
    print("=" * 80)

    for score, topic, matches in results:
        print(f"\nTopic {topic['topic_id']} ({topic['num_docs']} docs) — {score} match(es): {', '.join(matches)}")
        print(f"Keywords: {', '.join(topic['keywords'][:15])}")
        sample = topic.get('sample', '')
        if sample:
            print(f"Sample: {sample[:200]}...")
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Search topics by keyword")
    parser.add_argument("keywords", nargs="+", help="One or more keywords to search for")
    parser.add_argument("--json", default=None, help="Path to topic_model_results.json")
    args = parser.parse_args()

    if args.json:
        json_path = args.json
    else:
        project_root = Path(__file__).parent.parent.parent
        json_path = str(project_root / "data" / "topic_model_results.json")

    search_topics(args.keywords, json_path)


if __name__ == "__main__":
    main()
