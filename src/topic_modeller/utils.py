# Utility functions for text processing and visualization
import pandas as pd

import nltk
from enum import Enum
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
import sys
from pathlib import Path


def load_csv(file_path):
    """Load documents from a CSV file.

    Returns:
        tuple: (documents, doc_ids, metadata_df)
            - documents: list of content strings
            - doc_ids: list of document identifiers
            - metadata_df: DataFrame of all non-content columns, indexed to match documents
    """
    df = pd.read_csv(file_path)
    df_clean = df.dropna(subset=['content']).reset_index(drop=True)
    documents = df_clean['content'].tolist()

    if 'id' in df_clean.columns:
        doc_ids = df_clean['id'].tolist()
    else:
        doc_ids = df_clean.index.tolist()

    metadata_df = df_clean.drop(columns=['content'])

    print(f"✓ Loaded {len(documents)} documents\n")
    return documents, doc_ids, metadata_df


def load_custom_stopwords():
    """Load and return a combined set of stopwords from various languages and custom file."""
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

    danish_sw = set(stopwords.words("danish"))
    norwegian_sw = set(stopwords.words("norwegian"))
    spanish_sw = set(stopwords.words("spanish"))
    english_sw = set(stopwords.words("english"))
    german_sw = set(stopwords.words("german"))
    italian_sw = set(stopwords.words("italian"))
    portuguese_sw = set(stopwords.words("portuguese"))
    french_sw = set(stopwords.words("french"))
    swedish_sw = set(stopwords.words("swedish"))
    dutch_sw = set(stopwords.words("dutch"))
    finnish_sw = set(stopwords.words("finnish"))
    russian_sw = set(stopwords.words("russian"))
    turkish_sw = set(stopwords.words("turkish"))
    arabic_sw = set(stopwords.words("arabic"))

    custom_stopwords = set()
    custom_stopwords_file = "data/custom_stopwords.txt"
    try:
        with open(custom_stopwords_file, 'r', encoding='utf-8') as f:
            custom_stopwords = set(line.strip().lower() for line in f if line.strip())
        print(f"  Loaded {len(custom_stopwords)} custom stopwords from {custom_stopwords_file}")
    except FileNotFoundError:
        print(f"  No custom stopwords file found (looked for {custom_stopwords_file})")

    combined_stopwords = set(w.lower() for w in (
        ENGLISH_STOP_WORDS |
        english_sw | danish_sw | norwegian_sw | spanish_sw |
        german_sw | italian_sw | portuguese_sw |
        french_sw | swedish_sw | dutch_sw | finnish_sw |
        russian_sw | turkish_sw | arabic_sw
    )) | custom_stopwords

    return combined_stopwords


class Device(Enum):
    """Enumeration of supported device strings for PyTorch/transformers."""
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


def remove_stopwords(text, combined_stopwords):
    """Remove stopwords from text, tokenizing on whitespace."""
    if not isinstance(text, str):
        return ""
    tokens = [t.lower() for t in text.split() if t.lower() and t.lower() not in combined_stopwords]
    filtered = " ".join(tokens)
    return filtered if filtered.strip() else text


def save_array_to_json(array, file_path):
    """Save a Python list to a JSON file."""
    import json
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(array, f, ensure_ascii=False, indent=4)
    print(f"✓ Saved data to {file_path}\n")


def save_figure(filepath, fig, **kwargs):
    """Save matplotlib figure, creating directories as needed."""
    import os
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    if os.path.exists(filepath):
        print(f"⚠ Overwriting existing file: {filepath}")
    fig.savefig(filepath, **kwargs)


def get_device():
    """Detect and return the best available device for PyTorch computation.

    Returns:
        Device: Device.CUDA, Device.MPS, or Device.CPU
    """
    import torch

    if torch.cuda.is_available():
        device = Device.CUDA
        device_name = torch.cuda.get_device_name(0)
        print(f"✓ Using NVIDIA GPU: {device_name}\n")
        return device

    if torch.backends.mps.is_available():
        device = Device.MPS
        print("✓ Using Apple Silicon GPU (MPS) for acceleration\n")
        return device

    device = Device.CPU
    print("⚠ No GPU available, using CPU\n")
    return device


def find_cjk_font():
    """Find a font that supports CJK characters (Chinese, Japanese, Korean)."""
    font_candidates = []

    if sys.platform == "darwin":
        font_candidates = [
            "/Library/Fonts/AppleGothic.ttf",
            "/System/Library/Fonts/PingFang.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
            "/System/Library/Fonts/Hiragino Sans W3.otf",
        ]
    elif sys.platform == "linux":
        font_candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttf",
            "/usr/share/fonts/opentype/dejavu/DejaVuSans.ttf",
        ]
    elif sys.platform == "win32":
        font_candidates = [
            "C:\\Windows\\Fonts\\msyh.ttc",
            "C:\\Windows\\Fonts\\Arial.ttf",
        ]

    for font_path in font_candidates:
        if Path(font_path).exists():
            return font_path

    return None


def get_output_paths(domain_prefix=None):
    """Generate output file paths with optional domain prefix.

    Returns:
        tuple: (output_json_path, output_model_path, output_assignments_csv_path)
    """
    json_path = get_topic_results_path(domain_prefix)
    prefix = f"{domain_prefix}_" if domain_prefix else ""
    model_path = f"models/{prefix}topic_model_bertopic"
    assignments_path = f"data/{prefix}documents_with_topics.csv"
    return json_path, model_path, assignments_path


def get_topic_results_path(domain_prefix=None):
    """Generate the path to topic model results JSON file."""
    prefix = f"{domain_prefix}_" if domain_prefix else ""
    return f"data/{prefix}topic_model_results.json"
