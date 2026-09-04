# topic-modeller

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22296669.svg)](https://doi.org/10.5281/zenodo.22296669)

Multilingual topic modeling pipeline based on BERTopic. Accepts any CSV with a `content` column and optional metadata (year, url, domain, id, etc.). Identifies and visualises the main themes across your documents, with support for metadata-aware breakdowns by any column.

Originally developed for [KIDLINK-Topics](https://github.com/WEB-CHILD/KIDLINK-Topics).

## Features

- Multilingual BERTopic with sentence transformers
- Adaptive settings that scale to small and large datasets
- Metadata-aware visualisations (by year, domain, URL, or any column)
- Batch processing for multiple corpora
- GPU acceleration (NVIDIA CUDA, Apple Silicon MPS, CPU fallback)

## Input format

Required column:
- `content`: text of each document

Optional columns (any are preserved and usable in visualisations):
- `id`: document identifier (defaults to row index if absent)
- `year`, `url`, `domain`, or any other metadata field

```
id,content,year,url
doc1,"Some text here...",2003,https://example.com/page1
doc2,"More text...",2004,https://example.com/page2
```

## Project structure

```
topic-modeller/
├── src/topic_modeller/
│   ├── topic_model.py            # Main BERTopic pipeline
│   ├── visualise_topics.py       # Word clouds, bar charts, heatmaps
│   ├── visualise_metadata.py     # Breakdowns by metadata column
│   ├── simple_cluster.py         # Lightweight K-means alternative
│   ├── get_keywords_from_topic.py
│   ├── get_ids_from_topic.py
│   ├── search_topics.py
│   └── utils.py
├── batch_topic_models.py         # Batch runner for multiple corpora
├── batch_visualise.py
└── data/
    └── custom_stopwords.txt
```

## Setup

```bash
cd topic-modeller
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install bertopic sentence-transformers scikit-learn pandas numpy matplotlib wordcloud nltk torch
```

## Usage

### Run topic model

```bash
# Default: reads data/docs.csv
python src/topic_modeller/topic_model.py

# Specify input file
python src/topic_modeller/topic_model.py data/my_corpus.csv

# With a prefix for output filenames
python src/topic_modeller/topic_model.py data/my_corpus.csv --domain-prefix my_corpus
```

Output files:
- `data/{prefix_}topic_model_results.json` — topic summaries with keywords
- `data/{prefix_}documents_with_topics.csv` — all documents with topic assignments + metadata
- `models/{prefix_}topic_model_bertopic/` — saved BERTopic model

### Visualise topics

```bash
python src/topic_modeller/visualise_topics.py
python src/topic_modeller/visualise_topics.py --domain-prefix my_corpus
```

Outputs to `visualisations/topics/`.

### Visualise by metadata

Requires a metadata column in the original CSV (e.g. `year`, `url`, `domain`).

```bash
# Topic share over time
python src/topic_modeller/visualise_metadata.py --group-by year

# Topic distribution per domain
python src/topic_modeller/visualise_metadata.py --group-by domain --domain-prefix my_corpus

# Show only top 8 topics
python src/topic_modeller/visualise_metadata.py --group-by year --top-topics 8
```

Outputs to `visualisations/metadata/`.

### Query tools

```bash
# Print all keywords for topic 3
python src/topic_modeller/get_keywords_from_topic.py 3

# Print document IDs for topic 3
python src/topic_modeller/get_ids_from_topic.py 3

# Search topics by keyword
python src/topic_modeller/search_topics.py climate energy
```

### Batch processing

Config CSV format (`batch_config.csv`):
```
att,domain,csv-name
run1,corpus-a,corpus_a.csv
run1,corpus-b,corpus_b.csv
```

```bash
python batch_topic_models.py batch_config.csv
python batch_visualise.py batch_config.csv
```

## Custom stopwords

Add domain-specific words to `data/custom_stopwords.txt` (one per line). These are combined with multilingual NLTK stopwords covering Danish, Norwegian, English, Spanish, German, Italian, Portuguese, French, Swedish, Dutch, Finnish, Russian, Turkish, and Arabic.

## Parameters

Key settings in `topic_model.py`:
- `MIN_DOCUMENTS_PR_TOPIC = 80` — minimum documents to form a topic
- `AMOUNT_OF_KEYWORDS_PR_TOPIC = 50` — keywords extracted per topic
- Embedding model: `paraphrase-multilingual-MiniLM-L12-v2`
