# Assignment 1 – Text Preprocessing (Telugu)

**Course:** AI357 Natural Language Processing  
**Language:** Telugu (`te`)

## Overview

This assignment implements regex-based sentence and word tokenizers, processes two Telugu monolingual corpora, saves tokenized output as compressed parquet files, and computes corpus statistics.

| Task | Corpus | Source |
|------|--------|--------|
| 1 | IndicCorpV2 (Telugu) | [ai4bharat/IndicCorpV2](https://huggingface.co/datasets/ai4bharat/IndicCorpV2) |
| 2 | OSCAR-2301 (Telugu) | [oscar-corpus/OSCAR-2301](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301) |

## Project Structure

```
Assignment1/
├── tokenizer.py          # Regex sentence & word tokenizers
├── corpus_processor.py   # Download, tokenize, save parquet, compute stats
├── run_indiccorp.py      # Run pipeline on IndicCorpV2 Telugu
├── run_oscar.py            # Run pipeline on OSCAR-2301 Telugu
├── requirements.txt
└── output/
    ├── indiccorp/
    │   ├── indiccorp_te_tokenized.parquet
    │   └── indiccorp_te_statistics.json
    └── oscar/
        ├── oscar_te_tokenized.parquet
        └── oscar_te_statistics.json
```

## Setup

```bash
pip install -r requirements.txt
```

Set your HuggingFace token (required for OSCAR; recommended for IndicCorp):

```powershell
# Windows PowerShell
$env:HF_TOKEN = "your_huggingface_token_here"
```

```bash
# Linux / macOS
export HF_TOKEN="your_huggingface_token_here"
```

For OSCAR, you must also accept the dataset terms at the [OSCAR-2301 HuggingFace page](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301).

## Running

```bash
# Task 1 – IndicCorpV2 Telugu
python run_indiccorp.py

# Task 2 – OSCAR-2301 Telugu
python run_oscar.py
```

Both corpora are very large. For a quick test, set `MAX_DOCUMENTS = 1000` at the top of each run script.

## Tokenizer Design (Regex Only)

No NLTK, spaCy, or other NLP tokenization libraries are used. Tokenization is implemented with Python's `re` module.

### Word tokenizer handles

- **URLs** – `http://`, `https://`, `www.` links  
- **Email IDs** – `user@domain.com`  
- **Dates** – `YYYY-MM-DD`, `DD/MM/YYYY`, `DD-MM-YYYY`, `DD.MM.YYYY`  
- **Numbers** – integers and decimals (e.g. `42`, `3.14`)  
- **Punctuation** – each symbol as a separate token (including Telugu danda `।`)  
- **Words** – Unicode word characters including Telugu script  

### Sentence tokenizer

Splits paragraphs on `.`, `!`, `?`, and Telugu danda (`।`) followed by whitespace. Decimal points, URL dots, and email dots are protected from triggering a sentence break.

## Output Format

Each row in the parquet file contains one tokenized sentence in the column `tokenized_sentence`. Tokens within a sentence are joined by spaces, as required:

```
ఈ రోజు 3.14 విలువ . నా ఇమెయిల్ test@example.com .
```

## Corpus Statistics

After tokenization, the following statistics are saved to `*_statistics.json`:

| Statistic | Description |
|-----------|-------------|
| `total_sentences` | Total number of sentences |
| `total_words` | Total number of word tokens |
| `total_characters` | Total characters in original text |
| `average_sentence_length` | Mean words per sentence |
| `average_word_length` | Mean characters per word token |
| `unique_token_types` | Count of distinct tokens |
| `type_token_ratio` | Unique types ÷ total tokens (TTR) |

## Notes

- Data is streamed to avoid loading the full corpus into memory.
- Parquet files use **Snappy** compression to reduce storage size.
- The full IndicCorpV2 Telugu subset is ~275 GB; plan disk space accordingly or use `MAX_DOCUMENTS` for partial runs.
