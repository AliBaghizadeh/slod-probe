# SLoD Probing Prototype

Prototype implementation for probing Semantic Level of Detail (SLoD) in scientific text.

The prototype builds weak labels from QASPER document structure, extracts frozen embeddings, and trains a linear probe for three labels:

- `macro`
- `meso`
- `micro`

Scientific papers do not communicate at a single level of abstraction. Some spans summarize the whole contribution at a high level, some organize the discourse inside a section, and some describe concrete implementation or experimental detail. This project studies that variation as Semantic Level of Detail (SLoD): `macro` for global summary language, `meso` for intermediate organizing and section-level discourse, and `micro` for low-level technical content. The broader motivation is to understand whether these discourse levels are recoverable from frozen language-model representations, or whether apparent success mostly reflects surface structure such as section identity, text length, or common lexical cues.

The repository therefore serves two linked purposes. First, it provides a working weak-supervision pipeline that turns structured scientific papers into a labeled probing dataset using document structure as a proxy for abstraction level. Second, it evaluates whether simple linear probes on frozen embeddings can recover that signal under controlled conditions. In practical terms, the project is asking whether models such as SciBERT and MiniLM encode a usable notion of discourse abstraction, and whether the hardest middle band, `meso`, behaves like a real semantic layer or mostly reflects noisy boundaries between overview and detail.

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

Run the prototype exactly as:

```bash
python src/probe.py --train --eval --condition in_domain
python src/probe.py --train --eval --condition controlled
```

Conditions:

- `in_domain`: standard paper-level split on `data/processed/refined_qasper_slod.csv`
- `controlled`: length-controlled version on `data/processed/refined_qasper_slod_length_controlled.csv`

Models:

- `scibert` (default)
- `minilm`
- `tfidf`

Example:

```bash
python src/probe.py --train --eval --condition in_domain --model scibert
```

## Dataset and Labels

Weak labels follow the project prototype:

- `macro`: title, abstract, introduction, conclusion style text
- `meso`: section lead discourse and organizing text
- `micro`: technical detail from methods, experiments, and detailed results

Saved span files:

- `data/spans/refined_qasper_slod_in_domain.csv`
- `data/spans/refined_qasper_slod_controlled.csv`
- `data/spans/qasper_prototype_mode.csv`

Each span file includes:

- `paper_id`
- `section_name`
- `label`
- `text`
- `token_count`

## Rebuild Prototype Spans

To rebuild the prototype-style QASPER spans:

```bash
python src/dataset.py --extract-qasper --config configs/prototype_qasper_weak_labels.yaml
python src/dataset.py --export-prototype-spans --input data/processed/refined_qasper_slod.csv --output data/spans/refined_qasper_slod_in_domain.csv
python src/dataset.py --export-prototype-spans --input data/processed/refined_qasper_slod_length_controlled.csv --output data/spans/refined_qasper_slod_controlled.csv
```

## Dataset Download

This prototype uses the QASPER scientific-paper dataset.

If the raw QASPER files are not already present, they are downloaded automatically when you run:

```bash
python src/dataset.py --extract-qasper --config configs/prototype_qasper_weak_labels.yaml
```

The raw download is stored under `data/raw/qasper/` during extraction, and the processed prototype spans are then written to `data/spans/` and `data/processed/`.

## Cached Embeddings

The prototype uses cached frozen embeddings. The expected owner-facing `embeddings/` layout is documented in:

- `embeddings/README.md`

The active cache directories used by the code are:

- `data/cache_refined_qasper_slod_scibert/`
- `data/cache_refined_qasper_slod_scibert_length_controlled/`
- `data/cache_refined_qasper_slod_minilm/`
- `data/cache_refined_qasper_slod_minilm_length_controlled/`

## Results

The repository includes saved metrics, predictions, and confusion matrices under:

- `results/prototype/in_domain/`
- `results/prototype/controlled/`

Each model folder contains:

- `metrics.json`
- `predictions.csv`
- `confusion_matrix.csv`
- `confusion_matrix.png`

Condition summaries:

- `results/prototype/in_domain/summary.json`
- `results/prototype/controlled/summary.json`

Included deliverable artifacts:

- code under `src/`
- data samples under `data/spans/`
- cached embeddings under `data/cache_refined_qasper_slod_*/`
- cached results under `results/prototype/`

## Project Structure

```text
.
|-- README.md
|-- requirements.txt
|-- configs/
|   |-- prototype_qasper_weak_labels.yaml
|   `-- prototype_scibert.yaml
|-- data/
|   |-- cache_refined_qasper_slod_minilm/
|   |-- cache_refined_qasper_slod_minilm_length_controlled/
|   |-- cache_refined_qasper_slod_scibert/
|   |-- cache_refined_qasper_slod_scibert_length_controlled/
|   |-- processed/
|   |   |-- refined_qasper_slod.csv
|   |   `-- refined_qasper_slod_length_controlled.csv
|   `-- spans/
|       |-- qasper_prototype_mode.csv
|       |-- qasper_prototype_mode_diagnostics.json
|       |-- refined_qasper_slod_controlled.csv
|       `-- refined_qasper_slod_in_domain.csv
|-- embeddings/
|   `-- README.md
|-- results/
|   `-- prototype/
|       |-- controlled/
|       `-- in_domain/
`-- src/
    |-- __init__.py
    |-- controls.py
    |-- dataset.py
    |-- embed.py
    |-- probe.py
    |-- utils.py
    |-- controls/
    |   |-- __init__.py
    |   `-- length_control.py
    |-- data/
    |   |-- diagnostics.py
    |   |-- preprocess.py
    |   |-- qasper.py
    |   `-- weak_labels.py
    |-- features/
    |   |-- embed_text.py
    |   `-- tfidf_features.py
    |-- models/
    |   |-- evaluate.py
    |   `-- train_probe.py
    `-- utils/
        `-- seed.py
```
