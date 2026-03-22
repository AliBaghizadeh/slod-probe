# Embeddings Cache Map

The owner prototype expects cached embeddings under `embeddings/`.

This cleaned prototype keeps the actual cache files under `data/` and uses them directly from `src/probe.py`.

Prototype cache mapping:

- `data/cache_refined_qasper_slod_scibert/` -> SciBERT `in_domain`
- `data/cache_refined_qasper_slod_scibert_length_controlled/` -> SciBERT `controlled`
- `data/cache_refined_qasper_slod_minilm/` -> MiniLM `in_domain`
- `data/cache_refined_qasper_slod_minilm_length_controlled/` -> MiniLM `controlled`

Each cache directory contains:

- `embeddings_*.npy`
- `metadata_*.csv`
- `embedding_log_*.json`

No fine-tuning is performed. The probe uses these frozen cached embeddings directly.
