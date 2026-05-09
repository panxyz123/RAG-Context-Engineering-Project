# RapidFireAI RAG & Context Engineering

**Team 56**

A Retrieval-Augmented Generation (RAG) system built over the RapidFireAI documentation corpus (`.rst` files). The system retrieves relevant documentation spans for a given question, assembles a contiguous context, and generates a grounded answer via a large language model.

---

## What This System Does

Given a set of questions about RapidFireAI, the pipeline:
1. Chunks and indexes the `.rst` corpus into a FAISS vector store
2. Retrieves the most relevant chunks per query using dense similarity search
3. Reranks candidates with a cross-encoder and applies file-level aggregation to select the most relevant source document(s)
4. Merges nearby retrieved spans into contiguous regions and feeds the extracted text to an LLM
5. Outputs structured predictions with answers, retrieved context, and source line references

---

## Experiments

### Framework

Experiments were conducted using the **RapidFire AI `RFGridSearch`** framework with an interactive run controller. Each configuration was evaluated on a 22-question golden QA dataset covering 18 unique source files and 5 question categories. Metrics follow `project1_eval.py`: overlap-based Precision@5, Recall@5, F1@5 for retrieval, and Completeness (0–5), Correctness (0/1), Faithfulness (0/1) from an LLM judge for generation. The final Score is the normalized average of all six metrics.

### Configurations Tested

| ID | Chunking | Search | Top-K | Reranker | Post-processing | LLM | Remarks |
|----|----------|--------|-------|----------|-----------------|-----|---------|
| 1  | Recursive (128) | Similarity (K=10) | 5 | MiniLM | None | Mistral-small | Baseline |
| 2  | Recursive (128) | MMR (K=10) | 5 | MiniLM | None | Mistral-small | MMR search |
| 3  | Recursive (128) | Similarity (K=20) | 5 | BGE | None | Mistral-small | More retrieved chunks |
| 4  | Recursive (128) | Similarity (K=20) | 3 | BGE | None | Mistral-small | Top-5 → Top-3 |
| 5  | Recursive (256/64) | Similarity (K=20) | 3 | BGE | None | Mistral-small | Chunk size 128 → 256 |
| 6  | Recursive (256/128) | Similarity (K=20) | 3 | BGE | None | Mistral-small | Overlap 64 → 128 |
| 7  | Structural (256) | Similarity (K=20) | 3 | BGE | None | Mistral-small | Custom structural splitter |
| 8  | Recursive (256/128) | Similarity (K=20) | 3 | BGE | Yes | Mistral-small | Post-processing introduced |
| 9  | Recursive (256/128) | Similarity (K=20) | 3 | BGE | Yes | Claude-Sonnet | LLM upgrade |
| 10 | Structural (512) | Similarity (K=20) | 3 | BGE | Yes | Claude-Sonnet | **Final config** |

### Results

| ID | Prec. | Rec. | F1 | Comp. | Corr. | Faith. | Score | Remarks |
|----|-------|------|----|-------|-------|--------|-------|---------|
| 1  | 0.46 | 0.93 | 0.59 | 3.33 | 0.76 | 0.91 | 0.718 | Baseline |
| 2  | 0.42 | 0.88 | 0.54 | 3.18 | 0.76 | 0.93 | 0.694 | MMR underperforms |
| 3  | 0.45 | 0.98 | 0.59 | 3.62 | 0.82 | 0.93 | 0.749 | BGE improves recall |
| 4  | 0.51 | 0.89 | 0.62 | 3.69 | 0.87 | 0.98 | 0.766 | Top-3 improves precision |
| 5  | 0.52 | 0.92 | 0.63 | 3.60 | 0.87 | 0.96 | 0.769 | Larger chunk helps |
| 6  | 0.66 | 0.93 | 0.74 | 3.45 | 0.87 | 0.96 | 0.809 | More overlap boosts precision |
| 7  | 0.45 | 0.96 | 0.59 | 3.96 | 0.89 | 0.93 | 0.769 | Structural splitter improves completeness |
| 8  | 0.81 | 0.91 | 0.83 | 3.47 | 0.91 | 1.00 | 0.859 | Post-processing: +0.1 score |
| 9  | 0.83 | 0.84 | 0.81 | 4.07 | 0.91 | 1.00 | 0.868 | Claude improves completeness |
| 10 | 0.87 | 0.91 | 0.87 | 3.64 | 0.95 | 0.93 | **0.878** | Best overall |

### Key Findings

- **MMR hurts precision** — MMR penalises adjacent chunks from the same ground-truth document, which is exactly what the overlap-based metric rewards. Similarity search consistently outperforms MMR in this setup.
- **BGE reranker is critical** — Switching from MiniLM to `bge-reranker-v2-m3` markedly improved recall and generation quality.
- **Post-processing is the single biggest gain** — File-level rank-weighted aggregation (selecting the most relevant document before chunk-level ranking) raised precision from ~0.5 to 0.81+, improving the overall score by ~0.1.
- **Structural splitter + large chunks** — Paragraph-aligned chunking (ID 10) produces more semantically complete contexts, leading to the best retrieval F1 and correctness scores.
- **LLM choice trades completeness vs faithfulness** — Claude-Sonnet generates more complete answers but may reduce faithfulness; prompt constraints mitigate this.

---

## Final Pipeline (ID 10)

The final `main_strc.py` implements the following four-stage pipeline:

**1. Indexing**
The corpus directory of `.rst` files is loaded via `DirectoryLoader`. Each document is chunked with the custom RST structural splitter (`min_tokens=256`, `max_tokens=512`, one-paragraph overlap): sections are identified by RST heading underlines, paragraphs split on blank lines, small paragraphs merged, and oversized ones split line-by-line. Chunks are embedded with `api-tgpt-embeddings` and indexed into a FAISS flat vector store.

**2. Retrieval and Reranking**
Top-20 chunks are retrieved by cosine similarity. All candidates are scored by `BAAI/bge-reranker-v2-m3`. Scores are aggregated per file using a rank-weighted sum (`score × 1/(rank+1)`); the highest-scoring file is selected along with any file within 80% of its score. The top-3 chunks from the selected files are retained. Their line spans are merged if within 80 lines of each other, and the corresponding source file text is extracted as context.

**3. Generation**
The merged context is passed to the user-specified LLM via the Triton gateway. The prompt presents the question first, followed by the context truncated to fit within a 2000-token budget (system prompt and question overhead reserved), then repeats the question to prevent truncation from hiding it.

**4. Output**
Results are sorted by `question_id` and written to a JSON file. Each entry contains:
```json
{
  "question_id": 1,
  "answer": "...",
  "retrieved_context": "...",
  "sources": [
    {"file": "installation.rst", "lines": [42, 78]}
  ]
}
```

---

## Running the Pipeline

### Requirements

```bash
pip install langchain langchain-community langchain-openai langchain-text-splitters \
            langchain-huggingface faiss-cpu tiktoken sentence-transformers openai
```

### Usage

```bash
python main_strc.py \
    --input   questions.json \
    --output  predictions.json \
    --corpus-dir  path/to/sourcedocs/ \
    --apikey-txt  path/to/api-key.txt \
    --generation-model  mistral-large-3-675b-instruct
```

Use `--mode rapidfire` to run via the RapidFire API instead of the standalone path:

```bash
python main_strc.py \
    --input   questions.json \
    --output  predictions.json \
    --corpus-dir  path/to/sourcedocs/ \
    --apikey-txt  path/to/api-key.txt \
    --generation-model  mistral-large-3-675b-instruct \
    --mode rapidfire
```

The token-based splitter variant is available as `main_token.py` with identical arguments.

### Input Format

```json
[
  {"question_id": 1, "question": "How do I configure batch size in RapidFireAI?"},
  {"question_id": 2, "question": "..."}
]
```

### Output Format

```json
[
  {
    "question_id": 1,
    "answer": "The batch size is configured via the batch_size parameter...",
    "retrieved_context": "The full concatenated context text passed to the LLM...",
    "sources": [
      {"file": "configuration.rst", "lines": [45, 89]}
    ]
  }
]
```

### Evaluation

```bash
python evaluation.py \
    --output        predictions.json \
    --ground-truth  golden_qa_pairs.json \
    --apikey-txt    path/to/api-key.txt \
    --judge-model   api-gpt-oss-120b \
    --metrics-output  metrics.json
```

Prints all metrics to stdout and writes a `metrics.json` with per-example and summary scores.