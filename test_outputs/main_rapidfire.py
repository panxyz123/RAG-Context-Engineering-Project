"""main_rapidfire.py — RAG pipeline using the RapidFire AI API.

Usage:
    python3 main_rapidfire.py --input test_questions.json --output predictions.json

Input schema  (list of dicts):
    [{"question_id": <int>, "question": <str>}, ...]

Output schema (list of dicts):
    [{"question_id":       <int>,
      "answer":            <str>,
      "retrieved_context": <str>,
      "sources":           [{"file": <str>, "lines": [<int>, <int>]}, ...]
     }, ...]

Design choices (same as notebook)
----------------------------------
Splitter   : rst_structural_splitter — RST heading + paragraph boundaries,
             produces start_line/end_line natively.
Embedding  : OpenAI-compatible endpoint via TRITON_API_KEY
Retrieval  : FAISS similarity search k=10, CrossEncoder reranker top_n=5
Generator  : Mistral via Triton endpoint, max 2048 tokens
Context    : Top-5 chunks joined by double newline, truncated to 2000 GPT-2 tokens

RapidFire drives retrieval and generation via experiment.run_evals().
Per-example outputs (answer, context, sources) are accumulated inside
postprocess_fn into a shared list, then written to the output file after
the run completes.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict

import tiktoken
from datasets import Dataset
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import TextSplitter

from rapidfireai.automl import (
    List,
    RFLangChainRagSpec,
    RFOpenAIAPIModelConfig,
    RFPromptManager,
    RFGridSearch,
)
from rapidfireai import Experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Constants (identical to notebook) ────────────────────────────────────────

SOURCEDOCS_PATH    = Path("~/Project1/sourcedocs").expanduser()
API_KEY_PATH       = Path("~/Project1/api-key.txt").expanduser()
TRITON_BASE_URL    = "https://tritonai-api.ucsd.edu"
EMBEDDING_MODEL    = "api-tgpt-embeddings"
GENERATOR_MODEL    = "api-mistral-small-3.2-2506"
MAX_CONTEXT_TOKENS = 2000
RETRIEVAL_K        = 10
RERANKER_TOP_N     = 5
BATCH_SIZE         = 32

SYSTEM_INSTRUCTIONS = (
    "You are a helpful QA assistant. "
    "Answer the question based solely on the provided context. "
    "Be concise and accurate."
)

listtype = list  # alias used throughout notebook


def load_api_key() -> str:
    return API_KEY_PATH.read_text(encoding="utf-8").splitlines()[0].strip()


# ── RST structural splitter ───────────────────────────────────────────────────

_RST_HEADING_RE = re.compile(r"^([=\-~^'`#*+])(\1*)$")


def rst_structural_splitter(
    documents: list[Document],
    min_tokens: int = 30,
    max_tokens: int = 300,
    tokenizer_encoding: str = "gpt2",
) -> list[Document]:
    enc = tiktoken.get_encoding(tokenizer_encoding)

    def token_len(text: str) -> int:
        return len(enc.encode(text))

    chunks: list[Document] = []

    for doc in documents:
        src   = doc.metadata.get("source", "")
        lines = doc.page_content.splitlines(keepends=True)

        section_starts = [0]
        for i, line in enumerate(lines):
            stripped = line.rstrip("\n")
            if _RST_HEADING_RE.match(stripped) and len(stripped) >= 2:
                title_idx = max(0, i - 1)
                if title_idx not in section_starts:
                    section_starts.append(title_idx)

        section_starts = sorted(set(section_starts))
        section_ranges = list(zip(section_starts, section_starts[1:] + [len(lines)]))

        for sec_start, sec_end in section_ranges:
            section_lines = lines[sec_start:sec_end]

            paragraphs: list[list[tuple[int, str]]] = []
            current:    list[tuple[int, str]]        = []
            for offset, line in enumerate(section_lines):
                lineno = sec_start + offset + 1
                if line == "\n" and current:
                    paragraphs.append(current)
                    current = []
                else:
                    current.append((lineno, line))
            if current:
                paragraphs.append(current)

            def _flush(
                buf: list[tuple[int, str]],
                _src: str,
                chunk_type: str = "paragraph",
            ) -> None:
                if not buf:
                    return
                text = "".join(t for _, t in buf).strip()
                if not text:
                    return
                chunks.append(Document(
                    page_content=text,
                    metadata={
                        "source":     _src,
                        "start_line": buf[0][0],
                        "end_line":   buf[-1][0],
                        "chunk_type": chunk_type,
                    },
                ))

            buffer: list[tuple[int, str]] = []

            for para in paragraphs:
                para_text   = "".join(t for _, t in para).strip()
                para_tokens = token_len(para_text)

                if para_tokens > max_tokens:
                    _flush(buffer, src)
                    buffer = []
                    sentences = re.split(r"(?<=[.!?])\s+", para_text)
                    sent_buf:  list[tuple[int, str]] = []
                    sent_toks  = 0
                    base_line  = para[0][0]
                    for sent in sentences:
                        t = token_len(sent)
                        if sent_toks + t > max_tokens and sent_buf:
                            _flush(sent_buf, src, chunk_type="overflow")
                            sent_buf  = []
                            sent_toks = 0
                        sent_buf.append((base_line, sent + " "))
                        sent_toks += t
                    _flush(sent_buf, src, chunk_type="overflow")

                elif token_len("".join(t for _, t in buffer)) + para_tokens < min_tokens:
                    buffer.extend(para)

                else:
                    _flush(buffer, src)
                    buffer = para[:]

            _flush(buffer, src)

    return chunks


class StructuralRSTSplitter(TextSplitter):
    """TextSplitter wrapper so RFLangChainRagSpec accepts the structural splitter."""

    def __init__(self, min_tokens: int = 30, max_tokens: int = 300):
        super().__init__(chunk_size=max_tokens, chunk_overlap=0)
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def create_documents(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
    ) -> list[Document]:
        if metadatas is None:
            metadatas = [{} for _ in texts]
        docs = [Document(page_content=t, metadata=m)
                for t, m in zip(texts, metadatas)]
        return rst_structural_splitter(docs, self.min_tokens, self.max_tokens)

    def split_documents(self, documents: list[Document]) -> list[Document]:
        return rst_structural_splitter(documents, self.min_tokens, self.max_tokens)

    def split_text(self, text: str) -> list[str]:
        doc = Document(page_content=text, metadata={"source": ""})
        return [c.page_content for c in rst_structural_splitter(
            [doc], self.min_tokens, self.max_tokens)]


# ── Context serialisation ─────────────────────────────────────────────────────

def custom_template(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# ── Output accumulator ────────────────────────────────────────────────────────
# postprocess_fn appends to this list; run() reads it after run_evals() returns.

_output_rows: list[dict[str, Any]] = []


def _docs_to_sources(spans: list[tuple[str, int, int]]) -> list[dict[str, Any]]:
    """Convert (file, start_line, end_line) tuples to required sources schema."""
    seen  = set()
    out   = []
    for fname, start, end in spans:
        key = (fname, start, end)
        if key not in seen:
            seen.add(key)
            out.append({"file": fname, "lines": [start, end]})
    return out


# ── RapidFire preprocess / postprocess ───────────────────────────────────────

def preprocess_fn(
    batch: Dict[str, listtype],
    rag: RFLangChainRagSpec,
    prompt_manager: RFPromptManager,
) -> Dict[str, listtype]:
    """Retrieve context and build generator prompts."""
    enc = tiktoken.get_encoding("gpt2")

    def truncate(text: str) -> str:
        tokens = enc.encode(text)
        return enc.decode(tokens[:MAX_CONTEXT_TOKENS]) if len(tokens) > MAX_CONTEXT_TOKENS else text

    all_context = rag.get_context(batch_queries=batch["query"], serialize=False)

    retrieved_spans = [
        [
            (
                Path(doc.metadata.get("source", "")).name,
                int(doc.metadata.get("start_line", 0)),
                int(doc.metadata.get("end_line",   0)),
            )
            for doc in docs
        ]
        for docs in all_context
    ]

    serialized_context = [custom_template(docs) for docs in all_context]

    batch["query_id"] = [str(qid) for qid in batch["query_id"]]

    prompts = []
    for question, context in zip(batch["query"], serialized_context):
        user_content = truncate(
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
        prompts.append([
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user",   "content": user_content},
        ])

    return {
        "prompts":            prompts,
        "retrieved_spans":    retrieved_spans,
        "serialized_context": serialized_context,
        **batch,
    }


def postprocess_fn(batch: Dict[str, listtype]) -> Dict[str, listtype]:
    """Accumulate per-example outputs into _output_rows for final JSON writing."""
    for qid, answer, context, spans in zip(
        batch["query_id"],
        batch["generated_text"],
        batch["serialized_context"],
        batch["retrieved_spans"],
    ):
        _output_rows.append({
            "question_id":       int(qid),
            "answer":            answer,
            "retrieved_context": context,
            "sources":           _docs_to_sources(spans),
        })
    return batch


# ── Dummy metrics (required by run_evals, no eval data available at inference) ─

def compute_metrics_fn(batch: Dict[str, listtype]) -> Dict[str, Dict[str, Any]]:
    return {"Total": {"value": len(batch["query"])}}


def accumulate_metrics_fn(
    aggregated_metrics: Dict[str, listtype],
) -> Dict[str, Dict[str, Any]]:
    total = sum(m["value"] for m in aggregated_metrics["Total"])
    return {"Total": {"value": total}}


# ── Main ──────────────────────────────────────────────────────────────────────

def run(input_path: str, output_path: str) -> None:
    global _output_rows
    _output_rows = []

    with open(input_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    log.info("Loaded %d questions from %s", len(questions), input_path)

    api_key = load_api_key()

    # Build dataset — no reference_answer needed for inference
    queries = Dataset.from_dict({
        "query":    [str(q["question"])    for q in questions],
        "query_id": [str(q["question_id"]) for q in questions],
    })

    # RAG config — structural splitter only
    rag = RFLangChainRagSpec(
        document_loader=DirectoryLoader(
            path=str(SOURCEDOCS_PATH),
            glob="*.rst",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        ),
        text_splitter=StructuralRSTSplitter(min_tokens=30, max_tokens=300),
        embedding_cfg=[{
            "class": OpenAIEmbeddings,
            "model": EMBEDDING_MODEL,
            "api_key": api_key,
            "base_url": TRITON_BASE_URL,
            "check_embedding_ctx_length": False,
        }],
        vector_store_cfg={"type": "faiss", "batch_size": BATCH_SIZE},
        search_cfg=[{"type": "similarity", "k": RETRIEVAL_K}],
        reranker_cfg={
            "class": CrossEncoderReranker,
            "model_name": "cross-encoder/ms-marco-MiniLM-L6-v2",
            "model_kwargs": {"device": "cpu"},
            "top_n": RERANKER_TOP_N,
        },
        enable_gpu_search=False,
    )

    openai_config = RFOpenAIAPIModelConfig(
        client_config={
            "api_key":     api_key,
            "base_url":    TRITON_BASE_URL,
            "max_retries": 2,
        },
        model_config={
            "model":                 GENERATOR_MODEL,
            "max_completion_tokens": 2048,
            "temperature":           0.0,
        },
        rpm_limit=120,
        tpm_limit=1_000_000,
        rag=rag,
        prompt_manager=None,
    )

    config_set = [{
        "openai_config":          openai_config,
        "batch_size":             BATCH_SIZE,
        "preprocess_fn":          preprocess_fn,
        "postprocess_fn":         postprocess_fn,
        "compute_metrics_fn":     compute_metrics_fn,
        "accumulate_metrics_fn":  accumulate_metrics_fn,
        "online_strategy_kwargs": {
            "strategy_name":    "normal",
            "confidence_level": 0.95,
            "use_fpc":          True,
        },
    }]

    config_group = RFGridSearch(config_set)

    experiment = Experiment(experiment_name="inference_run", mode="evals")

    log.info("Starting RapidFire inference run ...")
    experiment.run_evals(
        config_group=config_group,
        dataset=queries,
        num_shards=4,
        seed=42,
    )
    experiment.end()

    # Sort by question_id to match input order
    _output_rows.sort(key=lambda r: r["question_id"])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(_output_rows, f, indent=2, ensure_ascii=False)
    log.info("Wrote %d results to %s", len(_output_rows), output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RapidFire AI RST Q&A pipeline")
    parser.add_argument("--input",  required=True, help="Input JSON (list of {question_id, question})")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()
    run(args.input, args.output)
