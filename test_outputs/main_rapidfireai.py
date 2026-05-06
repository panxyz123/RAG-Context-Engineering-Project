"""main.py — RAG inference pipeline using RapidFire AI API.

Best config from notebook experiments:
    Splitter  : LineNumberSplitter, chunk_size=128, overlap=32
    Retrieval : FAISS similarity, k=20
    Reranker  : BAAI/bge-reranker-v2-m3, top_n=5
    Generator : api-mistral-small-3.2-2506

Usage:
    python3 main.py --input test_questions.json --output predictions.json

Input schema  (list of dicts):
    [{"question_id": <int>, "question": <str>}, ...]

Output schema (list of dicts):
    [{"question_id":       <int>,
      "answer":            <str>,
      "retrieved_context": <str>,
      "sources":           [{"file": <str>, "lines": [<int>, <int>]}, ...]
     }, ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List as listtype

import tiktoken
from datasets import Dataset
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

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

# ── Constants ─────────────────────────────────────────────────────────────────

SOURCEDOCS_PATH = Path(__file__).parent / "sourcedocs"
API_KEY_PATH      = Path("~/api-key.txt").expanduser()
TRITON_API_KEY = os.environ.get("TRITON_API_KEY") or Path("~/Project1/api-key.txt").expanduser().read_text(encoding="utf-8").splitlines()[0].strip()
TRITON_BASE_URL  = "https://tritonai-api.ucsd.edu"
EMBEDDING_MODEL  = "api-tgpt-embeddings"
GENERATOR_MODEL  = "api-mistral-small-3.2-2506"
BATCH_SIZE       = 32
MAX_TOKENS       = 2000

SYSTEM_INSTRUCTIONS = (
    "You are a technical QA assistant for RapidFire AI documentation. "
    "Answer the question using ONLY the provided context. "
    "Your answer will be scored on three criteria: "
    "Correctness (matches the reference answer), "
    "Faithfulness (no information beyond the context), and "
    "Completeness (covers all key points in the context relevant to the question). "
    "You must provide the answer that best meets all three criteria."
)

# ── LineNumberSplitter ────────────────────────────────────────────────────────

class LineNumberSplitter(TextSplitter):
    """Wraps RecursiveCharacterTextSplitter; tags chunks with span_coord="char".

    Char offsets (start_index / end_index) are left in metadata as-is.
    _char_offsets_to_lines converts them to line numbers when building output.
    """
    def __init__(self, inner_splitter: TextSplitter):
        super().__init__(chunk_size=inner_splitter._chunk_size,
                         chunk_overlap=inner_splitter._chunk_overlap)
        self._inner = inner_splitter

    def create_documents(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
    ) -> list[Document]:
        docs = self._inner.create_documents(texts, metadatas=metadatas)
        for doc in docs:
            doc.metadata["span_coord"] = "char"
        return docs

    def split_documents(self, documents: list[Document]) -> list[Document]:
        chunks = self._inner.split_documents(documents)
        for doc in chunks:
            doc.metadata["span_coord"] = "char"
        return chunks

    def split_text(self, text: str) -> list[str]:
        return self._inner.split_text(text)


# ── Char offset → line number conversion ─────────────────────────────────────

_file_line_cache: dict[str, str] = {}


def _char_offsets_to_lines(fname: str, start_char: int, end_char: int) -> tuple[int, int]:
    """Convert char offsets to 1-based inclusive line numbers.

    Looks up the source file under SOURCEDOCS_PATH. Returns (1, 1) if
    the file cannot be found so output degrades gracefully rather than crashing.
    """
    if fname not in _file_line_cache:
        candidates = list(SOURCEDOCS_PATH.rglob(fname))
        if not candidates:
            return 1, 1
        _file_line_cache[fname] = candidates[0].read_text(encoding="utf-8")
    text     = _file_line_cache[fname]
    end_char = end_char if end_char >= 0 else len(text)
    start_line = text[:start_char].count("\n") + 1
    end_line   = text[:end_char].count("\n")   + 1
    return start_line, end_line


# ── Output accumulator ────────────────────────────────────────────────────────

_output_rows: list[dict[str, Any]] = []


def _spans_to_sources(
    raw_spans: list[tuple[str, int, int, str]],
) -> list[dict[str, Any]]:
    """Convert raw (file, start, end, span_coord) tuples to output sources schema.

    Converts char offsets to line numbers when span_coord="char".
    Deduplicates by (file, start_line, end_line).
    """
    seen = set()
    out  = []
    for fname, start, end, coord in raw_spans:
        if coord == "char":
            start_line, end_line = _char_offsets_to_lines(fname, start, end)
        else:
            start_line, end_line = int(start), int(end)
        key = (fname, start_line, end_line)
        if key not in seen:
            seen.add(key)
            out.append({"file": fname, "lines": [start_line, end_line]})
    return out


# ── RapidFire preprocess / postprocess ───────────────────────────────────────

def preprocess_fn(
    batch: Dict[str, listtype],
    rag: RFLangChainRagSpec,
    prompt_manager: RFPromptManager,
) -> Dict[str, listtype]:
    enc = tiktoken.get_encoding("gpt2")

    def truncate(text: str) -> str:
        tokens = enc.encode(text)
        return enc.decode(tokens[:MAX_TOKENS]) if len(tokens) > MAX_TOKENS else text

    all_context = rag.get_context(batch_queries=batch["query"], serialize=False)

    retrieved_spans = [
        [
            (
                Path(doc.metadata.get("source", "")).name,
                doc.metadata.get("start_index", doc.metadata.get("start_line", 0)),
                doc.metadata.get("end_index",   doc.metadata.get("end_line",   -1)),
                doc.metadata.get("span_coord", "char"),
            )
            for doc in docs
        ]
        for docs in all_context
    ]

    serialized_context = [
        "\n\n".join(doc.page_content for doc in docs)
        for docs in all_context
    ]

    batch["query_id"] = [str(qid) for qid in batch["query_id"]]

    prompts = []
    for question, context in zip(batch["query"], serialized_context):
        user_content = truncate(
            f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"
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
            "sources":           _spans_to_sources(spans),
        })
    return batch


# ── Dummy metrics (no ground truth at inference time) ────────────────────────

def compute_metrics_fn(batch: Dict[str, listtype]) -> Dict[str, Dict[str, Any]]:
    return {"Total": {"value": len(batch["query"])}}


def accumulate_metrics_fn(
    aggregated_metrics: Dict[str, listtype],
) -> Dict[str, Dict[str, Any]]:
    return {"Total": {"value": sum(m["value"] for m in aggregated_metrics["Total"])}}


# ── Main ──────────────────────────────────────────────────────────────────────

def run(input_path: str, output_path: str) -> None:
    global _output_rows
    _output_rows = []

    with open(input_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    log.info("Loaded %d questions from %s", len(questions), input_path)

    queries = Dataset.from_dict({
        "query":    [str(q["question"])    for q in questions],
        "query_id": [str(q["question_id"]) for q in questions],
    })

    rag = RFLangChainRagSpec(
        document_loader=DirectoryLoader(
            path=str(SOURCEDOCS_PATH),
            glob="*.rst",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            sample_seed=1337,
        ),
        text_splitter=LineNumberSplitter(
            RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="gpt2",
                chunk_size=128,
                chunk_overlap=32,
                add_start_index=True,
            )
        ),
        embedding_cfg=List([{
            "class":                      OpenAIEmbeddings,
            "model":                      EMBEDDING_MODEL,
            "api_key":                    TRITON_API_KEY,
            "base_url":                   TRITON_BASE_URL,
            "check_embedding_ctx_length": False,
        }]),
        vector_store_cfg={"type": "faiss", "batch_size": BATCH_SIZE},
        search_cfg=List([{"type": "similarity", "k": 20}]),
        reranker_cfg = CrossEncoderReranker(
        model=HuggingFaceCrossEncoder(
            model_name="BAAI/bge-reranker-v2-m3",
            model_kwargs={"device": "cpu"},
        ),
        top_n=5,
        )
    )
        # reranker_cfg={
        #     "class": CrossEncoderReranker,
        #     "model": HuggingFaceCrossEncoder(
        #         model="BAAI/bge-reranker-v2-m3",
        #         device="cpu",
        #     ),
        #     "top_n": 5,
        # },
        # enable_gpu_search=False,
    # )

    openai_config = RFOpenAIAPIModelConfig(
        client_config={
            "api_key":     TRITON_API_KEY,
            "base_url":    TRITON_BASE_URL,
            "max_retries": 2,
        },
        model_config={
            "model":                 GENERATOR_MODEL,
            "max_completion_tokens": 2048,
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

    experiment = Experiment(experiment_name="inference_run", mode="evals")

    log.info("Starting inference ...")
    experiment.run_evals(
        config_group=RFGridSearch(config_set),
        dataset=queries,
        num_shards=4,
        seed=42,
    )
    experiment.end()

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
