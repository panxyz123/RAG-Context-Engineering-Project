"""main_structural_rapidfire.py — RAG pipeline with StructuralRSTSplitter via RapidFire API.

Usage:
    python main_structural_rapidfire.py \
        --input questions.json \
        --output predictions.json \
        --corpus-dir sourcedocs/ \
        --apikey-txt api-key.txt \
        --generation-model mistral-large-3-675b-instruct
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List as listtype

import tiktoken
from datasets import Dataset
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings

from rapidfireai.automl import (
    List,
    RFLangChainRagSpec,
    RFOpenAIAPIModelConfig,
    RFPromptManager,
    RFGridSearch,
)
from rapidfireai import Experiment

from rag_shared import (
    StructuralRSTSplitter,
    SYSTEM_INSTRUCTIONS,
    MAX_TOKENS,
    MAX_COMPLETION,
    FILE_SCORE_THRESH,
    RETRIEVAL_K,
    load_api_key,
    merge_spans_and_text,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TRITON_BASE_URL = "https://tritonai-api.ucsd.edu"
EMBEDDING_MODEL = "api-tgpt-embeddings"
BATCH_SIZE      = 32

_output_rows: list[dict[str, Any]] = []
_corpus_path: Path                 = Path(".")


def preprocess_fn(
    batch: Dict[str, listtype],
    rag: RFLangChainRagSpec,
    prompt_manager: RFPromptManager,
) -> Dict[str, listtype]:

    if not hasattr(preprocess_fn, "_reranker"):
        preprocess_fn._reranker = HuggingFaceCrossEncoder(
            model_name="BAAI/bge-reranker-v2-m3",
            model_kwargs={"device": "cpu"},
        )
    reranker = preprocess_fn._reranker

    all_context_raw = rag.get_context(batch_queries=batch["query"], serialize=False)

    def _select(query: str, docs: list) -> list:
        if not docs:
            return docs
        pairs  = [(query, doc.page_content) for doc in docs]
        scores = reranker.score(pairs)
        scored = list(zip(scores, docs))
        file_scores: dict[str, float] = defaultdict(float)
        for rank, (score, doc) in enumerate(
            sorted(scored, key=lambda x: x[0], reverse=True)
        ):
            fname = Path(doc.metadata.get("source", "")).name
            file_scores[fname] += score * (1.0 / (rank + 1))
        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        best_score   = sorted_files[0][1]
        selected     = {f for f, s in sorted_files[:2] if s >= best_score * FILE_SCORE_THRESH}
        filtered     = [(s, d) for s, d in scored
                        if Path(d.metadata.get("source", "")).name in selected]
        filtered.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in filtered[:3]]

    all_context = [_select(q, docs) for q, docs in zip(batch["query"], all_context_raw)]

    retrieved_spans = [
        [
            (
                Path(doc.metadata.get("source", "")).name,
                doc.metadata.get("start_line", 0),
                doc.metadata.get("end_line",   -1),
                doc.page_content,
                doc.metadata.get("span_coord", "line"),
            )
            for doc in docs
        ]
        for docs in all_context
    ]

    serialized_context = []
    for spans_raw in retrieved_spans:
        _, merged_text = merge_spans_and_text(spans_raw, _corpus_path)
        serialized_context.append(merged_text)

    batch["query_id"] = [str(qid) for qid in batch["query_id"]]

    enc = tiktoken.get_encoding("gpt2")
    prompts = []
    for question, context in zip(batch["query"], serialized_context):
        overhead = len(enc.encode(
            SYSTEM_INSTRUCTIONS + "\n\n" +
            f"Question: {question}\n\nFind the answer in the Context:\n\n"
            f"Now answer the question:"
        ))
        budget        = max(MAX_TOKENS - overhead, 100)
        trunc_context = enc.decode(enc.encode(context)[:budget])
        user_content  = (
            f"Question: {question}\n\n"
            f"Find the answer in the Context: {trunc_context}\n\n"
            f"Now answer the question correctly, completely and faithfully '{question}':"
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
    for qid, answer, context, spans_raw in zip(
        batch["query_id"],
        batch["generated_text"],
        batch["serialized_context"],
        batch["retrieved_spans"],
    ):
        sources, _ = merge_spans_and_text(spans_raw, _corpus_path)
        _output_rows.append({
            "question_id":       int(qid),
            "answer":            answer,
            "retrieved_context": context,
            "sources":           sources,
        })
    return batch


def compute_metrics_fn(batch: Dict[str, listtype]) -> Dict[str, Dict[str, Any]]:
    return {"Total": {"value": len(batch["query"])}}


def accumulate_metrics_fn(aggregated_metrics: Dict[str, listtype]) -> Dict[str, Dict[str, Any]]:
    return {"Total": {"value": sum(m["value"] for m in aggregated_metrics["Total"])}}


def run(input_path: str, output_path: str, corpus_dir: str,
        apikey_txt: str, generation_model: str) -> None:
    global _output_rows, _corpus_path
    _output_rows = []
    _corpus_path = Path(corpus_dir).expanduser()

    with open(input_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    log.info("Loaded %d questions", len(questions))

    api_key = load_api_key(apikey_txt)

    queries = Dataset.from_dict({
        "query":    [str(q["question"])    for q in questions],
        "query_id": [str(q["question_id"]) for q in questions],
    })

    rag = RFLangChainRagSpec(
        document_loader=DirectoryLoader(
            path=str(_corpus_path),
            glob="*.rst",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        ),
        text_splitter=StructuralRSTSplitter(min_tokens=256, max_tokens=512),
        embedding_cfg=List([{
            "class":                      OpenAIEmbeddings,
            "model":                      EMBEDDING_MODEL,
            "api_key":                    api_key,
            "base_url":                   TRITON_BASE_URL,
            "check_embedding_ctx_length": False,
        }]),
        vector_store_cfg={"type": "faiss", "batch_size": BATCH_SIZE},
        search_cfg=List([{"type": "similarity", "k": RETRIEVAL_K}]),
        reranker_cfg=None,
        enable_gpu_search=False,
    )

    openai_config = RFOpenAIAPIModelConfig(
        client_config={"api_key": api_key, "base_url": TRITON_BASE_URL, "max_retries": 2},
        model_config={"model": generation_model, "max_completion_tokens": MAX_COMPLETION},
        rpm_limit=120,
        tpm_limit=1_000_000,
        rag=rag,
        prompt_manager=None,
    )

    config_set = [{"openai_config": openai_config, "batch_size": BATCH_SIZE,
                   "preprocess_fn": preprocess_fn, "postprocess_fn": postprocess_fn,
                   "compute_metrics_fn": compute_metrics_fn,
                   "accumulate_metrics_fn": accumulate_metrics_fn,
                   "online_strategy_kwargs": {"strategy_name": "normal",
                                              "confidence_level": 0.95, "use_fpc": True}}]

    experiment = Experiment(experiment_name="inference_structural_rf", mode="evals")
    log.info("Starting inference (structural splitter, RapidFire)...")
    experiment.run_evals(config_group=RFGridSearch(config_set), dataset=queries,
                         num_shards=4, seed=42)
    experiment.end()

    _output_rows.sort(key=lambda r: r["question_id"])
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(_output_rows, f, indent=2, ensure_ascii=False)
    log.info("Wrote %d results to %s", len(_output_rows), output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG pipeline — structural splitter (RapidFire)")
    parser.add_argument("--input",            required=True)
    parser.add_argument("--output",           required=True)
    parser.add_argument("--corpus-dir",       required=True)
    parser.add_argument("--apikey-txt",       required=True)
    parser.add_argument("--generation-model", required=True)
    args = parser.parse_args()
    run(args.input, args.output, args.corpus_dir, args.apikey_txt, args.generation_model)
