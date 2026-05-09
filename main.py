"""main_strc.py — Complete RAG pipeline with StructuralRSTSplitter (min=256, max=512).

All logic mirrors RAG_Experiments_merge.ipynb exactly:
  - Embedding   : api-tgpt-embeddings via Triton
  - Retrieval   : FAISS similarity k=20
  - Reranker    : BAAI/bge-reranker-v2-m3, file-level weighted scoring,
                  top-2 files within 0.8 of best score, top-3 chunks
  - Span merge  : gap=80 lines, merged text fed to LLM
  - Prompt      : question → context (truncated to token budget) → question repeat

Usage:
    python main_strc.py \
        --input  questions.json \
        --output predictions.json \
        --corpus-dir sourcedocs/ \
        --apikey-txt api-key.txt \
        --generation-model mistral-large-3-675b-instruct
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import tiktoken
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import TextSplitter
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
TRITON_BASE_URL   = "https://tritonai-api.ucsd.edu"
EMBEDDING_MODEL   = "api-tgpt-embeddings"
RETRIEVAL_K       = 20
RERANKER_TOP_N    = 3
FILE_SCORE_THRESH = 0.8
MERGE_GAP         = 30
MAX_TOKENS        = 2000
MAX_COMPLETION    = 3000
BATCH_SIZE        = 32

SYSTEM_INSTRUCTIONS = (
    "You are a technical QA assistant for RapidFire AI documentation. "
    "Answer the question using ONLY the provided context. "
    "Your answer will be scored on three criteria: "
    "Correctness (matches the reference answer), "
    "Faithfulness (no information beyond the context), and "
    "Completeness (covers all key points in the context relevant to the question). "
    "You must provide the answer that best meets all three criteria."
)

# ── API key ───────────────────────────────────────────────────────────────────

def load_api_key(apikey_txt: str) -> str:
    return Path(apikey_txt).expanduser().read_text(encoding="utf-8").splitlines()[0].strip()

# ── RST structural splitter ───────────────────────────────────────────────────

_RST_HEADING_RE = re.compile(r"^([=\-~^'`#*+])(\1*)$")


def rst_structural_splitter(
    documents: list[Document],
    min_tokens: int = 256,
    max_tokens: int = 512,
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
                if line == "" and current:
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
                        "span_coord": "line",
                    },
                ))

            buffer:    list[tuple[int, str]] = []
            prev_para: list[tuple[int, str]] = []

            for para in paragraphs:
                para_text   = "".join(t for _, t in para).strip()
                para_tokens = token_len(para_text)

                if para_tokens > max_tokens:
                    _flush(buffer, src)
                    buffer    = []
                    prev_para = []
                    line_buf:  list[tuple[int, str]] = []
                    line_toks  = 0
                    for lineno, line_text in para:
                        t = token_len(line_text)
                        if line_toks + t > max_tokens and line_buf:
                            _flush(line_buf, src, chunk_type="overflow")
                            line_buf  = []
                            line_toks = 0
                        line_buf.append((lineno, line_text))
                        line_toks += t
                    _flush(line_buf, src, chunk_type="overflow")

                elif token_len("".join(t for _, t in buffer)) + para_tokens < min_tokens:
                    buffer.extend(para)
                    prev_para = para

                else:
                    _flush(buffer, src)
                    buffer    = prev_para[:] + para[:]
                    prev_para = para

            _flush(buffer, src)

    return chunks


class StructuralRSTSplitter(TextSplitter):
    """Wraps rst_structural_splitter; span_coord='line'."""

    def __init__(self, min_tokens: int = 256, max_tokens: int = 512):
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
        docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
        return rst_structural_splitter(docs, self.min_tokens, self.max_tokens)

    def split_documents(self, documents: list[Document]) -> list[Document]:
        return rst_structural_splitter(documents, self.min_tokens, self.max_tokens)

    def split_text(self, text: str) -> list[str]:
        doc = Document(page_content=text, metadata={"source": ""})
        return [c.page_content for c in rst_structural_splitter(
            [doc], self.min_tokens, self.max_tokens)]


def make_splitter() -> StructuralRSTSplitter:
    return StructuralRSTSplitter(min_tokens=256, max_tokens=512)

# ── Index building ────────────────────────────────────────────────────────────

def build_index(corpus_dir: Path, api_key: str) -> FAISS:
    loader = DirectoryLoader(
        path=str(corpus_dir),
        glob="*.rst",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    raw_docs = loader.load()
    log.info("Loaded %d RST files", len(raw_docs))

    chunks = make_splitter().split_documents(raw_docs)
    log.info("Split into %d chunks", len(chunks))

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=api_key,
        base_url=TRITON_BASE_URL,
        check_embedding_ctx_length=False,
    )
    index = FAISS.from_documents(chunks, embeddings)
    log.info("FAISS index built")
    return index

# ── Reranker + file-level selection ──────────────────────────────────────────

def load_reranker() -> HuggingFaceCrossEncoder:
    return HuggingFaceCrossEncoder(
        model_name="BAAI/bge-reranker-v2-m3",
        model_kwargs={"device": "cpu"},
    )


def select_best_files_and_rerank(
    query: str,
    docs: list[Document],
    reranker: HuggingFaceCrossEncoder,
) -> list[Document]:
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

    filtered = [(s, d) for s, d in scored
                if Path(d.metadata.get("source", "")).name in selected]
    filtered.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in filtered[:RERANKER_TOP_N]]

# ── Span → lines + merge ──────────────────────────────────────────────────────

_file_line_cache: dict[str, str] = {}


def _get_file_lines(fname: str, corpus_dir: Path) -> list[str]:
    if fname not in _file_line_cache:
        candidates = list(corpus_dir.rglob(fname))
        if not candidates:
            return []
        _file_line_cache[fname] = candidates[0].read_text(encoding="utf-8")
    return _file_line_cache[fname].splitlines()


def _content_to_lines(fname: str, content: str, corpus_dir: Path) -> tuple[int, int]:
    file_lines = _get_file_lines(fname, corpus_dir)
    if not file_lines:
        return 1, 1
    text   = "\n".join(file_lines)
    needle = content.strip()
    pos    = text.find(needle)
    if pos == -1:
        first_line = next((l.strip() for l in needle.splitlines() if l.strip()), "")
        pos = text.find(first_line) if first_line else -1
    if pos == -1:
        return 1, 1
    return text[:pos].count("\n") + 1, text[:pos + len(needle)].count("\n") + 1


def merge_spans_and_text(
    raw_spans: list[tuple[str, int, int, str, str]],
    corpus_dir: Path,
) -> tuple[list[dict[str, Any]], str]:
    converted: list[tuple[str, int, int]] = []
    for fname, start, end, content, coord in raw_spans:
        sl, el = _content_to_lines(fname, content, corpus_dir)
        converted.append((fname, sl, el))

    if not converted:
        return [], ""

    by_file: dict[str, list[tuple[int, int]]] = {}
    for fname, s, e in converted:
        by_file.setdefault(fname, []).append((s, e))

    merged: list[tuple[str, int, int]] = []
    for fname, intervals in by_file.items():
        intervals.sort(key=lambda x: x[0])
        cur_s, cur_e = intervals[0]
        for s, e in intervals[1:]:
            if s - cur_e <= MERGE_GAP:
                cur_e = max(cur_e, e)
            else:
                merged.append((fname, cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((fname, cur_s, cur_e))

    sources: list[dict[str, Any]] = []
    texts:   list[str]            = []
    for fname, s, e in merged:
        file_lines = _get_file_lines(fname, corpus_dir)
        lo   = max(0, s - 1)
        hi   = min(len(file_lines), e)
        text = "\n".join(file_lines[lo:hi])
        sources.append({"file": fname, "lines": [s, e]})
        texts.append(text)

    return sources, "\n\n".join(texts)

# ── Prompt + generation ───────────────────────────────────────────────────────

def build_user_content(question: str, context: str) -> str:
    enc = tiktoken.get_encoding("gpt2")
    overhead = len(enc.encode(
        SYSTEM_INSTRUCTIONS + "\n\n" +
        f"Question: {question}\n\nFind the answer in the Context:\n\n"
        f"Now answer the question:"
    ))
    budget        = max(MAX_TOKENS - overhead, 100)
    trunc_context = enc.decode(enc.encode(context)[:budget])
    return (
        f"Question: {question}\n\n"
        f"Find the answer in the Context: {trunc_context}\n\n"
        f"Now answer the question correctly, completely and faithfully '{question}':"
    )


def generate(question: str, context: str, client: OpenAI, model: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user",   "content": build_user_content(question, context)},
        ],
        max_tokens=MAX_COMPLETION,
    )
    return resp.choices[0].message.content.strip()

# ── Standalone pipeline ───────────────────────────────────────────────────────

def run_standalone(
    questions: list[dict],
    output_path: str,
    corpus_dir: Path,
    api_key: str,
    generation_model: str,
) -> None:
    index    = build_index(corpus_dir, api_key)
    reranker = load_reranker()
    client   = OpenAI(api_key=api_key, base_url=TRITON_BASE_URL)

    results = []
    for i, item in enumerate(questions):
        question_id = int(item["question_id"])
        question    = str(item["question"])
        log.info("[%d/%d] qid=%d", i + 1, len(questions), question_id)

        candidates = index.similarity_search(question, k=RETRIEVAL_K)
        top_docs   = select_best_files_and_rerank(question, candidates, reranker)

        raw_spans = [
            (
                Path(doc.metadata.get("source", "")).name,
                doc.metadata.get("start_line", 0),
                doc.metadata.get("end_line",   -1),
                doc.page_content,
                doc.metadata.get("span_coord", "line"),
            )
            for doc in top_docs
        ]

        sources, merged_text = merge_spans_and_text(raw_spans, corpus_dir)
        answer = generate(question, merged_text, client, generation_model)

        results.append({
            "question_id":       question_id,
            "answer":            answer,
            "retrieved_context": merged_text,
            "sources":           sources,
        })

    results.sort(key=lambda r: r["question_id"])
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("Wrote %d results to %s", len(results), output_path)

# ── RapidFire pipeline ────────────────────────────────────────────────────────

def run_rapidfire(
    questions: list[dict],
    output_path: str,
    corpus_dir: Path,
    api_key: str,
    generation_model: str,
) -> None:
    from typing import Dict, List as listtype
    from datasets import Dataset
    from rapidfireai.automl import (
        List, RFLangChainRagSpec, RFOpenAIAPIModelConfig, RFPromptManager, RFGridSearch,
    )
    from rapidfireai import Experiment

    output_rows: list[dict[str, Any]] = []
    _reranker: HuggingFaceCrossEncoder | None = None

    def preprocess_fn(batch: Dict[str, listtype], rag: RFLangChainRagSpec,
                      prompt_manager: RFPromptManager) -> Dict[str, listtype]:
        nonlocal _reranker
        if _reranker is None:
            _reranker = load_reranker()

        all_context_raw = rag.get_context(batch_queries=batch["query"], serialize=False)

        all_context = [
            select_best_files_and_rerank(q, docs, _reranker)
            for q, docs in zip(batch["query"], all_context_raw)
        ]

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
            _, merged_text = merge_spans_and_text(spans_raw, corpus_dir)
            serialized_context.append(merged_text)

        batch["query_id"] = [str(qid) for qid in batch["query_id"]]

        prompts = [
            [
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user",   "content": build_user_content(q, ctx)},
            ]
            for q, ctx in zip(batch["query"], serialized_context)
        ]

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
            sources, _ = merge_spans_and_text(spans_raw, corpus_dir)
            output_rows.append({
                "question_id":       int(qid),
                "answer":            answer,
                "retrieved_context": context,
                "sources":           sources,
            })
        return batch

    def compute_metrics_fn(batch: Dict[str, listtype]) -> Dict[str, Dict[str, Any]]:
        return {"Total": {"value": len(batch["query"])}}

    def accumulate_metrics_fn(agg: Dict[str, listtype]) -> Dict[str, Dict[str, Any]]:
        return {"Total": {"value": sum(m["value"] for m in agg["Total"])}}

    queries = Dataset.from_dict({
        "query":    [str(q["question"])    for q in questions],
        "query_id": [str(q["question_id"]) for q in questions],
    })

    rag = RFLangChainRagSpec(
        document_loader=DirectoryLoader(
            path=str(corpus_dir),
            glob="*.rst",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        ),
        text_splitter=make_splitter(),
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

    config_set = [{
        "openai_config":          openai_config,
        "batch_size":             BATCH_SIZE,
        "preprocess_fn":          preprocess_fn,
        "postprocess_fn":         postprocess_fn,
        "compute_metrics_fn":     compute_metrics_fn,
        "accumulate_metrics_fn":  accumulate_metrics_fn,
        "online_strategy_kwargs": {
            "strategy_name": "normal", "confidence_level": 0.95, "use_fpc": True,
        },
    }]

    experiment = Experiment(experiment_name="inference_structural", mode="evals")
    log.info("Starting inference (StructuralRSTSplitter, RapidFire)...")
    experiment.run_evals(
        config_group=RFGridSearch(config_set),
        dataset=queries,
        num_shards=4,
        seed=42,
    )
    experiment.end()

    output_rows.sort(key=lambda r: r["question_id"])
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_rows, f, indent=2, ensure_ascii=False)
    log.info("Wrote %d results to %s", len(output_rows), output_path)

# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG pipeline — StructuralRSTSplitter (min=256, max=512)"
    )
    parser.add_argument("--input",            required=True,
                        help="Input JSON [{question_id, question}, ...]")
    parser.add_argument("--output",           required=True,
                        help="Output predictions JSON path")
    parser.add_argument("--corpus-dir",       required=True,
                        help="Root of .rst corpus (sourcedocs/ layout)")
    parser.add_argument("--apikey-txt",       required=True,
                        help="Path to TritonAI gateway API key text file")
    parser.add_argument("--generation-model", required=True,
                        help="Gateway model id for answer generation")
    # parser.add_argument("--mode",             choices=["standalone", "rapidfire"],
    #                     default="standalone",
    #                     help="standalone: direct LangChain; rapidfire: RapidFire API")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        questions = json.load(f)
    log.info("Loaded %d questions from %s", len(questions), args.input)

    api_key    = load_api_key(args.apikey_txt)
    corpus_dir = Path(args.corpus_dir).expanduser()

    run_standalone(questions, args.output, corpus_dir, api_key, args.generation_model)


if __name__ == "__main__":
    main()
