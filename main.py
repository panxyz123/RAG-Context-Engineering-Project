"""main_standalone.py — RAG inference pipeline without RapidFire AI API.

Same config as main.py:
    Splitter  : LineNumberSplitter, chunk_size=128, overlap=32
    Retrieval : FAISS similarity, k=20
    Reranker  : BAAI/bge-reranker-v2-m3, top_n=5
    Generator : api-mistral-small-3.2-2506

Usage:
    python3 main_standalone.py --input test_questions.json --output predictions.json

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
from typing import Any

import tiktoken
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

SOURCEDOCS_PATH = Path("/home/yup031/Project1/sourcedocs/")
TRITON_API_KEY  = os.environ.get("TRITON_API_KEY") or \
                  Path("/home/yup031/Project1/api-key.txt").expanduser().read_text(encoding="utf-8").splitlines()[0].strip()
TRITON_BASE_URL = "https://tritonai-api.ucsd.edu"
EMBEDDING_MODEL = "api-tgpt-embeddings"
GENERATOR_MODEL = "api-mistral-small-3.2-2506"
RETRIEVAL_K     = 20
RERANKER_TOP_N  = 3
BATCH_SIZE      = 32
MAX_TOKENS      = 2000

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


def _content_to_lines(fname: str, content: str) -> tuple[int, int]:
    """Find chunk content in source file and return 1-based line numbers."""
    if fname not in _file_line_cache:
        candidates = list(SOURCEDOCS_PATH.rglob(fname))
        if not candidates:
            return 1, 1
        _file_line_cache[fname] = candidates[0].read_text(encoding="utf-8")
    text = _file_line_cache[fname]

    # Try exact match first
    needle = content.strip()
    pos = text.find(needle)

    # Fallback: match on first non-empty line only
    if pos == -1:
        first_line = next((l.strip() for l in needle.splitlines() if l.strip()), "")
        pos = text.find(first_line) if first_line else -1

    if pos == -1:
        return 1, 1

    start_line = text[:pos].count("\n") + 1
    end_line   = text[:pos + len(needle)].count("\n") + 1
    log.info("Mapped char offsets to lines for %s: %d-%d", fname, start_line, end_line)
    return start_line, end_line


# ── Output formatting ─────────────────────────────────────────────────────────

def _spans_to_sources(
    raw_spans: list[tuple[str, int, int, str, str]],
) -> list[dict[str, Any]]:
    """Convert raw (file, start, end, span_coord) tuples to output sources schema.

    Converts char offsets to line numbers when span_coord="char".
    Deduplicates by (file, start_line, end_line).
    """
    seen = set()
    out  = []
    for fname, start, end, content, coord in raw_spans:
        if coord == "char":
            start_line, end_line = _content_to_lines(fname, content)
        else:
            start_line, end_line = int(start), int(end)
        key = (fname, start_line, end_line)
        if key not in seen:
            seen.add(key)
            out.append({"file": fname, "lines": [start_line, end_line]})
    return out


# ── Index building ────────────────────────────────────────────────────────────

def build_index() -> tuple[FAISS, list[Document]]:
    log.info("Loading corpus from %s", SOURCEDOCS_PATH)
    loader = DirectoryLoader(
        path=str(SOURCEDOCS_PATH),
        glob="*.rst",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    raw_docs = loader.load()
    log.info("Loaded %d RST files", len(raw_docs))

    splitter = LineNumberSplitter(
        RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="gpt2",
            chunk_size=256,
            chunk_overlap=64,
            add_start_index=True,
        )
    )
    chunks = splitter.split_documents(raw_docs)
    log.info("Split into %d chunks", len(chunks))

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=TRITON_API_KEY,
        base_url=TRITON_BASE_URL,
        check_embedding_ctx_length=False,
    )

    log.info("Building FAISS index ...")
    index = FAISS.from_documents(chunks, embeddings)
    log.info("Index built")
    return index, chunks


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    index: FAISS,
    reranker: HuggingFaceCrossEncoder,
) -> list[Document]:
    """Retrieve top-k by similarity then rerank with BGE."""
    candidates = index.similarity_search(query, k=RETRIEVAL_K)

    # Score all candidates with cross-encoder
    pairs  = [(query, doc.page_content) for doc in candidates]
    scores = reranker.score(pairs)

    # Sort by score descending, keep top_n
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:RERANKER_TOP_N]]


# ── Generation ────────────────────────────────────────────────────────────────

def generate(question: str, context: str, client: OpenAI, enc: tiktoken.Encoding) -> str:
    def truncate(text: str) -> str:
        tokens = enc.encode(text)
        return enc.decode(tokens[:MAX_TOKENS]) if len(tokens) > MAX_TOKENS else text

    user_content = truncate(f"Question: {question}\n\nContext:\n{context}\n\nAnswer:")
    resp = client.chat.completions.create(
        model=GENERATOR_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user",   "content": user_content},
        ],
        max_tokens=2048,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


# ── Main ──────────────────────────────────────────────────────────────────────

def run(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    log.info("Loaded %d questions from %s", len(questions), input_path)

    index, _ = build_index()

    reranker = HuggingFaceCrossEncoder(
        model_name="BAAI/bge-reranker-v2-m3",
        model_kwargs={"device": "cpu"},
    )

    client = OpenAI(api_key=TRITON_API_KEY, base_url=TRITON_BASE_URL)
    enc    = tiktoken.get_encoding("gpt2")

    results = []
    for i, item in enumerate(questions):
        # test on first 10 questions for now
        if i>10:
            continue
        question_id = int(item["question_id"])
        question    = str(item["question"])
        log.info("[%d/%d] question_id=%d", i + 1, len(questions), question_id)

        docs    = retrieve(question, index, reranker)
        context = "\n\n".join(doc.page_content for doc in docs)
        answer  = generate(question, context, client, enc)
        
        raw_spans = [
            (
                Path(doc.metadata.get("source", "")).name,
                doc.metadata.get("start_index", doc.metadata.get("start_line", 0)),
                doc.metadata.get("start_index", 0) + len(doc.page_content)
                    if doc.metadata.get("span_coord", "char") == "char"
                    else doc.metadata.get("end_line", -1),
                doc.page_content,
                doc.metadata.get("span_coord", "char"),
            )
            for doc in docs
        ]

        results.append({
            "question_id":       question_id,
            "answer":            answer,
            "retrieved_context": context,
            "sources":           _spans_to_sources(raw_spans),
        })

    results.sort(key=lambda r: r["question_id"])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("Wrote %d results to %s", len(results), output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RapidFire AI RST Q&A pipeline (standalone)")
    parser.add_argument("--input",  required=True, help="Input JSON (list of {question_id, question})")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()
    run(args.input, args.output)
