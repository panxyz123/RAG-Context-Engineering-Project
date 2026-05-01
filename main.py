"""main.py — Standalone RAG pipeline for RapidFire AI documentation Q&A.

Usage:
    python3 main.py --input test_questions.json --output predictions.json

Input schema  (list of dicts):
    [{"question_id": <int>, "question": <str>}, ...]

Output schema (list of dicts):
    [{"question_id": <int>,
      "answer": <str>,
      "retrieved_context": <str>,
      "sources": [{"file": <str>, "lines": [<int>, <int>]}, ...]
     }, ...]

Design choices
--------------
Splitter   : rst_structural_splitter — splits on RST heading underlines and
             blank-line paragraph boundaries. Produces start_line/end_line
             natively without char-offset conversion. Best alignment with
             ground-truth spans which are also paragraph/section-level.
Embedding  : OpenAI-compatible endpoint via TRITON_API_KEY
Retrieval  : FAISS similarity search, k=10, then CrossEncoder reranker top_n=5
Generator  : Mistral via Triton endpoint, max 2048 tokens
Context    : Top-5 reranked chunks joined by double newline, truncated to
             2000 GPT-2 tokens before being sent to the generator
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import tiktoken
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

SOURCEDOCS_PATH   = Path("~/Project1/sourcedocs").expanduser()
API_KEY_PATH      = Path("~/Project1/api-key.txt").expanduser()
TRITON_BASE_URL   = "https://tritonai-api.ucsd.edu"
EMBEDDING_MODEL   = "api-tgpt-embeddings"
GENERATOR_MODEL   = "api-mistral-small-3.2-2506"
MAX_CONTEXT_TOKENS = 2000
RETRIEVAL_K        = 20
RERANKER_TOP_N     = 3
BATCH_SIZE         = 32

SYSTEM_INSTRUCTIONS = (
    "You are a helpful QA assistant. "
    "Answer the question based solely on the provided context. "
    "Be concise and accurate."
)

# ── API key ──────────────────────────────────────────────────────────────────

def load_api_key() -> str:
    return API_KEY_PATH.read_text(encoding="utf-8").splitlines()[0].strip()


# ── RST structural splitter ──────────────────────────────────────────────────

_RST_HEADING_RE = re.compile(r"^([=\-~^'`#*+])(\1*)$")


def rst_structural_splitter(
    documents: list[Document],
    min_tokens: int = 250,
    max_tokens: int = 600,
    tokenizer_encoding: str = "gpt2",
) -> list[Document]:
    """Split RST documents on structural boundaries (sections, paragraphs).

    Writes start_line / end_line (1-based, inclusive) directly into metadata.
    Falls back to sentence splitting for paragraphs that exceed max_tokens.
    """
    enc = tiktoken.get_encoding(tokenizer_encoding)

    def token_len(text: str) -> int:
        return len(enc.encode(text))

    chunks: list[Document] = []

    for doc in documents:
        src   = doc.metadata.get("source", "")
        lines = doc.page_content.splitlines(keepends=True)

        # pass 1: locate section boundaries
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

            # pass 2: split into paragraphs on blank lines
            paragraphs: list[list[tuple[int, str]]] = []
            current:    list[tuple[int, str]]        = []
            for offset, line in enumerate(section_lines):
                lineno = sec_start + offset + 1  # 1-based
                if line == "" and current:
                    paragraphs.append(current)
                    current = []
                else:
                    current.append((lineno, line))
            if current:
                paragraphs.append(current)

            # pass 3: merge tiny paragraphs; split oversized ones
            # flush is defined outside the section loop to avoid closure capture issues
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
                    # Split by individual lines to preserve exact line numbers.
                    # Sentence splitting loses line number info because para_text
                    # is a stripped string with no per-line tracking.
                    _flush(buffer, src)
                    buffer = []
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

                else:
                    _flush(buffer, src)
                    buffer = para[:]

            _flush(buffer, src)

    return chunks


# ── Corpus loading & indexing ─────────────────────────────────────────────────

def build_index(api_key: str) -> tuple[FAISS, list[Document]]:
    """Load RST corpus, split structurally, embed, and return FAISS index."""
    log.info("Loading corpus from %s", SOURCEDOCS_PATH)
    loader = DirectoryLoader(
        path=str(SOURCEDOCS_PATH),
        glob="*.rst",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    raw_docs = loader.load()
    log.info("Loaded %d RST files", len(raw_docs))

    chunks = rst_structural_splitter(raw_docs, min_tokens=30, max_tokens=300)
    log.info("Split into %d chunks", len(chunks))

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=api_key,
        base_url=TRITON_BASE_URL,
        check_embedding_ctx_length=False,
    )

    texts     = [c.page_content for c in chunks]
    metadatas = [c.metadata      for c in chunks]

    log.info("Building FAISS index ...")
    index = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    log.info("Index built")
    return index, chunks


# ── Retrieval ────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    index: FAISS,
    reranker: CrossEncoderReranker,
) -> list[Document]:
    """Retrieve top-k docs by similarity then rerank."""
    docs = index.similarity_search(query, k=RETRIEVAL_K)
    # CrossEncoderReranker expects a compressor interface; call compress_documents
    reranked = reranker.compress_documents(docs, query)
    return reranked[:RERANKER_TOP_N]


# ── Context serialisation ────────────────────────────────────────────────────

def custom_template(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def truncate_to_tokens(text: str, max_tokens: int, enc: tiktoken.Encoding) -> str:
    tokens = enc.encode(text)
    return enc.decode(tokens[:max_tokens]) if len(tokens) > max_tokens else text


# ── Generation ───────────────────────────────────────────────────────────────

def generate(
    question: str,
    context: str,
    client: OpenAI,
    enc: tiktoken.Encoding,
) -> str:
    user_content = truncate_to_tokens(
        f"Question: {question}\n\nContext:\n{context}\n\nAnswer:",
        MAX_CONTEXT_TOKENS,
        enc,
    )
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


# ── Output formatting ────────────────────────────────────────────────────────

def docs_to_sources(docs: list[Document]) -> list[dict[str, Any]]:
    """Convert retrieved Documents to the required sources schema.

    Deduplicates (file, start_line, end_line) triples. A single file can
    appear multiple times if different line ranges were retrieved.
    """
    seen  = set()
    out   = []
    for doc in docs:
        fname      = Path(doc.metadata.get("source", "")).name
        start_line = int(doc.metadata.get("start_line", 0))
        end_line   = int(doc.metadata.get("end_line",   0))
        key = (fname, start_line, end_line)
        if key not in seen:
            seen.add(key)
            out.append({"file": fname, "lines": [start_line, end_line]})
    return out


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(input_path: str, output_path: str) -> None:
    # Load input
    with open(input_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    log.info("Loaded %d questions from %s", len(questions), input_path)

    api_key = load_api_key()

    # Build index (one-time cost)
    index, _ = build_index(api_key)

    # Reranker
    # reranker = CrossEncoderReranker(
    #     model="cross-encoder/ms-marco-MiniLM-L6-v2",
    #     top_n=RERANKER_TOP_N,
    # )
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    reranker = CrossEncoderReranker(
        model=HuggingFaceCrossEncoder(
            model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        ),
        top_n=RERANKER_TOP_N,
    )

    # Generator client
    client = OpenAI(api_key=api_key, base_url=TRITON_BASE_URL)
    enc    = tiktoken.get_encoding("gpt2")

    results = []
    for i, item in enumerate(questions):
        question_id = int(item["question_id"])
        question    = str(item["question"])
        log.info("[%d/%d] question_id=%d", i + 1, len(questions), question_id)

        # Retrieve + rerank
        docs    = retrieve(question, index, reranker)
        context = custom_template(docs)

        # Generate
        answer  = generate(question, context, client, enc)

        results.append({
            "question_id":        question_id,
            "answer":             answer,
            "retrieved_context":  context,
            "sources":            docs_to_sources(docs),
        })

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("Wrote %d results to %s", len(results), output_path)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RapidFire AI RST Q&A pipeline")
    parser.add_argument("--input",  required=True, help="Input JSON file (list of {question_id, question})")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    args = parser.parse_args()
    run(args.input, args.output)
