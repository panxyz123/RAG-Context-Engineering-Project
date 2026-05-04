"""
Golden Q&A Dataset Generator for RAG Project
============================================
Chunking:  chapter-first → paragraph-cut at 600 chars
QA flow:   per-chunk question proposal (LLM picks category)
           → LLM answers (may refuse) → save valid pairs only
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from dotenv import load_dotenv

# =============================================================================
# Setup & Configuration
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
SOURCE_DIR = PROJECT_ROOT / "sourcedocs"

CONFIG = {
    "output_qa_path": "golden_qa_pairs_0503raw.json",
    "output_distribution_path": "question-type-distribution.json",
    "llm_model": "mistral.mistral-large-3-675b-instruct",
    "llm_temperature": 0.2,
    "llm_max_tokens": 700,
    "chunk_size_limit": 600,       # cut when accumulated chars exceed this
    "sleep_seconds": 1,
    "api_key_file": "/home/yup031/Project1/api-key.txt",
    "api_base_url": "https://tritonai-api.ucsd.edu",
}

QUESTION_TYPES = [
    "factual_lookup",
    "conceptual_explanation",
    "procedural",
    "comparative",
    "multi_file_synthesis",
]


# =============================================================================
# Data model
# =============================================================================

@dataclass
class TextChunk:
    content: str
    filename: str
    start_line: int
    end_line: int
    source_path: str

    def to_evidence(self) -> Dict[str, Any]:
        return {
            "file": self.filename,
            "lines": [self.start_line, self.end_line],
        }


# =============================================================================
# RST Loading  ─  chapter-first, then paragraph-cut at 600 chars
# =============================================================================

# RST title-overline/underline characters (standard + common extras)
_RST_ADORNMENT_RE = re.compile(r'^([=\-~^"#*+`\']{3,})\s*$')


def _is_chapter_boundary(lines: List[str], idx: int) -> bool:
    """
    Return True when line[idx] is the *overline* of a chapter heading,
    i.e. it matches  =====…=====  (or any repeated adornment char that
    spans the full width of the title), the next non-empty line is the
    title text, and the line after that is a matching underline.

    We treat *only* '======' style (= characters) as chapter boundaries
    because that is the conventional top-level marker in Sphinx / RST docs.
    Adjust the set if your project uses a different convention.
    """
    line = lines[idx]
    if not _RST_ADORNMENT_RE.match(line):
        return False
    # Must be '=' characters specifically for chapter level
    if not all(c == '=' for c in line.strip()):
        return False
    # Look ahead: title line + matching underline
    if idx + 2 >= len(lines):
        return False
    title_line = lines[idx + 1]
    under_line = lines[idx + 2]
    if not title_line.strip():
        return False
    if not _RST_ADORNMENT_RE.match(under_line):
        return False
    return True


def _split_into_chapters(lines: List[str]) -> List[List[str]]:
    """
    Split raw lines into chapter blocks.  Each block starts at the
    overline of a chapter heading (or at the very beginning of the file
    for any content that precedes the first heading).
    """
    chapter_starts = [i for i in range(len(lines)) if _is_chapter_boundary(lines, i)]

    if not chapter_starts:
        return [lines]

    chapters: List[List[str]] = []
    # Content before the first chapter heading (preamble)
    if chapter_starts[0] > 0:
        chapters.append(lines[: chapter_starts[0]])

    for k, start in enumerate(chapter_starts):
        end = chapter_starts[k + 1] if k + 1 < len(chapter_starts) else len(lines)
        chapters.append(lines[start:end])

    return [ch for ch in chapters if any(l.strip() for l in ch)]


def _paragraphs_from_lines(lines: List[str], base_line: int) -> List[tuple]:
    """
    Yield (content_str, abs_start_line, abs_end_line) for each paragraph
    in *lines*, where abs line numbers start at *base_line* (1-based).
    A paragraph is a run of non-empty lines separated by at least one
    blank line from its neighbours.
    """
    paragraphs: List[tuple] = []
    buf: List[str] = []
    buf_start = base_line

    for local_idx, line in enumerate(lines):
        abs_line = base_line + local_idx
        if line.strip():
            if not buf:
                buf_start = abs_line
            buf.append(line)
        else:
            if buf:
                paragraphs.append(("\n".join(buf), buf_start, abs_line - 1))
                buf = []

    if buf:
        paragraphs.append(("\n".join(buf), buf_start, base_line + len(lines) - 1))

    return paragraphs


def _cut_paragraphs_to_chunks(
    paragraphs: List[tuple],
    size_limit: int,
) -> List[tuple]:
    """
    Greedily accumulate paragraphs until the combined text exceeds
    *size_limit* characters, then seal the chunk and start a new one.
    Each returned tuple is (combined_text, start_line, end_line).
    """
    chunks: List[tuple] = []
    buf_texts: List[str] = []
    buf_start: Optional[int] = None
    buf_end: Optional[int] = None

    for para_text, p_start, p_end in paragraphs:
        if not para_text.strip():
            continue

        if buf_start is None:
            buf_start = p_start

        combined = "\n\n".join(buf_texts + [para_text])

        if buf_texts and len(combined) > size_limit:
            # Seal current buffer *before* this paragraph
            chunks.append(("\n\n".join(buf_texts), buf_start, buf_end))
            buf_texts = [para_text]
            buf_start = p_start
            buf_end = p_end
        else:
            buf_texts.append(para_text)
            buf_end = p_end

    if buf_texts:
        chunks.append(("\n\n".join(buf_texts), buf_start, buf_end))

    return chunks


def load_rst_files(source_dir: Path) -> List[TextChunk]:
    """
    Load all .rst files from *source_dir*.
    Chunking strategy:
      1. Split each file into chapters at '=====' headings.
      2. Within each chapter, split at paragraph boundaries (blank lines).
      3. Accumulate paragraphs until the combined text exceeds 600 chars,
         then start a new chunk.
    """
    chunks: List[TextChunk] = []
    rst_files = sorted(source_dir.glob("*.rst"))

    for rst_file in rst_files:
        text = rst_file.read_text(encoding="utf-8").replace("\r\n", "\n")
        all_lines = text.splitlines()

        chapters = _split_into_chapters(all_lines)
        current_abs_line = 1

        for chapter_lines in chapters:
            paragraphs = _paragraphs_from_lines(chapter_lines, current_abs_line)
            raw_chunks = _cut_paragraphs_to_chunks(paragraphs, CONFIG["chunk_size_limit"])

            for chunk_text, start_line, end_line in raw_chunks:
                if chunk_text.strip():
                    chunks.append(
                        TextChunk(
                            content=chunk_text.strip(),
                            filename=rst_file.name,
                            start_line=start_line,
                            end_line=end_line,
                            source_path=str(rst_file),
                        )
                    )

            current_abs_line += len(chapter_lines)

    logger.info(f"Created {len(chunks)} chunks from {len(rst_files)} RST files")
    return chunks


# =============================================================================
# LLM Client
# =============================================================================

def _get_client() -> OpenAI:
    api_key = Path(CONFIG["api_key_file"]).read_text().strip()
    return OpenAI(api_key=api_key, base_url=CONFIG["api_base_url"])


def _call_llm(client: OpenAI, prompt: str) -> str:
    response = client.chat.completions.create(
        model=CONFIG["llm_model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=CONFIG["llm_temperature"],
        max_tokens=CONFIG["llm_max_tokens"],
    )
    return response.choices[0].message.content.strip()


# =============================================================================
# Step 1 — Question Proposal  (LLM picks category + writes question)
# =============================================================================

_CATEGORY_LIST = "\n".join(f"- {qt}" for qt in QUESTION_TYPES)

QUESTION_PROPOSAL_PROMPT = """You are building a Q&A evaluation dataset for a RAG system.
Given the text chunk below, propose ONE high-quality question that can be answered using only the information in the chunk.
Also choose the single most appropriate category from the list provided.

Categories:
{categories}

Text chunk:
\"\"\"
{chunk}
\"\"\"

Respond with valid JSON only — no markdown fences, no extra text:
{{
  "category": "<one of the categories above>",
  "question": "<your proposed question>"
}}
If the chunk contains too little meaningful information to form a sensible question, return only the string [REFUSE].
"""


def propose_question(client: OpenAI, chunk: TextChunk) -> Optional[Dict[str, str]]:
    prompt = QUESTION_PROPOSAL_PROMPT.format(
        categories=_CATEGORY_LIST,
        chunk=chunk.content,
    )
    raw = _call_llm(client, prompt)

    if raw == "[REFUSE]":
        return None

    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        data = json.loads(match.group(0))
        category = data.get("category", "").strip()
        question = data.get("question", "").strip()
        if not category or not question:
            return None
        if category not in QUESTION_TYPES:
            # Accept closest match (case-insensitive) or reject
            lower_map = {qt.lower(): qt for qt in QUESTION_TYPES}
            category = lower_map.get(category.lower(), "")
            if not category:
                return None
        return {"category": category, "question": question}
    except (json.JSONDecodeError, AttributeError):
        return None


# =============================================================================
# Step 2 — Answer Generation  (same prompt style as original; refusal allowed)
# =============================================================================

ANSWER_GENERATION_PROMPT = """You are provided with a text snippet and a question about it.
Generate a high-quality answer using only the information visible in the snippet.
If the question is inappropriate, unanswerable from the snippet, or nonsensical,
return only the string [REFUSE].  Do not be overly strict — answer if it is reasonable.

Category: {category}

Snippet:
\"\"\"
{snippet}
\"\"\"

Question: {question}

Respond with valid JSON only — no markdown fences, no extra text:
{{
  "answer": "..."
}}
"""


def generate_answer(
    client: OpenAI, chunk: TextChunk, question: str, category: str
) -> Optional[str]:
    prompt = ANSWER_GENERATION_PROMPT.format(
        category=category,
        snippet=chunk.content,
        question=question,
    )
    raw = _call_llm(client, prompt)

    if raw == "[REFUSE]":
        return None

    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        data = json.loads(match.group(0))
        answer = data.get("answer", "").strip()
        if not answer or answer == "[REFUSE]":
            return None
        return answer
    except (json.JSONDecodeError, AttributeError):
        return None


# =============================================================================
# Output helpers
# =============================================================================

def generate_question_type_distribution(qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    distribution: Dict[str, Any] = {}
    for qt in QUESTION_TYPES:
        ids = [i for i, p in enumerate(qa_pairs) if p.get("question_type") == qt]
        distribution[qt] = {"question_ids": ids, "count": len(ids)}

    summary = {
        "total_questions": len(qa_pairs),
        "unique_type_assignments": sum(
            1 for qt in QUESTION_TYPES if distribution[qt]["count"] > 0
        ),
        "questions_with_single_type": len(
            [p for p in qa_pairs if isinstance(p.get("question_type"), str)]
        ),
        "questions_with_multiple_types": 0,
    }
    return {"question_type_distribution": distribution, "summary": summary}


def save_outputs(qa_pairs: List[Dict[str, Any]], output_dir: Path) -> None:
    # --- qa json (same schema as original) ---
    output_file = output_dir / CONFIG["output_qa_path"]
    output_records = [
        {
            "question": p["question"],
            "reference_answer": p["reference_answer"],
            "source_evidence": p["source_evidence"],
        }
        for p in qa_pairs
    ]
    output_file.write_text(
        json.dumps(output_records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"Saved {len(output_records)} Q&A pairs → {output_file}")

    # --- distribution / analysis json (same schema as original) ---
    dist_file = output_dir / CONFIG["output_distribution_path"]
    dist_file.write_text(
        json.dumps(
            generate_question_type_distribution(qa_pairs), indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    logger.info(f"Saved distribution → {dist_file}")


# =============================================================================
# Main pipeline
# =============================================================================

def run_golden_qa_pipeline() -> None:
    logger.info("Starting Golden Q&A Generation Pipeline")

    chunks = load_rst_files(SOURCE_DIR)
    if not chunks:
        logger.error("No chunks found — check SOURCE_DIR path.")
        return

    client = _get_client()
    qa_pairs: List[Dict[str, Any]] = []
    seen_questions: set = set()

    refused_proposal = 0
    refused_answer = 0
    duplicate = 0

    for idx, chunk in enumerate(chunks):
        logger.info(f"[{idx + 1}/{len(chunks)}] {chunk.filename} lines {chunk.start_line}–{chunk.end_line}")

        # --- Step 1: propose question ---
        proposal = propose_question(client, chunk)
        time.sleep(CONFIG["sleep_seconds"])

        if proposal is None:
            refused_proposal += 1
            logger.debug("  → proposal refused")
            continue

        question = proposal["question"]
        category = proposal["category"]

        if question in seen_questions:
            duplicate += 1
            logger.debug("  → duplicate question, skipping")
            continue

        # --- Step 2: generate answer ---
        answer = generate_answer(client, chunk, question, category)
        time.sleep(CONFIG["sleep_seconds"])

        if answer is None:
            refused_answer += 1
            logger.debug("  → answer refused")
            continue

        qa_pairs.append(
            {
                "question": question,
                "reference_answer": answer,
                "source_evidence": [chunk.to_evidence()],
                "question_type": category,
            }
        )
        seen_questions.add(question)
        logger.info(f"  ✓ [{category}] {question[:80]}…")

    logger.info(
        f"Done. valid={len(qa_pairs)}, "
        f"refused_proposal={refused_proposal}, "
        f"refused_answer={refused_answer}, "
        f"duplicates={duplicate}"
    )

    if qa_pairs:
        save_outputs(qa_pairs, BASE_DIR)
    else:
        logger.warning("No valid Q&A pairs generated — nothing saved.")

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    run_golden_qa_pipeline()
