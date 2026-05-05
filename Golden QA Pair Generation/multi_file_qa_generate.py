"""
Multi-File QA Synthesiser
=========================
1. Load golden_qa_pairs_0503raw.json
2. Load the original .rst source texts
3. Embed all questions, cluster similar ones that span different files
4. For each cluster, call LLM with all source snippets to synthesise a
   single multi-file-synthesis QA pair
5. Save results to multi_file_qa_pairs.json
"""

import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# =============================================================================
# Configuration
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
load_dotenv()

BASE_DIR   = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR.parent / "sourcedocs"

CONFIG = {
    "raw_json"         : Path("/home/yup031/Project1/Golden QA Pair Generation/golden_qa_pairs_0503raw.json"),
    "output_path"      : Path("/home/yup031/Project1/Golden QA Pair Generation/multi_file_qa_pairs.json"),
    "embedding_model"  : "sentence-transformers/all-MiniLM-L6-v2",
    "similarity_threshold" : 0.72,   # cosine sim above this → candidate for merging
    "min_sources"      : 2,          # must span at least this many distinct files
    "max_sources"      : 4,          # cap snippets fed to LLM to avoid token overflow
    "llm_model"        : "mistral.mistral-large-3-675b-instruct",
    "llm_temperature"  : 0.2,
    "llm_max_tokens"   : 900,
    "sleep_seconds"    : 1,
    "api_key_file"     : "/home/yup031/Project1/api-key.txt",
    "api_base_url"     : "https://tritonai-api.ucsd.edu",
}

# =============================================================================
# Loaders
# =============================================================================

def load_raw_pairs(path: Path) -> List[Dict[str, Any]]:
    pairs = json.loads(path.read_text(encoding="utf-8"))
    # Attach explicit question_id if not present
    for i, p in enumerate(pairs):
        if "question_id" not in p:
            p["question_id"] = i
    logger.info(f"Loaded {len(pairs)} raw QA pairs")
    return pairs


def load_rst_sources(source_dir: Path) -> Dict[str, List[str]]:
    """Return {filename: list_of_lines}."""
    sources: Dict[str, List[str]] = {}
    for rst_file in sorted(source_dir.glob("*.rst")):
        text = rst_file.read_text(encoding="utf-8").replace("\r\n", "\n")
        sources[rst_file.name] = text.splitlines()
    logger.info(f"Loaded {len(sources)} RST source files")
    return sources


def extract_snippet(sources: Dict[str, List[str]], evidence: Dict[str, Any]) -> str:
    filename = evidence.get("file", "")
    lines    = evidence.get("lines", [])
    file_lines = sources.get(filename, [])
    if not file_lines:
        return ""
    if lines and len(lines) == 2:
        start = max(0, lines[0] - 1)
        end   = min(len(file_lines), lines[1])
        return "\n".join(file_lines[start:end])
    return "\n".join(file_lines)


# =============================================================================
# Embedding & clustering
# =============================================================================

def embed_questions(questions: List[str], model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(questions, convert_to_numpy=True, normalize_embeddings=True)
    logger.info(f"Embedded {len(questions)} questions")
    return embeddings


def find_similar_clusters(
    pairs: List[Dict[str, Any]],
    embeddings: np.ndarray,
    threshold: float,
    min_sources: int,
    max_sources: int,
) -> List[List[int]]:
    """
    Greedily cluster questions by cosine similarity.
    A cluster is only kept if it spans ≥ min_sources distinct files.
    Each pair index appears in at most one cluster (first-match wins).
    """
    n = len(pairs)
    sim_matrix = embeddings @ embeddings.T   # (n, n) — normalised so dot = cosine

    assigned = [False] * n
    clusters: List[List[int]] = []

    for i in range(n):
        if assigned[i]:
            continue

        # Find all unassigned pairs similar to i, from different files
        anchor_file = _primary_file(pairs[i])
        group = [i]
        seen_files = {anchor_file}

        # Sort candidates by similarity descending
        sims = sim_matrix[i]
        candidates = sorted(
            [j for j in range(n) if j != i and not assigned[j]],
            key=lambda j: sims[j],
            reverse=True,
        )

        for j in candidates:
            if sims[j] < threshold:
                break
            if len(group) >= max_sources:
                break
            jfile = _primary_file(pairs[j])
            if jfile not in seen_files:        # enforce cross-file diversity
                group.append(j)
                seen_files.add(jfile)

        if len(seen_files) >= min_sources:
            clusters.append(group)
            for idx in group:
                assigned[idx] = True

    logger.info(f"Found {len(clusters)} multi-file clusters")
    return clusters


def _primary_file(pair: Dict[str, Any]) -> str:
    evidence = pair.get("source_evidence", [])
    return evidence[0].get("file", "unknown") if evidence else "unknown"


# =============================================================================
# LLM
# =============================================================================

def _get_client() -> OpenAI:
    api_key = Path(CONFIG["api_key_file"]).read_text().strip()
    return OpenAI(api_key=api_key, base_url=CONFIG["api_base_url"])


def _call_llm(client: OpenAI, prompt: str) -> Optional[str]:
    response = client.chat.completions.create(
        model=CONFIG["llm_model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=CONFIG["llm_temperature"],
        max_tokens=CONFIG["llm_max_tokens"],
    )
    choice = response.choices[0]
    content = choice.message.content
    if content is None:
        logger.warning(f"LLM returned None content (finish_reason={choice.finish_reason!r})")
        return None
    return content.strip()


SYNTHESIS_PROMPT = """You are building a high-quality Q&A evaluation dataset for a RAG system.

You are given {n} related text snippets from DIFFERENT source files, along with existing
questions and answers derived from each snippet individually.

Your task:
1. Read all snippets carefully.
2. Identify the common theme or relationship that connects them.
3. Write ONE new question that can ONLY be answered by drawing on information from
   MULTIPLE snippets — a question that no single snippet alone can fully answer.
4. Write a comprehensive answer that explicitly references information from each snippet.

Rules:
- Use ONLY information present in the snippets. Do not add outside knowledge.
- The question must genuinely require cross-file synthesis — not just a rephrasing of one snippet.
- If the snippets are too unrelated to form a meaningful cross-file question, return [REFUSE].
- Do not be overly strict — merge if there is a reasonable thematic connection.

{snippets_block}

Existing individual questions for context (do NOT reuse these verbatim):
{existing_questions}

Respond with valid JSON only — no markdown fences, no extra text:
{{
  "question": "...",
  "answer"  : "..."
}}
"""


def build_synthesis_prompt(
    cluster_pairs: List[Dict[str, Any]],
    snippets: List[Tuple[str, str]],   # [(filename, text), ...]
) -> str:
    snippets_block = "\n\n".join(
        f"Snippet {i+1} — [{fname}]:\n\"\"\"\n{text}\n\"\"\""
        for i, (fname, text) in enumerate(snippets)
    )
    existing_qs = "\n".join(
        f"  - {p.get('question', '')}" for p in cluster_pairs
    )
    return SYNTHESIS_PROMPT.format(
        n=len(snippets),
        snippets_block=snippets_block,
        existing_questions=existing_qs,
    )


def parse_synthesis_response(raw: str) -> Optional[Dict[str, str]]:
    if raw.strip() == "[REFUSE]":
        return None
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        data = json.loads(match.group(0))
        question = data.get("question", "").strip()
        answer   = data.get("answer",   "").strip()
        if not question or not answer or answer == "[REFUSE]":
            return None
        return {"question": question, "answer": answer}
    except (json.JSONDecodeError, AttributeError):
        return None


# =============================================================================
# Pipeline
# =============================================================================

def synthesise_cluster(
    client: OpenAI,
    cluster_indices: List[int],
    pairs: List[Dict[str, Any]],
    sources: Dict[str, List[str]],
) -> Optional[Dict[str, Any]]:
    """Build one multi-file QA pair from a cluster of similar single-file pairs."""
    cluster_pairs = [pairs[i] for i in cluster_indices]

    # Gather snippets  (deduplicate by file)
    seen_files: set = set()
    snippets: List[Tuple[str, str]] = []
    evidence_out: List[Dict[str, Any]] = []

    for pair in cluster_pairs:
        for ev in pair.get("source_evidence", []):
            fname = ev.get("file", "")
            if fname in seen_files:
                continue
            text = extract_snippet(sources, ev)
            if not text.strip():
                continue
            snippets.append((fname, text))
            evidence_out.append(ev)
            seen_files.add(fname)

    if len(seen_files) < CONFIG["min_sources"]:
        return None

    prompt = build_synthesis_prompt(cluster_pairs, snippets)
    raw = _call_llm(client, prompt)
    if raw is None:
        return None

    qa = parse_synthesis_response(raw)
    if qa is None:
        return None

    return {
        "question"        : qa["question"],
        "reference_answer": qa["answer"],
        "source_evidence" : evidence_out,
        "question_type"   : "multi_file_synthesis",
        "merged_from_ids" : [p["question_id"] for p in cluster_pairs],
    }


def run_multi_file_pipeline() -> None:
    logger.info("Starting Multi-File QA Synthesis Pipeline")

    raw_pairs = load_raw_pairs(CONFIG["raw_json"])
    sources   = load_rst_sources(SOURCE_DIR)

    questions  = [p.get("question", "") for p in raw_pairs]
    embeddings = embed_questions(questions, CONFIG["embedding_model"])

    clusters = find_similar_clusters(
        raw_pairs,
        embeddings,
        threshold   = CONFIG["similarity_threshold"],
        min_sources = CONFIG["min_sources"],
        max_sources = CONFIG["max_sources"],
    )

    if not clusters:
        logger.warning("No multi-file clusters found — try lowering similarity_threshold")
        return

    client = _get_client()
    output_pairs: List[Dict[str, Any]] = []
    refused = 0

    for idx, cluster in enumerate(clusters):
        files = list({_primary_file(raw_pairs[i]) for i in cluster})
        logger.info(f"[{idx+1}/{len(clusters)}] Cluster size={len(cluster)} files={files}")

        result = synthesise_cluster(client, cluster, raw_pairs, sources)
        time.sleep(CONFIG["sleep_seconds"])

        if result is None:
            refused += 1
            logger.info("  → refused / skipped")
            continue

        output_pairs.append(result)
        logger.info(f"  ✓ {result['question'][:90]}…")

    logger.info(f"Generated {len(output_pairs)} multi-file pairs ({refused} refused)")

    if not output_pairs:
        logger.warning("Nothing to save.")
        return

    out_path = CONFIG["output_path"]
    out_path.write_text(
        json.dumps(output_pairs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Saved → {out_path}")

    # Summary
    all_files: set = set()
    for p in output_pairs:
        for ev in p.get("source_evidence", []):
            all_files.add(ev.get("file", ""))
    logger.info(f"  Unique source files covered : {len(all_files)}")
    logger.info(f"  Avg sources per pair        : "
                f"{sum(len(p['source_evidence']) for p in output_pairs)/len(output_pairs):.1f}")
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    run_multi_file_pipeline()