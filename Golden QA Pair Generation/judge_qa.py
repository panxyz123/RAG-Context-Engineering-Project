"""
QA Judge  —  evaluate golden_qa_pairs.json
==========================================
Metrics per QA pair (evaluated independently for Q and A):
  • completeness  : 1–5  (does it fully address the information in the source?)
  • faithfulness  : pass/fail  (no hallucination vs. the source RST text?)
  • correctness   : pass/fail  (factually accurate given the source?)

Final pair quality score = average of the 6 normalised sub-scores (3 for Q, 3 for A).

Output
  • judged_qa_scores.csv  — per-pair scores + metadata
  • judged_qa_sorted.csv  — hybrid-sorted with unique index
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# =============================================================================
# Configuration
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
load_dotenv()

BASE_DIR   = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR.parent / "sourcedocs"

CONFIG = {
    "qa_input_path"        : BASE_DIR / "golden_qa_pairs_0503raw.json",
    "output_scores_path"   : BASE_DIR / "judged_qa_scores.csv",
    "output_sorted_path"   : BASE_DIR / "judged_qa_sorted.csv",
    "llm_model"            : "api-gemma-4-26b",
    "llm_temperature"      : 0.1,     # low temp → deterministic judgements
    "llm_max_tokens"       : 2000,
    "sleep_seconds"        : 1,
    "api_key_file"         : "/home/yup031/Project1/api-key.txt",
    "api_base_url"         : "https://tritonai-api.ucsd.edu",
    # Hybrid sort weights  (must sum to 1.0)
    "weight_quality"       : 0.60,
    "weight_file_diversity": 0.20,
    "weight_type_diversity": 0.20,
}

# =============================================================================
# RST source loader  —  returns {filename: full_text}
# =============================================================================

def load_rst_sources(source_dir: Path) -> Dict[str, str]:
    sources: Dict[str, str] = {}
    for rst_file in sorted(source_dir.glob("*.rst")):
        sources[rst_file.name] = rst_file.read_text(encoding="utf-8")
    logger.info(f"Loaded {len(sources)} RST source files")
    return sources


def get_source_text(sources: Dict[str, str], evidence: List[Dict]) -> str:
    """
    Extract the relevant lines from the source file(s) named in evidence.
    Falls back to the full file if line numbers are missing / out of range.
    """
    snippets: List[str] = []
    for ev in evidence:
        filename = ev.get("file", "")
        lines_range = ev.get("lines", [])
        text = sources.get(filename, "")
        if not text:
            continue
        all_lines = text.splitlines()
        if lines_range and len(lines_range) == 2:
            start = max(0, lines_range[0] - 1)
            end   = min(len(all_lines), lines_range[1])
            snippet = "\n".join(all_lines[start:end])
        else:
            snippet = text
        snippets.append(f"[{filename}]\n{snippet}")
    return "\n\n".join(snippets) if snippets else ""


# =============================================================================
# LLM client
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
    choice = response.choices[0]
    content = choice.message.content
    if content is None:
        logger.warning(f"Response content is None (finish_reason={choice.finish_reason!r})")
        return None
    return content.strip()


# =============================================================================
# Judge prompts
# =============================================================================

JUDGE_PROMPT = """You are a strict but fair evaluator for a RAG (Retrieval-Augmented Generation) dataset.

Evaluate BOTH the question and the answer on three metrics.
Base all judgements solely on the SOURCE TEXT — do not use outside knowledge.

METRICS
-------
1. completeness (integer 1-5, with mean around 4 for complete coverage of the source and <3 for significant gaps)

2. faithfulness (pass | fail, where "pass" means no hallucination vs. the source RST text, and "fail" means some hallucination)

3. correctness (pass | fail, where "pass" means factually accurate given the source RST text, and "fail" means some factual inaccuracy)

SOURCE TEXT:
\"\"\"
{source}
\"\"\"

QUESTION:
{question}

ANSWER:
{answer}

Respond with valid JSON only — no markdown fences, no extra text:
{{
  "question": {{
    "completeness": <1-5>,
    "faithfulness": "pass" | "fail",
    "correctness" : "pass" | "fail"
  }},
  "answer": {{
    "completeness": <1-5>,
    "faithfulness": "pass" | "fail",
    "correctness" : "pass" | "fail"
  }}
}}
"""


# =============================================================================
# Parse judge response
# =============================================================================

def _pf_to_float(value: str) -> float:
    """Convert 'pass'/'fail' to 1.0/0.0; handle typos gracefully."""
    return 1.0 if str(value).strip().lower() == "pass" else 0.0


def _completeness_to_float(value: Any) -> float:
    """Normalise a 1–5 completeness score to [0, 1]."""
    try:
        v = float(value)
        return max(0.0, min(1.0, (v - 1) / 4.0))
    except (TypeError, ValueError):
        return 0.0


def parse_judge_response(raw: str) -> Optional[Dict[str, Any]]:
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        data = json.loads(match.group(0))

        result: Dict[str, Any] = {}
        for role in ("question", "answer"):
            block = data.get(role, {})
            result[f"{role}_completeness"] = int(block.get("completeness", 1))
            result[f"{role}_faithfulness"] = str(block.get("faithfulness", "fail")).lower()
            result[f"{role}_correctness"]  = str(block.get("correctness",  "fail")).lower()

        # Normalised sub-scores  (each in [0,1])
        norm_scores = [
            _completeness_to_float(result["question_completeness"]),
            _pf_to_float(result["question_faithfulness"]),
            _pf_to_float(result["question_correctness"]),
            _completeness_to_float(result["answer_completeness"]),
            _pf_to_float(result["answer_faithfulness"]),
            _pf_to_float(result["answer_correctness"]),
        ]
        result["quality_score"] = round(sum(norm_scores) / len(norm_scores), 4)
        return result

    except (json.JSONDecodeError, AttributeError, KeyError):
        return None


# =============================================================================
# Judge all pairs
# =============================================================================

def judge_all_pairs(
    qa_pairs: List[Dict[str, Any]],
    sources: Dict[str, str],
    client: OpenAI,
) -> List[Dict[str, Any]]:

    rows: List[Dict[str, Any]] = []

    for i, pair in enumerate(qa_pairs):
        question  = pair.get("question", "")
        answer    = pair.get("reference_answer", "")
        evidence  = pair.get("source_evidence", [])
        q_type    = pair.get("question_type", "unknown")
        filename  = evidence[0].get("file", "unknown") if evidence else "unknown"

        logger.info(f"[{i+1}/{len(qa_pairs)}] Judging: {question[:70]}…")

        source_text = get_source_text(sources, evidence)
        if not source_text:
            logger.warning(f"  No source text found for pair {i} — skipping")
            continue

        prompt = JUDGE_PROMPT.format(
            source=source_text,
            question=question,
            answer=answer,
        )
        raw = _call_llm(client, prompt)
        if raw is None:
            logger.warning(f"  Skipping pair {i} — LLM returned None")
            continue
        time.sleep(CONFIG["sleep_seconds"])

        metrics = parse_judge_response(raw)
        if metrics is None:
            logger.warning(f"  Could not parse judge response for pair {i}")
            continue

        row = {
            "question"               : question,
            "reference_answer"       : answer,
            "filename"               : filename,
            "question_type"          : q_type,
            **metrics,
        }
        rows.append(row)
        logger.info(
            f"  quality={metrics['quality_score']:.3f}  "
            f"Q[comp={metrics['question_completeness']} "
            f"faith={metrics['question_faithfulness']} "
            f"corr={metrics['question_correctness']}]  "
            f"A[comp={metrics['answer_completeness']} "
            f"faith={metrics['answer_faithfulness']} "
            f"corr={metrics['answer_correctness']}]"
        )

    return rows


# =============================================================================
# Hybrid sort
# =============================================================================

def hybrid_sort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort rows by a composite score that rewards:
      - high quality_score      (weight 0.60)
      - diversity in filename   (weight 0.20) — penalise repetition
      - diversity in type       (weight 0.20) — penalise repetition

    Diversity penalty: each additional row from the same file/type after
    the first adds a cumulative penalty, implemented as an ordinal rank
    within each group normalised to [0,1] then inverted.
    """
    df = df.copy().reset_index(drop=True)

    # Within-group ordinal rank (0-based) for file and type
    df["_file_seen_rank"] = df.groupby("filename").cumcount()          # 0 = first occurrence
    df["_type_seen_rank"] = df.groupby("question_type").cumcount()

    # Normalise ranks to [0, 1] (higher rank → more repetition → worse diversity)
    max_file = max(df["_file_seen_rank"].max(), 1)
    max_type = max(df["_type_seen_rank"].max(), 1)
    df["_file_div"] = 1.0 - df["_file_seen_rank"] / max_file   # 1 = unique so far
    df["_type_div"] = 1.0 - df["_type_seen_rank"] / max_type

    w_q = CONFIG["weight_quality"]
    w_f = CONFIG["weight_file_diversity"]
    w_t = CONFIG["weight_type_diversity"]

    df["composite_score"] = (
        w_q * df["quality_score"]
        + w_f * df["_file_div"]
        + w_t * df["_type_div"]
    ).round(4)

    df_sorted = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df_sorted.index.name = "qa_id"
    df_sorted.index = df_sorted.index + 1   # 1-based unique index

    # Drop internal helper columns
    df_sorted.drop(columns=["_file_seen_rank", "_type_seen_rank",
                             "_file_div", "_type_div"], inplace=True)

    return df_sorted


# =============================================================================
# Main
# =============================================================================

def run_judge_pipeline() -> None:
    logger.info("Starting QA Judge Pipeline")

    # Load inputs
    qa_input = Path(CONFIG["qa_input_path"])
    if not qa_input.exists():
        logger.error(f"QA input file not found: {qa_input}")
        return

    qa_pairs: List[Dict[str, Any]] = json.loads(qa_input.read_text(encoding="utf-8"))
    logger.info(f"Loaded {len(qa_pairs)} QA pairs")

    sources = load_rst_sources(SOURCE_DIR)
    if not sources:
        logger.error("No RST source files found — cannot judge without source text.")
        return

    client = _get_client()

    # Judge
    rows = judge_all_pairs(qa_pairs, sources, client)
    if not rows:
        logger.error("No judged rows produced — check LLM responses.")
        return

    # Raw scores dataframe
    df_scores = pd.DataFrame(rows)
    df_scores.index.name = "qa_id"
    df_scores.index = df_scores.index + 1

    scores_path = Path(CONFIG["output_scores_path"])
    df_scores.to_csv(scores_path, encoding="utf-8")
    logger.info(f"Saved raw scores → {scores_path}  ({len(df_scores)} rows)")

    # Hybrid-sorted dataframe
    df_sorted = hybrid_sort(df_scores.reset_index(drop=True))

    sorted_path = Path(CONFIG["output_sorted_path"])
    df_sorted.to_csv(sorted_path, encoding="utf-8", index_label="qa_id")
    logger.info(f"Saved sorted output → {sorted_path}  ({len(df_sorted)} rows)")

    # Summary
    logger.info("=== Summary ===")
    logger.info(f"  Total judged pairs    : {len(df_scores)}")
    logger.info(f"  Mean quality score    : {df_scores['quality_score'].mean():.3f}")
    logger.info(f"  Question faithfulness : {(df_scores['question_faithfulness']=='pass').mean()*100:.1f}% pass")
    logger.info(f"  Answer faithfulness   : {(df_scores['answer_faithfulness']  =='pass').mean()*100:.1f}% pass")
    logger.info(f"  Question correctness  : {(df_scores['question_correctness'] =='pass').mean()*100:.1f}% pass")
    logger.info(f"  Answer correctness    : {(df_scores['answer_correctness']   =='pass').mean()*100:.1f}% pass")
    logger.info(f"  Avg Q completeness    : {df_scores['question_completeness'].mean():.2f} / 5")
    logger.info(f"  Avg A completeness    : {df_scores['answer_completeness'].mean():.2f} / 5")
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    run_judge_pipeline()
