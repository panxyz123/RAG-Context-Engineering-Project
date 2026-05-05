"""
QA Merger & Reporter
====================
1. Take top-60 rows from judged_qa_sorted.csv
2. Match each question to its exact source_evidence and question_id
   from golden_qa_pairs_0503raw.json
3. Match each question to its category from question-type-distribution.json
4. Output:
     • top60_qa_pairs.json          — final QA records in submission format
     • top60_distribution_report.json — category + file distribution analysis
"""

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# =============================================================================
# Configuration
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

CONFIG = {
    "sorted_csv"        : BASE_DIR / "judged_qa_sorted.csv",
    "raw_json"          : BASE_DIR / "golden_qa_pairs_0503raw.json",
    "distribution_json" : BASE_DIR / "question-type-distribution.json",
    "output_qa"         : BASE_DIR / "top60_qa_pairs.json",
    "output_report"     : BASE_DIR / "top60_distribution_report.json",
    "top_n"             : 60,
}


# =============================================================================
# Loaders
# =============================================================================

def load_sorted_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalise column name — some saves use 'qa_id' as index label
    if "qa_id" not in df.columns and df.index.name == "qa_id":
        df = df.reset_index()
    logger.info(f"Loaded {len(df)} rows from {path.name}")
    return df


def load_raw_pairs(path: Path) -> List[Dict[str, Any]]:
    pairs = json.loads(path.read_text(encoding="utf-8"))
    logger.info(f"Loaded {len(pairs)} raw QA pairs from {path.name}")
    return pairs


def load_distribution(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    logger.info(f"Loaded distribution from {path.name}")
    return data


# =============================================================================
# Build lookup indexes
# =============================================================================

def build_raw_lookup(raw_pairs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Map normalised question text → raw record.
    The raw record is expected to have at minimum:
      { "question": ..., "reference_answer": ..., "source_evidence": [...] }
    question_id is the 0-based list position if not explicitly present.
    """
    lookup: Dict[str, Dict[str, Any]] = {}
    for idx, pair in enumerate(raw_pairs):
        key = _normalise(pair.get("question", ""))
        record = dict(pair)
        if "question_id" not in record:
            record["question_id"] = idx
        lookup[key] = record
    return lookup


def build_category_lookup(distribution: Dict[str, Any]) -> Dict[str, str]:
    """
    Map question_id (int) → category string, derived from
    question-type-distribution.json structure:
      { "question_type_distribution": { "<category>": { "question_ids": [...] } } }
    """
    lookup: Dict[int, str] = {}
    dist_block = distribution.get("question_type_distribution", {})
    for category, info in dist_block.items():
        for qid in info.get("question_ids", []):
            lookup[int(qid)] = category
    return lookup


def _normalise(text: str) -> str:
    """Lower-case + collapse whitespace for fuzzy-safe matching."""
    return " ".join(text.lower().split())


# =============================================================================
# Matching
# =============================================================================

def match_pairs(
    top_df: pd.DataFrame,
    raw_lookup: Dict[str, Dict[str, Any]],
    category_lookup: Dict[int, str],
) -> List[Dict[str, Any]]:
    """
    For each row in top_df, find the corresponding raw record and
    assemble the final output object.
    """
    results: List[Dict[str, Any]] = []
    unmatched: List[str] = []

    for _, row in top_df.iterrows():
        q_norm = _normalise(str(row.get("question", "")))
        raw = raw_lookup.get(q_norm)

        if raw is None:
            # Fallback: partial prefix match (first 80 chars)
            prefix = q_norm[:80]
            raw = next(
                (v for k, v in raw_lookup.items() if k.startswith(prefix)),
                None,
            )

        if raw is None:
            unmatched.append(row.get("question", "")[:80])
            logger.warning(f"  No raw match for: {row.get('question','')[:80]}…")
            continue

        question_id = int(raw.get("question_id", 0))

        # Category: prefer distribution JSON, fall back to CSV column
        category = category_lookup.get(
            question_id,
            str(row.get("question_type", "unknown")),
        )

        record = {
            "question_id"     : question_id,
            "question"        : raw.get("question", row.get("question", "")),
            "reference_answer": raw.get("reference_answer", row.get("reference_answer", "")),
            "source_evidence" : raw.get("source_evidence", []),
            "question_type"   : category,
            # Quality metadata kept for transparency
            "quality_score"   : float(row.get("quality_score", 0.0)),
            "composite_score" : float(row.get("composite_score", 0.0)),
        }
        results.append(record)

    if unmatched:
        logger.warning(f"{len(unmatched)} questions could not be matched to raw JSON")

    return results


# =============================================================================
# Distribution report
# =============================================================================

def build_report(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyse category and file distribution of the top-60 set.
    """
    category_counter: Counter = Counter()
    file_counter: Counter = Counter()
    category_file_map: Dict[str, Counter] = defaultdict(Counter)
    quality_by_category: Dict[str, List[float]] = defaultdict(list)
    quality_by_file: Dict[str, List[float]] = defaultdict(list)

    for rec in records:
        cat  = rec.get("question_type", "unknown")
        qs   = rec.get("quality_score", 0.0)
        comp = rec.get("composite_score", 0.0)
        files = [ev.get("file", "unknown") for ev in rec.get("source_evidence", [])]
        primary_file = files[0] if files else "unknown"

        category_counter[cat] += 1
        file_counter[primary_file] += 1
        category_file_map[cat][primary_file] += 1
        quality_by_category[cat].append(qs)
        quality_by_file[primary_file].append(qs)

    total = len(records)

    # Per-category stats
    category_stats = {}
    for cat, count in sorted(category_counter.items(), key=lambda x: -x[1]):
        qs_list = quality_by_category[cat]
        category_stats[cat] = {
            "count"          : count,
            "percentage"     : round(count / total * 100, 1),
            "avg_quality"    : round(sum(qs_list) / len(qs_list), 4),
            "file_breakdown" : dict(category_file_map[cat]),
        }

    # Per-file stats
    file_stats = {}
    for fname, count in sorted(file_counter.items(), key=lambda x: -x[1]):
        qs_list = quality_by_file[fname]
        file_stats[fname] = {
            "count"      : count,
            "percentage" : round(count / total * 100, 1),
            "avg_quality": round(sum(qs_list) / len(qs_list), 4),
        }

    # Overall
    all_qs = [r.get("quality_score", 0.0) for r in records]
    overall = {
        "total_pairs"           : total,
        "avg_quality_score"     : round(sum(all_qs) / len(all_qs), 4) if all_qs else 0.0,
        "min_quality_score"     : round(min(all_qs), 4) if all_qs else 0.0,
        "max_quality_score"     : round(max(all_qs), 4) if all_qs else 0.0,
        "unique_categories"     : len(category_counter),
        "unique_source_files"   : len(file_counter),
    }

    return {
        "overall"           : overall,
        "category_distribution" : category_stats,
        "file_distribution" : file_stats,
    }


# =============================================================================
# Main
# =============================================================================

def run_merge_pipeline() -> None:
    logger.info("Starting QA Merge & Report Pipeline")

    # Load all inputs
    df_sorted   = load_sorted_csv(Path(CONFIG["sorted_csv"]))
    raw_pairs   = load_raw_pairs(Path(CONFIG["raw_json"]))
    distribution = load_distribution(Path(CONFIG["distribution_json"]))

    # Take top N
    top_df = df_sorted.head(CONFIG["top_n"]).copy()
    logger.info(f"Processing top {len(top_df)} rows")

    # Build lookups
    raw_lookup      = build_raw_lookup(raw_pairs)
    category_lookup = build_category_lookup(distribution)

    # Match
    records = match_pairs(top_df, raw_lookup, category_lookup)
    logger.info(f"Successfully matched {len(records)} / {len(top_df)} pairs")

    # Sort final output by question_id for clean indexing
    records.sort(key=lambda r: r["question_id"])

    # Save QA JSON
    qa_out = Path(CONFIG["output_qa"])
    qa_out.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved → {qa_out}")

    # Save report
    report = build_report(records)
    report_out = Path(CONFIG["output_report"])
    report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved → {report_out}")

    # Print summary to console
    ov = report["overall"]
    logger.info("=== Report Summary ===")
    logger.info(f"  Pairs            : {ov['total_pairs']}")
    logger.info(f"  Avg quality      : {ov['avg_quality_score']}")
    logger.info(f"  Quality range    : {ov['min_quality_score']} – {ov['max_quality_score']}")
    logger.info(f"  Categories       : {ov['unique_categories']}")
    logger.info(f"  Source files     : {ov['unique_source_files']}")
    logger.info("  Category breakdown:")
    for cat, stats in report["category_distribution"].items():
        logger.info(f"    {cat:<30} {stats['count']:>3}  ({stats['percentage']}%)  avg_q={stats['avg_quality']}")
    logger.info("  File breakdown:")
    for fname, stats in report["file_distribution"].items():
        logger.info(f"    {fname:<35} {stats['count']:>3}  ({stats['percentage']}%)  avg_q={stats['avg_quality']}")

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    run_merge_pipeline()
