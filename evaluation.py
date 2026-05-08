"""evaluation.py — Compute all eval metrics from a predictions JSON file.

Mirrors the metric computation in sample_compute_metrics_fn from the notebook.
Retrieval metrics use project1_eval.py. Generation metrics use LLM-as-judge.

Usage:
    python evaluation.py \
        --output predictions.json \
        --ground-truth golden_qa_pairs.json \
        --apikey-txt api-key.txt \
        [--judge-model api-gpt-oss-120b] \
        [--metrics-output metrics.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

from project1_eval import (
    to_spans,
    precision_at_k,
    recall_at_k,
    f1_at_k,
    call_judge,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TRITON_BASE_URL = "https://tritonai-api.ucsd.edu"
K = 5


def load_api_key(apikey_txt: str) -> str:
    return Path(apikey_txt).expanduser().read_text(encoding="utf-8").splitlines()[0].strip()


def build_qrels(golden_qa_pairs: list[dict]) -> dict[int, dict]:
    """Index golden QA pairs by question_id for O(1) lookup."""
    index = {}
    for qa in golden_qa_pairs:
        qid = int(qa["question_id"])
        gt_spans = []
        for evidence in qa.get("source_evidence", []):
            fname      = Path(evidence.get("file", "")).name
            raw_lines  = evidence.get("lines", [])
            if len(raw_lines) >= 2:
                start_line, end_line = int(min(raw_lines)), int(max(raw_lines))
            elif len(raw_lines) == 1:
                start_line = end_line = int(raw_lines[0])
            else:
                continue
            gt_spans.append((fname, start_line, end_line))
        index[qid] = {
            "reference_answer": qa.get("reference_answer", ""),
            "gt_spans":         to_spans(gt_spans),
            "question":         qa.get("question", ""),
        }
    return index


def run(
    output_path: str,
    ground_truth_path: str,
    apikey_txt: str,
    judge_model: str,
    metrics_output: str,
) -> None:
    with open(output_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    with open(ground_truth_path, "r", encoding="utf-8") as f:
        golden_qa_pairs = json.load(f)

    api_key = load_api_key(apikey_txt)
    qrels   = build_qrels(golden_qa_pairs)

    log.info("Evaluating %d predictions against %d ground-truth QA pairs",
             len(predictions), len(golden_qa_pairs))

    precisions, recalls, f1s = [], [], []
    correctness_scores, faithfulness_scores, completeness_scores = [], [], []
    judge_failures = 0
    per_example: list[dict[str, Any]] = []

    for pred in predictions:
        qid     = int(pred["question_id"])
        answer  = str(pred.get("answer", ""))
        context = str(pred.get("retrieved_context", ""))

        # ── Retrieval metrics ─────────────────────────────────────────────────
        raw_sources = pred.get("sources", [])
        try:
            pred_spans = to_spans([
                (src["file"], int(src["lines"][0]), int(src["lines"][1]))
                for src in raw_sources
            ])
        except (KeyError, IndexError, ValueError) as e:
            log.warning("qid=%d malformed sources: %s", qid, e)
            pred_spans = []

        gt_entry = qrels.get(qid, {})
        gt_spans = gt_entry.get("gt_spans", [])

        p = precision_at_k(pred_spans, gt_spans, k=K)
        r = recall_at_k(pred_spans,    gt_spans, k=K)
        f = f1_at_k(pred_spans,        gt_spans, k=K)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

        # ── Generation metrics (LLM-as-judge) ─────────────────────────────────
        ref_ans  = gt_entry.get("reference_answer", "")
        question = gt_entry.get("question", "")

        if not question:
            log.warning("qid=%d not found in ground truth, skipping judge", qid)
            correctness_scores.append(0.0)
            faithfulness_scores.append(0.0)
            completeness_scores.append(0.0)
            judge_failures += 1
        else:
            judge = call_judge(
                query=question,
                reference_answer=ref_ans,
                retrieved_context=context,
                system_answer=answer,
                model=judge_model,
                base_url=TRITON_BASE_URL,
                api_key=api_key,
            )
            if judge.get("failed"):
                judge_failures += 1
                log.warning("qid=%d judge failed: %s", qid, judge.get("error", ""))
            correctness_scores.append(judge["correctness"])
            faithfulness_scores.append(judge["faithfulness"])
            completeness_scores.append(judge["completeness"])

        per_example.append({
            "question_id":   qid,
            "precision":     p,
            "recall":        r,
            "f1":            f,
            "correctness":   correctness_scores[-1],
            "faithfulness":  faithfulness_scores[-1],
            "completeness":  completeness_scores[-1],
        })

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    n = len(predictions)
    avg_precision    = mean(precisions)
    avg_recall       = mean(recalls)
    avg_f1           = mean(f1s)
    avg_correctness  = mean(correctness_scores)
    avg_faithfulness = mean(faithfulness_scores)
    avg_completeness = mean(completeness_scores)

    retrieval_score   = (avg_f1 + avg_precision + avg_recall) / 3
    generation_score  = (avg_correctness + avg_faithfulness + avg_completeness / 5) / 3
    leaderboard_score = 0.5 * retrieval_score + 0.5 * generation_score

    summary: dict[str, Any] = {
        "Total":            n,
        f"Precision_at{K}": avg_precision,
        f"Recall_at{K}":    avg_recall,
        f"F1_at{K}":        avg_f1,
        "Correctness":      avg_correctness,
        "Faithfulness":     avg_faithfulness,
        "Completeness":     avg_completeness,
        "JudgeFailures":    judge_failures,
        "RetrievalScore":   retrieval_score,
        "GenerationScore":  generation_score,
        "LeaderboardScore": leaderboard_score,
    }

    # ── Print all metrics ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for key, val in summary.items():
        if isinstance(val, float):
            print(f"  {key:<25} {val:.4f}")
        else:
            print(f"  {key:<25} {val}")
    print("=" * 60 + "\n")

    # ── Write metrics JSON ────────────────────────────────────────────────────
    output: dict[str, Any] = {
        "summary":     summary,
        "per_example": per_example,
    }
    with open(metrics_output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log.info("Metrics written to %s", metrics_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline predictions")
    parser.add_argument("--output",         required=True,
                        help="Predictions JSON from main.py")
    parser.add_argument("--ground-truth",   required=True,
                        help="Golden QA pairs JSON (validation-set-golden-qa-pairs.json)")
    parser.add_argument("--apikey-txt",     required=True,
                        help="Path to TritonAI API key text file")
    parser.add_argument("--judge-model",    default="api-gpt-oss-120b",
                        help="LLM model ID for judge (default: api-gpt-oss-120b)")
    parser.add_argument("--metrics-output", default="metrics.json",
                        help="Where to write metrics JSON (default: metrics.json)")
    args = parser.parse_args()
    run(args.output, args.ground_truth, args.apikey_txt,
        args.judge_model, args.metrics_output)
