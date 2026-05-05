"""
Multi-File QA Judge
===================
Evaluates multi_file_qa_pairs.json on four metrics:

  Question metrics:
    • completeness  : 1–5  (covers all relevant info across all source snippets)
    • faithfulness  : pass/fail  (no hallucination vs. source texts)
    • correctness   : pass/fail  (factually accurate given sources)
    • non_trivial   : pass/fail  (genuinely requires multiple sources to answer;
                                  not answerable from any single snippet alone)

  Answer metrics:
    • completeness  : 1–5
    • faithfulness  : pass/fail
    • correctness   : pass/fail

quality_score = mean of all 7 normalised sub-scores

Output:
    multi_file_qa_judged.json  — sorted by quality_score descending,
                                  same shape as input + quality fields
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from dotenv import load_dotenv

# =============================================================================
# Configuration
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
load_dotenv()

INPUT_PATH  = Path("/home/yup031/Project1/Golden QA Pair Generation/multi_file_qa_pairs.json")
OUTPUT_PATH = Path("/home/yup031/Project1/Golden QA Pair Generation/multi_file_qa_judged.json")
SOURCE_DIR  = Path("/home/yup031/Project1/sourcedocs")

CONFIG = {
    "llm_model"       : "mistral.mistral-large-3-675b-instruct",
    "llm_temperature" : 0.1,
    "llm_max_tokens"  : 900,
    "sleep_seconds"   : 1,
    "api_key_file"    : "/home/yup031/Project1/api-key.txt",
    "api_base_url"    : "https://tritonai-api.ucsd.edu",
}

# =============================================================================
# Source loader
# =============================================================================

def load_rst_sources(source_dir: Path) -> Dict[str, List[str]]:
    sources: Dict[str, List[str]] = {}
    for rst_file in sorted(source_dir.glob("*.rst")):
        sources[rst_file.name] = (
            rst_file.read_text(encoding="utf-8").replace("\r\n", "\n").splitlines()
        )
    logger.info(f"Loaded {len(sources)} RST source files")
    return sources


def get_source_snippets(
    sources: Dict[str, List[str]], evidence: List[Dict]
) -> List[Dict[str, str]]:
    """Return [{'file': ..., 'text': ...}, ...] for each evidence entry."""
    snippets = []
    seen: set = set()
    for ev in evidence:
        fname = ev.get("file", "")
        if fname in seen:
            continue
        file_lines = sources.get(fname, [])
        lines = ev.get("lines", [])
        if lines and len(lines) == 2:
            start = max(0, lines[0] - 1)
            end   = min(len(file_lines), lines[1])
            text  = "\n".join(file_lines[start:end])
        else:
            text = "\n".join(file_lines)
        if text.strip():
            snippets.append({"file": fname, "text": text})
            seen.add(fname)
    return snippets


# =============================================================================
# LLM client
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
    choice  = response.choices[0]
    content = choice.message.content
    if content is None:
        logger.warning(f"LLM returned None (finish_reason={choice.finish_reason!r})")
        return None
    return content.strip()


# =============================================================================
# Judge prompt
# =============================================================================

JUDGE_PROMPT = """You are a strict evaluator for a multi-source RAG evaluation dataset. Be critical — most pairs have flaws.

You are given SOURCE SNIPPETS from MULTIPLE different files, a QUESTION, and an ANSWER.
Base ALL judgements solely on the provided source snippets. Do not use outside knowledge.

━━━ METRICS ━━━

QUESTION metrics:

1. completeness (integer 1–5)
   Does the question address the full scope of what the combined sources discuss?
   5 = question captures the complete cross-file relationship or contrast
   3 = addresses the main theme but ignores important detail from at least one source
   1 = vague, superficial, or misses the cross-file angle entirely

2. faithfulness (pass | fail)
   pass = the question only refers to concepts actually present in the sources
   fail = the question introduces concepts, entities, or claims not in any source

3. correctness (pass | fail)
   pass = the question accurately represents what the sources say
   fail = the question mischaracterises or contradicts the source content

4. non_trivial (pass | fail)
   pass = the question CANNOT be fully answered from any single snippet alone;
          it genuinely requires synthesising information across multiple files
   fail = one snippet is sufficient to answer it, making the multi-file framing
          unnecessary or cosmetic

ANSWER metrics:

5. completeness (integer 1–5)
   Does the answer cover all relevant points from ALL source snippets?
   5 = explicitly draws on every snippet and addresses all key points
   3 = covers main points but noticeably omits content from at least one snippet
   1 = shallow, or relies on only one snippet despite multiple being available

6. faithfulness (pass | fail)
   pass = every claim traces back to the source snippets; nothing is invented
   fail = contains ANY information not present in the snippets

7. correctness (pass | fail)
   pass = no factual errors relative to what the snippets state
   fail = contradicts or misrepresents anything in the snippets

━━━ COMMON FAILURE PATTERNS — actively check for these ━━━
  - non_trivial FAIL: the answer could be copy-pasted from one snippet alone
  - non_trivial FAIL: question asks "what is X" where X is defined in one file only
  - faithfulness FAIL: answer adds background context not in any snippet
  - completeness <=2: answer summarises only the longest snippet and ignores others
  - correctness FAIL: answer merges facts from two snippets incorrectly

━━━ SOURCE SNIPPETS ━━━
{snippets_block}

QUESTION:
{question}

ANSWER:
{answer}

Think step by step:
1. Identify the key facts each snippet contributes independently.
2. Check whether the question truly requires ALL snippets.
3. Check whether the answer draws on ALL snippets.
4. Then assign scores.

Respond with valid JSON only, no markdown fences, no extra text:
{{
  "question": {{
    "completeness" : <1-5>,
    "faithfulness" : "pass or fail",
    "correctness"  : "pass or fail",
    "non_trivial"  : "pass or fail",
    "reasoning"    : "<one concise sentence>"
  }},
  "answer": {{
    "completeness" : <1-5>,
    "faithfulness" : "pass or fail",
    "correctness"  : "pass or fail",
    "reasoning"    : "<one concise sentence>"
  }}
}}
"""


def build_judge_prompt(
    snippets: List[Dict[str, str]], question: str, answer: str
) -> str:
    snippets_block = "\n\n".join(
        f"[{s['file']}]\n\"\"\"\n{s['text']}\n\"\"\""
        for s in snippets
    )
    return JUDGE_PROMPT.format(
        snippets_block=snippets_block,
        question=question,
        answer=answer,
    )


# =============================================================================
# Parse & score
# =============================================================================

def _pf(value: str) -> float:
    return 1.0 if str(value).strip().lower() == "pass" else 0.0


def _comp(value: Any) -> float:
    try:
        return max(0.0, min(1.0, (float(value) - 1) / 4.0))
    except (TypeError, ValueError):
        return 0.0


def parse_judge_response(raw: str) -> Optional[Dict[str, Any]]:
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        data = json.loads(match.group(0))

        q = data.get("question", {})
        a = data.get("answer",   {})

        result: Dict[str, Any] = {
            "question_completeness" : int(q.get("completeness", 1)),
            "question_faithfulness" : str(q.get("faithfulness", "fail")).lower(),
            "question_correctness"  : str(q.get("correctness",  "fail")).lower(),
            "question_non_trivial"  : str(q.get("non_trivial",  "fail")).lower(),
            "question_reasoning"    : str(q.get("reasoning",    "")).strip(),
            "answer_completeness"   : int(a.get("completeness", 1)),
            "answer_faithfulness"   : str(a.get("faithfulness", "fail")).lower(),
            "answer_correctness"    : str(a.get("correctness",  "fail")).lower(),
            "answer_reasoning"      : str(a.get("reasoning",    "")).strip(),
        }

        # 7 normalised sub-scores (question: 4, answer: 3)
        sub_scores = [
            _comp(result["question_completeness"]),
            _pf(result["question_faithfulness"]),
            _pf(result["question_correctness"]),
            _pf(result["question_non_trivial"]),
            _comp(result["answer_completeness"]),
            _pf(result["answer_faithfulness"]),
            _pf(result["answer_correctness"]),
        ]
        result["quality_score"] = round(sum(sub_scores) / len(sub_scores), 4)
        return result

    except (json.JSONDecodeError, AttributeError, KeyError):
        return None


# =============================================================================
# Main pipeline
# =============================================================================

def run_judge_pipeline() -> None:
    logger.info("Starting Multi-File QA Judge Pipeline")

    if not INPUT_PATH.exists():
        logger.error(f"Input not found: {INPUT_PATH}")
        return

    pairs: List[Dict[str, Any]] = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    logger.info(f"Loaded {len(pairs)} multi-file QA pairs")

    sources = load_rst_sources(SOURCE_DIR)
    client  = _get_client()

    judged: List[Dict[str, Any]] = []
    skipped = 0

    for i, pair in enumerate(pairs):
        question = pair.get("question", "")
        answer   = pair.get("reference_answer", "")
        evidence = pair.get("source_evidence", [])

        logger.info(f"[{i+1}/{len(pairs)}] {question[:75]}...")

        snippets = get_source_snippets(sources, evidence)
        if len(snippets) < 2:
            logger.warning("  Fewer than 2 snippets resolved — skipping")
            skipped += 1
            continue

        prompt = build_judge_prompt(snippets, question, answer)
        raw    = _call_llm(client, prompt)
        time.sleep(CONFIG["sleep_seconds"])

        if raw is None:
            logger.warning("  LLM returned None — skipping")
            skipped += 1
            continue

        metrics = parse_judge_response(raw)
        if metrics is None:
            logger.warning("  Could not parse response — skipping")
            skipped += 1
            continue

        logger.info(
            f"  quality={metrics['quality_score']:.3f}  "
            f"non_trivial={metrics['question_non_trivial']}  "
            f"Q[comp={metrics['question_completeness']} "
            f"faith={metrics['question_faithfulness']} "
            f"corr={metrics['question_correctness']}]  "
            f"A[comp={metrics['answer_completeness']} "
            f"faith={metrics['answer_faithfulness']} "
            f"corr={metrics['answer_correctness']}]"
        )
        logger.info(f"  Q-reason: {metrics['question_reasoning']}")
        logger.info(f"  A-reason: {metrics['answer_reasoning']}")

        judged.append({
            # Core output fields
            "question"              : question,
            "reference_answer"      : answer,
            "question_type"         : "multi_file_synthesis",
            "source_evidence"       : evidence,
            # Evaluation detail
            "question_completeness" : metrics["question_completeness"],
            "question_faithfulness" : metrics["question_faithfulness"],
            "question_correctness"  : metrics["question_correctness"],
            "question_non_trivial"  : metrics["question_non_trivial"],
            "question_reasoning"    : metrics["question_reasoning"],
            "answer_completeness"   : metrics["answer_completeness"],
            "answer_faithfulness"   : metrics["answer_faithfulness"],
            "answer_correctness"    : metrics["answer_correctness"],
            "answer_reasoning"      : metrics["answer_reasoning"],
            "quality_score"         : metrics["quality_score"],
        })

    # Sort by quality score descending, assign rank-based qa_id
    judged.sort(key=lambda x: x["quality_score"], reverse=True)
    for rank, rec in enumerate(judged, start=1):
        rec["qa_id"] = rank

    logger.info(f"Judged {len(judged)} pairs  ({skipped} skipped)")

    if not judged:
        logger.warning("Nothing to save.")
        return

    OUTPUT_PATH.write_text(json.dumps(judged, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved -> {OUTPUT_PATH}")

    # Summary
    qs      = [r["quality_score"] for r in judged]
    nt_pass = sum(1 for r in judged if r["question_non_trivial"] == "pass")
    qf_pass = sum(1 for r in judged if r["question_faithfulness"] == "pass")
    af_pass = sum(1 for r in judged if r["answer_faithfulness"]   == "pass")
    logger.info("=== Summary ===")
    logger.info(f"  Total judged        : {len(judged)}")
    logger.info(f"  Avg quality score   : {sum(qs)/len(qs):.3f}")
    logger.info(f"  Quality range       : {min(qs):.3f} - {max(qs):.3f}")
    logger.info(f"  Non-trivial pass    : {nt_pass}/{len(judged)} ({nt_pass/len(judged)*100:.1f}%)")
    logger.info(f"  Q faithfulness pass : {qf_pass}/{len(judged)}")
    logger.info(f"  A faithfulness pass : {af_pass}/{len(judged)}")
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    run_judge_pipeline()