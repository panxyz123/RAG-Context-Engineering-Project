#!/usr/bin/env python3
"""
select_top20_qa_pairs.py
════════════════════════
Reads  : /home/yup031/Project1/Golden QA Pair Generation/top60_qa_pairs copy 2.json
Outputs: top20_qa_pairs_selected.json   (20 selected QA pairs)
         top20_qa_pairs_selected.csv    (human-readable summary)

What this script does
─────────────────────
1. ENRICH categories (additive, never overwrites):
     Each entry already has a "category" field in the JSON.  An LLM call
     checks whether any of the 4 standard labels also fit and appends them
     to a new "categories" list.  The original "category" value is always
     kept as the first element.

     Fixed taxonomy (may add zero, one, or several per entry):
       • procedural            – how to do / accomplish something step-by-step
       • factual_lookup        – retrieve a specific fact, value, name, or location
       • conceptual_explanation – understand why / how something works conceptually
       • comparative           – compare, contrast, or choose between ≥ 2 things

2. SELECT 20 entries satisfying (in priority order):
     Rule 2 – every category in the enriched pool has ≥ 3 samples
     Rule 3 – every unique source file has ≥ 1 sample
     Rule 4 – remaining slots filled by composite_score descending

Dependencies
────────────
    pip install openai python-dotenv
"""

import json
import csv
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

INPUT_PATH  = Path("/home/yup031/Project1/Golden QA Pair Generation/top60_qa_pairs copy 2.json")
OUTPUT_DIR  = INPUT_PATH.parent
OUTPUT_JSON = OUTPUT_DIR / "top20_qa_pairs_selected.json"
OUTPUT_CSV  = OUTPUT_DIR / "top20_qa_pairs_selected.csv"

TARGET_N       = 20
MIN_PER_CAT    = 3
LLM_MODEL      = "mistral.mistral-large-3-675b-instruct"
LLM_BATCH_SIZE = 5
LLM_RETRY      = 3
LLM_RETRY_WAIT = 5

API_KEY_FILE = "/home/yup031/Project1/api-key.txt"
API_BASE_URL = "https://tritonai-api.ucsd.edu"

VALID_LABELS = {"procedural", "factual_lookup", "conceptual_explanation", "comparative"}


# ─────────────────────────────────────────────────────────────────────────────
# LLM CLIENT
# ─────────────────────────────────────────────────────────────────────────────

def _get_client() -> OpenAI:
    api_key = Path(API_KEY_FILE).read_text().strip()
    return OpenAI(api_key=api_key, base_url=API_BASE_URL)


def _call_llm_raw(client: OpenAI, system: str, user: str) -> str | None:
    """
    Single LLM call. Returns the response text or None if content is missing.
    The TritonAI endpoint is OpenAI-compatible; system prompt is passed as a
    system-role message rather than a separate parameter.
    """
    for attempt in range(1, LLM_RETRY + 1):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                max_tokens=512,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            choice  = response.choices[0]
            content = choice.message.content
            if content is None:
                raise ValueError(
                    f"Response content is None (finish_reason={choice.finish_reason!r})"
                )
            return content.strip()

        except Exception as exc:
            print(f"  [WARN] LLM attempt {attempt}/{LLM_RETRY} failed: {exc}")
            if attempt < LLM_RETRY:
                time.sleep(LLM_RETRY_WAIT)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# LLM CATEGORY ENRICHMENT
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a QA dataset curator. Your job is to assign additional question-type \
labels to QA pairs from a fixed taxonomy. You must ONLY add labels — never \
remove or rename the existing category already assigned to each entry.

Taxonomy (pick any that genuinely fit, or none):
  procedural            – the question asks how to do or accomplish something \
step-by-step (e.g. installation, configuration, workflow).
  factual_lookup        – the question asks for a specific concrete fact, value, \
name, path, or location that can be looked up directly.
  conceptual_explanation – the question asks why something works the way it does, \
or what a concept/term means at a deeper level.
  comparative           – the question asks to compare, contrast, or choose \
between two or more things.

Rules:
  • A single entry CAN receive multiple additional labels if they genuinely fit.
  • If NO additional label fits beyond what is already assigned, return an empty \
list for that entry.
  • Base your decision solely on the question text and reference answer.
  • Do NOT invent labels outside the four listed above.
"""

USER_PROMPT_TEMPLATE = """\
Below are {n} QA entries. For each entry return ONLY a JSON array of additional \
labels to add (subset of the four labels; may be empty []).

Return a single JSON object like:
{{
  "<question_id>": ["label1", "label2"],
  "<question_id>": [],
  ...
}}

Do NOT wrap the JSON in markdown fences. Return raw JSON only.

Entries:
{entries}
"""


def _parse_llm_json(raw: str) -> dict:
    """Strip markdown fences if present, then parse JSON."""
    text = raw.strip()
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$",       "", text)
    return json.loads(text.strip())


def _call_llm_batch(entries: list[dict], client: OpenAI) -> dict[str, list[str]]:
    """
    Call the LLM for a batch of entries.
    Returns {question_id_str: [additional_labels]}.
    """
    entries_text = "\n\n".join(
        f'ID {e["question_id"]}\n'
        f'Existing category: {e.get("category", "")}\n'
        f'Question: {e["question"]}\n'
        f'Answer: {e["reference_answer"]}'
        for e in entries
    )
    prompt = USER_PROMPT_TEMPLATE.format(n=len(entries), entries=entries_text)

    raw = _call_llm_raw(client, SYSTEM_PROMPT, prompt)

    if raw is None:
        print("  [WARN] All retries exhausted; no labels added for this batch.")
        return {str(e["question_id"]): [] for e in entries}

    try:
        result = _parse_llm_json(raw)
        cleaned: dict[str, list[str]] = {}
        for qid, labels in result.items():
            valid = [l for l in labels if l in VALID_LABELS]
            cleaned[str(qid)] = valid
        return cleaned
    except (json.JSONDecodeError, AttributeError) as exc:
        print(f"  [WARN] JSON parse failed: {exc}\n  Raw response: {raw[:300]}")
        return {str(e["question_id"]): [] for e in entries}


def enrich_categories(data: list[dict]) -> list[dict]:
    """
    For every entry:
      1. Start "categories" from the existing "category" field.
      2. Ask the LLM which of the 4 standard labels also apply.
      3. Append any new labels not already present.
    """
    client = _get_client()

    # Initialise "categories" from the existing "category" field
    for entry in data:
        existing = entry.get("category", "")
        if isinstance(existing, list):
            entry["categories"] = list(existing)
        elif existing:
            entry["categories"] = [existing]
        else:
            entry["categories"] = []

    total   = len(data)
    batches = [data[i: i + LLM_BATCH_SIZE] for i in range(0, total, LLM_BATCH_SIZE)]
    print(f"[INFO] Enriching categories for {total} entries "
          f"({len(batches)} batches of ≤{LLM_BATCH_SIZE}) via {LLM_MODEL}...")

    for idx, batch in enumerate(batches, 1):
        print(f"  Batch {idx}/{len(batches)} "
              f"(ids: {[e['question_id'] for e in batch]}) ...", end=" ", flush=True)

        additions = _call_llm_batch(batch, client)

        for entry in batch:
            qid      = str(entry["question_id"])
            new_labs = additions.get(qid, [])
            for lab in new_labs:
                if lab not in entry["categories"]:
                    entry["categories"].append(lab)

        added_total = sum(len(v) for v in additions.values())
        print(f"added {added_total} label(s)")

    return data


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_source_files(entry: dict) -> list[str]:
    seen: dict[str, None] = {}
    for ev in entry.get("source_evidence", []):
        f = ev.get("file", "").strip()
        if f:
            seen[f] = None
    return list(seen)


def get_score(entry: dict) -> float:
    return float(entry.get("composite_score", 0.0))


def entry_label(e: dict) -> str:
    return f"qid={e['question_id']} score={get_score(e):.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# GREEDY SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def select(data: list[dict]) -> list[dict]:
    all_categories = sorted({c for e in data for c in e["categories"]})
    all_files      = sorted({f for e in data for f in get_source_files(e)})

    print(f"\n[INFO] Pool : {len(data)} entries")
    print(f"[INFO] Cats : {all_categories}")
    print(f"[INFO] Files: {all_files}\n")

    pool_sorted = sorted(data, key=get_score, reverse=True)

    selected: list[dict]  = []
    selected_ids: set     = set()

    def add(entry: dict, reason: str):
        selected.append(entry)
        selected_ids.add(entry["question_id"])
        print(f"  + {entry_label(entry)}  cats={entry['categories']}  [{reason}]")

    def remaining() -> list[dict]:
        return [e for e in pool_sorted if e["question_id"] not in selected_ids]

    def cat_count(cat: str) -> int:
        return sum(1 for e in selected if cat in e["categories"])

    def covered_files() -> set[str]:
        return {f for e in selected for f in get_source_files(e)}

    def best_for(predicate) -> dict | None:
        for e in pool_sorted:
            if e["question_id"] not in selected_ids and predicate(e):
                return e
        return None

    # ── Pass 1: guarantee every source file is represented ───────────────────
    print("── Pass 1: file coverage ────────────────────────────────")
    for fname in all_files:
        if fname in covered_files():
            continue
        candidate = best_for(lambda e, f=fname: f in get_source_files(e))
        if candidate:
            add(candidate, f"covers file '{fname}'")
        else:
            print(f"  [WARN] No free entry found for file '{fname}'")

    # ── Pass 2: guarantee every category has ≥ MIN_PER_CAT samples ──────────
    print(f"\n── Pass 2: category coverage (≥{MIN_PER_CAT} each) ──────────────")
    for cat in all_categories:
        while cat_count(cat) < MIN_PER_CAT:
            if len(selected) >= TARGET_N:
                print(f"  [WARN] Hit TARGET_N before satisfying '{cat}' "
                      f"(have {cat_count(cat)})")
                break
            candidate = best_for(lambda e, c=cat: c in e["categories"])
            if candidate:
                add(candidate, f"boosts '{cat}' → {cat_count(cat) + 1}")
            else:
                avail = sum(1 for e in data if cat in e["categories"])
                print(f"  [WARN] Pool only has {avail} entries for '{cat}'; "
                      f"cannot reach {MIN_PER_CAT}")
                break

    # ── Pass 3: fill remaining slots by score ────────────────────────────────
    print(f"\n── Pass 3: score fill (target={TARGET_N}) ───────────────────────")
    for e in remaining():
        if len(selected) >= TARGET_N:
            break
        add(e, "score fill")

    print(f"\n[INFO] Selected {len(selected)} / {TARGET_N} entries.")
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION REPORT
# ─────────────────────────────────────────────────────────────────────────────

def validate(selected: list[dict], full_data: list[dict]):
    print("\n══ Validation Report ════════════════════════════════════")

    all_cats  = sorted({c for e in full_data for c in e["categories"]})
    all_files = sorted({f for e in full_data for f in get_source_files(e)})

    cat_counts: dict[str, int] = defaultdict(int)
    for e in selected:
        for c in e["categories"]:
            cat_counts[c] += 1

    covered = {f for e in selected for f in get_source_files(e)}

    print(f"\nCategory counts (need ≥ {MIN_PER_CAT}):")
    all_ok = True
    for cat in all_cats:
        n   = cat_counts.get(cat, 0)
        sym = "✓" if n >= MIN_PER_CAT else "✗"
        print(f"  {sym}  {cat:<35} {n}")
        if n < MIN_PER_CAT:
            all_ok = False

    print(f"\nFile coverage (need ≥ 1):")
    for f in all_files:
        sym = "✓" if f in covered else "✗"
        print(f"  {sym}  {f}")
        if f not in covered:
            all_ok = False

    scores = [get_score(e) for e in selected]
    print(f"\nScores — min:{min(scores):.4f}  "
          f"mean:{sum(scores)/len(scores):.4f}  max:{max(scores):.4f}")
    print(f"\nAll constraints satisfied: {'YES ✓' if all_ok else 'NO  ✗ — see above'}")
    print("═════════════════════════════════════════════════════════\n")


# ─────────────────────────────────────────────────────────────────────────────
# WRITE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def write_outputs(selected: list[dict]):
    final = sorted(selected, key=get_score, reverse=True)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    print(f"[OUT] JSON → {OUTPUT_JSON}")

    fields = ["rank", "question_id", "composite_score",
              "original_category", "categories", "source_files", "question_preview"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rank, e in enumerate(final, 1):
            w.writerow({
                "rank":              rank,
                "question_id":       e.get("question_id", ""),
                "composite_score":   f"{get_score(e):.4f}",
                "original_category": e.get("category", ""),
                "categories":        " | ".join(e.get("categories", [])),
                "source_files":      " | ".join(get_source_files(e)),
                "question_preview":  e.get("question", "")[:120],
            })
    print(f"[OUT] CSV  → {OUTPUT_CSV}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not INPUT_PATH.exists():
        sys.exit(f"[ERROR] File not found:\n  {INPUT_PATH}")

    with open(INPUT_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    # Accept a bare list or a dict wrapper like {"data": [...]}
    if isinstance(raw, list):
        data = raw
    elif isinstance(raw, dict):
        for key in ("data", "qa_pairs", "pairs", "items"):
            if key in raw and isinstance(raw[key], list):
                data = raw[key]
                break
        else:
            data = list(raw.values())[0]
    else:
        sys.exit("[ERROR] Unexpected JSON root type.")

    if not data:
        sys.exit("[ERROR] Parsed an empty list — check the file.")

    print(f"[INFO] Loaded {len(data)} entries from '{INPUT_PATH.name}'")

    # Step 1 — additive LLM enrichment
    data = enrich_categories(data)

    # Steps 2–4 — greedy selection
    selected = select(data)

    # Report
    validate(selected, data)

    # Save
    write_outputs(selected)


if __name__ == "__main__":
    main()