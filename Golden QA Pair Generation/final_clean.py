"""
clean_qa_pairs.py
=================
Reads  : top20_qa_pairs_selected.json
Outputs: qa_pairs_clean.json        — minimal format (question_id, question,
                                       reference_answer, source_evidence)
         qa_category_analysis.json  — category distribution analysis
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR     = Path("/home/yup031/Project1/Golden QA Pair Generation")
INPUT_PATH   = BASE_DIR / "[FINAL] QA Pairs.json"
OUTPUT_CLEAN = BASE_DIR / "[FINAL] Golden_QA_Pairs.json"
OUTPUT_ANALYSIS = BASE_DIR / "qa_category_analysis.json"

# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

with open(INPUT_PATH, encoding="utf-8") as f:
    data = json.load(f)

if isinstance(data, dict):
    for key in ("data", "qa_pairs", "pairs", "items"):
        if key in data:
            data = data[key]
            break

total = len(data)
print(f"Loaded {total} entries")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_labels(entry):
    cats = entry.get("categories")
    if cats and isinstance(cats, list):
        return cats
    single = entry.get("category") or entry.get("question_type", "")
    return [single] if single else []


def get_files(entry):
    return list({
        ev.get("file", "")
        for ev in entry.get("source_evidence", [])
        if ev.get("file")
    })


# ─────────────────────────────────────────────────────────────────────────────
# Category analysis
# ─────────────────────────────────────────────────────────────────────────────

label_counter = Counter()
label_to_ids  = defaultdict(list)
multi_label_count = 0

for entry in data:
    qid    = entry.get("question_id", "?")
    labels = get_labels(entry)
    if len(labels) > 1:
        multi_label_count += 1
    for lab in labels:
        label_counter[lab] += 1
        label_to_ids[lab].append(qid)

# ─────────────────────────────────────────────────────────────────────────────
# File analysis
# ─────────────────────────────────────────────────────────────────────────────

file_counter = Counter()
file_to_ids  = defaultdict(list)
multi_file_count = 0

for entry in data:
    qid   = entry.get("question_id", "?")
    files = get_files(entry)
    if len(files) > 1:
        multi_file_count += 1
    for fname in files:
        file_counter[fname] += 1
        file_to_ids[fname].append(qid)

# ─────────────────────────────────────────────────────────────────────────────
# Assemble and save analysis
# ─────────────────────────────────────────────────────────────────────────────

analysis = {
    "total_pairs": total,
    "category_analysis": {
        "unique_labels"    : len(label_counter),
        "multi_label_pairs": multi_label_count,
        "label_distribution": {
            lab: {
                "count"       : cnt,
                "percentage"  : round(cnt / total * 100, 1),
                "question_ids": label_to_ids[lab],
            }
            for lab, cnt in sorted(label_counter.items(), key=lambda x: -x[1])
        },
    },
    "file_analysis": {
        "unique_files"    : len(file_counter),
        "multi_file_pairs": multi_file_count,
        "file_distribution": {
            fname: {
                "count"       : cnt,
                "percentage"  : round(cnt / total * 100, 1),
                "question_ids": file_to_ids[fname],
            }
            for fname, cnt in sorted(file_counter.items(), key=lambda x: -x[1])
        },
    },
}

with open(OUTPUT_ANALYSIS, "w", encoding="utf-8") as f:
    json.dump(analysis, f, indent=2, ensure_ascii=False)

# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

print(f"\nCategory analysis ({len(label_counter)} unique labels):")
for lab, info in analysis["category_analysis"]["label_distribution"].items():
    print(f"  {lab:<35} {info['count']:>3}  ({info['percentage']}%)")
print(f"  Multi-label pairs : {multi_label_count}")

print(f"\nFile analysis ({len(file_counter)} unique files):")
for fname, info in analysis["file_analysis"]["file_distribution"].items():
    print(f"  {fname:<35} {info['count']:>3}  ({info['percentage']}%)")
print(f"  Multi-file pairs  : {multi_file_count}")

# ─────────────────────────────────────────────────────────────────────────────
# Clean output — keep only the four required fields
# ─────────────────────────────────────────────────────────────────────────────

clean = [
    {
        "question_id"     : entry.get("question_id"),
        "question"        : entry.get("question", ""),
        "reference_answer": entry.get("reference_answer", ""),
        "source_evidence" : entry.get("source_evidence", []),
    }
    for entry in data
]

with open(OUTPUT_CLEAN, "w", encoding="utf-8") as f:
    json.dump(clean, f, indent=2, ensure_ascii=False)

print(f"\nSaved {len(clean)} clean pairs  → {OUTPUT_CLEAN}")
print(f"Saved analysis              → {OUTPUT_ANALYSIS}")