"""
Golden Q&A Dataset Generator for RAG Project
============================================
Rewritten for category-first sampling, section-based chunking, and LLM quality gating.
"""

import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
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
API_KEY_FILE = "/home/yup031/Project1/api-key.txt"

CONFIG = {
    "output_qa_path": "golden_qa_pairs.json",
    "output_distribution_path": "question-type-distribution.json",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "target_pairs": 40,
    "llm_model": "mistral.mistral-large-3-675b-instruct",
    "llm_temperature": 0.2,
    "llm_max_tokens": 700,
    "max_quality_attempts": 100,
    "sleep_seconds": 1,
}

PERSONAS = ["Novice User", "System Architect", "ML Engineer"]
QUESTION_TYPES = [
    "factual_lookup",
    "conceptual_explanation",
    "procedural",
    "comparative",
    "multi_file_synthesis"
]
HEADER_KEYWORDS = [
    "Architecture", "Definition", "Overview", "Concept", "Design", "Purpose", "Introduction"
]


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
            "lines": [self.start_line, self.end_line]
        }


# =============================================================================
# RST File Loading & Section-Based Chunking
# =============================================================================

import re
from pathlib import Path
from typing import List

def load_rst_files(source_dir: Path) -> List[TextChunk]:
    chunks: List[TextChunk] = []
    rst_files = sorted(source_dir.glob("*.rst"))
    
    for rst_file in rst_files:
        # 统一换行符并保留空行，不要 strip()，否则行号会乱
        text = rst_file.read_text(encoding='utf-8').replace('\r\n', '\n')
        lines = text.splitlines()
        
        current_buffer = []
        start_line = 1
        empty_count = 0
        
        for idx, line in enumerate(lines):
            is_empty = not line.strip()
            
            if is_empty:
                empty_count += 1
                # 只有遇到连续 2 个及以上的空行，且 buffer 不为空时才触发分块
                if empty_count >= 2 and current_buffer:
                    # 这里的 end_line 需要减去刚才计入的空行
                    chunk_content = "\n".join(current_buffer).strip()
                    if chunk_content:
                        end_line = idx - 1 # 上一行是第二个空行，再上一行是内容结束
                        chunks.append(TextChunk(
                            content=chunk_content,
                            filename=rst_file.name,
                            start_line=start_line,
                            end_line=end_line,
                            source_path=str(rst_file)
                        ))
                    current_buffer = []
                    start_line = idx + 2 # 跳过当前空行，从下一行开始算
            else:
                # 遇到非空行，如果之前有一个空行，记得补回 buffer 保持行号对齐
                if empty_count == 1:
                    current_buffer.append("")
                empty_count = 0
                current_buffer.append(line)

        # 处理最后一个块
        if current_buffer:
            chunk_content = "\n".join(current_buffer).strip()
            if chunk_content:
                chunks.append(TextChunk(
                    content=chunk_content,
                    filename=rst_file.name,
                    start_line=start_line,
                    end_line=len(lines),
                    source_path=str(rst_file)
                ))
    logger.info(f"Created {len(chunks)} chunks")
    return chunks


# =============================================================================
# Embeddings & Similarity
# =============================================================================

def build_embeddings(chunks: List[TextChunk], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    texts = [chunk.content for chunk in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings


def cosine_similarity(embeddings: np.ndarray, anchor_idx: int) -> np.ndarray:
    return embeddings @ embeddings[anchor_idx]


def find_similar_indices(
    chunks: List[TextChunk],
    embeddings: np.ndarray,
    anchor_idx: int,
    threshold: float = 0.7,
    max_results: int = 2
) -> List[int]:
    similarities = cosine_similarity(embeddings, anchor_idx)
    candidates = [i for i in range(len(chunks)) if i != anchor_idx]
    candidates.sort(key=lambda idx: similarities[idx], reverse=True)

    selected: List[int] = []
    seen_files = {chunks[anchor_idx].filename}
    for idx in candidates:
        if similarities[idx] < threshold:
            break
        if chunks[idx].filename not in seen_files:
            selected.append(idx)
            seen_files.add(chunks[idx].filename)
        if len(selected) >= max_results:
            break

    for idx in candidates:
        if len(selected) >= max_results:
            break
        if idx in selected:
            continue
        selected.append(idx)
    return selected[:max_results]


# =============================================================================
# Category-Aware Sampling
# =============================================================================

def score_information_density(chunk: TextChunk) -> float:
    words = len(chunk.content.split())
    sentences = chunk.content.count('.') + chunk.content.count(':') + chunk.content.count(';')
    return words + sentences * 2


def sample_high_density_chunk(chunks: List[TextChunk]) -> TextChunk:
    sorted_chunks = sorted(chunks, key=score_information_density, reverse=True)
    top = sorted_chunks[: min(20, len(sorted_chunks))]
    return random.choice(top)


def sample_header_chunk(chunks: List[TextChunk]) -> TextChunk:
    candidates = [chunk for chunk in chunks if any(keyword in chunk.content for keyword in HEADER_KEYWORDS)]
    if candidates:
        return max(candidates, key=score_information_density)
    return sample_high_density_chunk(chunks)


def sample_procedural_chunks(chunks: List[TextChunk]) -> List[TextChunk]:
    chunks_by_file: Dict[str, List[TextChunk]] = {}
    for chunk in chunks:
        chunks_by_file.setdefault(chunk.filename, []).append(chunk)

    for file_chunks in chunks_by_file.values():
        file_chunks.sort(key=lambda c: c.start_line)

    viable_files = [filename for filename, file_chunks in chunks_by_file.items() if len(file_chunks) >= 3]
    if not viable_files:
        return random.sample(chunks, min(3, len(chunks)))

    selected_file = random.choice(viable_files)
    file_chunks = chunks_by_file[selected_file]
    start_idx = random.randint(0, max(0, len(file_chunks) - 3))
    count = random.choice([2, 3])
    return file_chunks[start_idx:start_idx + count]


def sample_similar_chunks(chunks: List[TextChunk], embeddings: np.ndarray) -> List[TextChunk]:
    anchor_idx = random.randrange(len(chunks))
    similar = find_similar_indices(chunks, embeddings, anchor_idx)
    selected = [chunks[anchor_idx]]
    selected.extend(chunks[idx] for idx in similar)
    if len(selected) == 1 and len(chunks) > 1:
        other_idx = random.choice([i for i in range(len(chunks)) if i != anchor_idx])
        selected.append(chunks[other_idx])
    return selected


def sample_chunks_by_category(category: str, chunks: List[TextChunk], embeddings: np.ndarray) -> List[TextChunk]:
    if category == "factual_lookup":
        return [sample_high_density_chunk(chunks)]
    if category == "conceptual_explanation":
        return [sample_header_chunk(chunks)]
    if category == "procedural":
        return sample_procedural_chunks(chunks)
    if category in {"comparative", "multi_file_synthesis"}:
        return sample_similar_chunks(chunks, embeddings)
    return [sample_high_density_chunk(chunks)]


# =============================================================================
# LLM Prompting & Quality Gate
# =============================================================================

def build_generation_prompt(category: str, snippets: List[TextChunk]) -> str:
    snippet_text = "\n\n".join(
        f"Snippet {idx + 1}:\n{chunk.content}" for idx, chunk in enumerate(snippets)
    )
    return f"""You are provided with {len(snippets)} snippet{'s' if len(snippets) > 1 else ''}.
Based on the logical relationship between them, generate a high-quality Question and Answer pair.
Use only the information visible in the snippets.
If the snippets are disjointed, lack meaningful information, or cannot form a coherent question for the assigned category, return only the string [REFUSE].
For comparative and multi_file_synthesis categories, make sure the answer references the relationship between the snippets.

Category: {category}

{snippet_text}

Respond with valid JSON only:
{{
  "question": "...",
  "answer": "..."
}}
""".strip()


def parse_qa_response(response_text: str) -> Optional[Dict[str, str]]:
    response_text = response_text.strip()
    if response_text == "[REFUSE]":
        return None
    try:
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not match:
            return None
        qa_json = json.loads(match.group(0))
        question = qa_json.get("question", "").strip()
        answer = qa_json.get("answer", "").strip()
        if not question or not answer or answer == "[REFUSE]":
            return None
        return {"question": question, "answer": answer}
    except json.JSONDecodeError:
        return None


def ask_model(category: str, snippets: List[TextChunk]) -> Optional[Dict[str, str]]:
    prompt = build_generation_prompt(category, snippets)

    api_key = open("/home/yup031/Project1/api-key.txt").read().strip()
    base_url = "https://tritonai-api.ucsd.edu"
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=CONFIG["llm_model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=CONFIG["llm_temperature"],
        max_tokens=CONFIG["llm_max_tokens"]
    )
    return parse_qa_response(response.choices[0].message.content)


# =============================================================================
# Output Generation
# =============================================================================

def create_source_evidence(snippets: List[TextChunk]) -> List[Dict[str, Any]]:
    return [chunk.to_evidence() for chunk in snippets]


def generate_question_type_distribution(qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    distribution: Dict[str, Any] = {}
    for question_type in QUESTION_TYPES:
        question_ids = [i for i, pair in enumerate(qa_pairs) if pair.get("question_type") == question_type]
        distribution[question_type] = {
            "question_ids": question_ids,
            "count": len(question_ids)
        }
    summary = {
        "total_questions": len(qa_pairs),
        "unique_type_assignments": len([t for t in QUESTION_TYPES if distribution[t]["count"] > 0]),
        "questions_with_single_type": len([pair for pair in qa_pairs if isinstance(pair.get("question_type"), str)]),
        "questions_with_multiple_types": 0
    }
    return {
        "question_type_distribution": distribution,
        "summary": summary
    }


def save_outputs(qa_pairs: List[Dict[str, Any]], output_dir: Path) -> None:
    output_file = output_dir / CONFIG["output_qa_path"]
    output_pairs = [
        {
            "question": pair["question"],
            "reference_answer": pair["reference_answer"],
            "source_evidence": pair["source_evidence"]
        }
        for pair in qa_pairs
    ]
    output_file.write_text(json.dumps(output_pairs, indent=2, ensure_ascii=False), encoding='utf-8')
    logger.info(f"Saved Q&A pairs to {output_file}")

    distribution_path = output_dir / CONFIG["output_distribution_path"]
    distribution_path.write_text(json.dumps(generate_question_type_distribution(qa_pairs), indent=2, ensure_ascii=False), encoding='utf-8')
    logger.info(f"Saved distribution to {distribution_path}")


# =============================================================================
# Main Generation Loop
# =============================================================================

def generate_valid_qa_pairs(chunks: List[TextChunk], embeddings: np.ndarray, target_count: int = 100) -> List[Dict[str, Any]]:
    qa_pairs: List[Dict[str, Any]] = []
    seen_questions = set()
    category_idx = 0
    attempts = 0

    while len(qa_pairs) < target_count and attempts < CONFIG["max_quality_attempts"]:
        category = QUESTION_TYPES[category_idx % len(QUESTION_TYPES)]
        snippets = sample_chunks_by_category(category, chunks, embeddings)
        if not snippets:
            attempts += 1
            continue

        qa_data = ask_model(category, snippets)
        attempts += 1

        if not qa_data:
            continue

        if qa_data["question"] in seen_questions:
            continue

        qa_pairs.append({
            "question": qa_data["question"],
            "reference_answer": qa_data["answer"],
            "source_evidence": create_source_evidence(snippets),
            "question_type": category
        })
        seen_questions.add(qa_data["question"])

        if len(qa_pairs) % 10 == 0:
            logger.info(f"Collected {len(qa_pairs)} valid QA pairs")

        category_idx += 1
        time.sleep(CONFIG["sleep_seconds"])

    if len(qa_pairs) < target_count:
        logger.warning(f"Only generated {len(qa_pairs)} valid pairs after {attempts} attempts")

    return qa_pairs


def run_golden_qa_pipeline() -> None:
    logger.info("Starting Golden Q&A Generation Pipeline")

    chunks = load_rst_files(SOURCE_DIR)
    if not chunks:
        logger.error("No chunks available for generation.")
        return

    embeddings = build_embeddings(chunks, CONFIG["embedding_model"])
    qa_pairs = generate_valid_qa_pairs(chunks, embeddings, target_count=CONFIG["target_pairs"])
    save_outputs(qa_pairs, BASE_DIR)
    logger.info("Generation complete.")


if __name__ == "__main__":
    run_golden_qa_pipeline()
