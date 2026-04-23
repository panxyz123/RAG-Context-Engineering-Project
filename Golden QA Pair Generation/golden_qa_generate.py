"""
Golden Q&A Dataset Generator for RAG Project
============================================
Generates high-quality Q&A pairs from RST documentation using semantic clustering
and LLM-based generation with multi-source synthesis.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from dotenv import load_dotenv

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# ML/Data imports
from sklearn.cluster import KMeans
import numpy as np

# =============================================================================
# Setup & Configuration
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Configuration
CONFIG = {
    "sourcedocs_path": "home/yup031/Project1/sourcedocs",
    "output_qa_path": "golden_qa_pairs.json",
    "output_distribution_path": "question-type-distribution.json",
    "num_clusters": 8,  # k=6-8 as per requirement
    "min_qa_pairs": 40,
    "chunk_size": 500,
    "chunk_overlap": 100,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "openai_api_key": open("/home/yup031/Project1/api-key.txt").read().strip(),
    "openai_base_url": "https://tritonai-api.ucsd.edu",
    "model_id": "mistral.mistral-large-3-675b-instruct",  # TritonAI available model
    "delay_seconds": 2,
}

# Personas for diversity
PERSONAS = ["Novice User", "System Architect", "ML Engineer"]

# Question types for coverage
QUESTION_TYPES = [
    "factual_lookup",
    "conceptual_explanation",
    "procedural",
    "comparative",
    "feature_enumeration",
    "multi_file_synthesis"
]

# =============================================================================
# Data Models
# =============================================================================
@dataclass
class TextChunk:
    """Represents a chunk of text from an RST file."""
    content: str
    filename: str
    start_line: int
    end_line: int
    source_path: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "source_path": self.source_path
        }


# =============================================================================
# RST File Loading & Chunking
# =============================================================================
def load_rst_files(sourcedocs_path: str) -> List[TextChunk]:
    """Load and chunk all RST files from sourcedocs directory."""
    chunks = []
    sourcedocs_path = Path(sourcedocs_path)
    
    # Get all RST files
    rst_files = sorted(sourcedocs_path.glob("*.rst"))
    logger.info(f"Found {len(rst_files)} RST files")
    
    # Initialize text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG["chunk_size"],
        chunk_overlap=CONFIG["chunk_overlap"],
        separators=["\n\n", "\n", " ", ""]
    )
    
    for rst_file in rst_files:
        try:
            with open(rst_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into chunks
            split_texts = splitter.split_text(content)
            
            # Calculate line numbers
            line_counter = 1
            for chunk_text in split_texts:
                start_line = line_counter
                # Count newlines in chunk to estimate end line
                end_line = line_counter + chunk_text.count('\n')
                
                chunk = TextChunk(
                    content=chunk_text,
                    filename=rst_file.name,
                    start_line=start_line,
                    end_line=end_line,
                    source_path=str(rst_file)
                )
                chunks.append(chunk)
                line_counter = end_line + 1
                
            logger.info(f"Processed {rst_file.name}: {len(split_texts)} chunks")
        except Exception as e:
            logger.error(f"Error processing {rst_file.name}: {e}")
    
    logger.info(f"Total chunks created: {len(chunks)}")
    return chunks


# =============================================================================
# Semantic Clustering
# =============================================================================
def cluster_chunks(chunks: List[TextChunk]) -> Tuple[List[TextChunk], np.ndarray, KMeans]:
    """
    Cluster chunks using HuggingFace embeddings and K-Means.
    Returns chunks, cluster labels, and fitted KMeans model.
    """
    logger.info("Initializing embeddings model...")
    embeddings_model = HuggingFaceEmbeddings(
        model_name=CONFIG["embedding_model"],
        model_kwargs={"device": "cpu"}  # Use "cuda" if GPU available
    )
    
    # Get embeddings for all chunks
    logger.info(f"Embedding {len(chunks)} chunks...")
    chunk_contents = [chunk.content for chunk in chunks]
    embeddings = embeddings_model.embed_documents(chunk_contents)
    embeddings_array = np.array(embeddings)
    
    # Apply K-Means clustering
    logger.info(f"Clustering into {CONFIG['num_clusters']} clusters...")
    kmeans = KMeans(
        n_clusters=CONFIG["num_clusters"],
        random_state=42,
        n_init=10
    )
    cluster_labels = kmeans.fit_predict(embeddings_array)
    
    # Log cluster distribution
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        logger.info(f"Cluster {cluster_id}: {count} chunks")
    
    return chunks, cluster_labels, kmeans


# =============================================================================
# Multi-Source Sampling
# =============================================================================
def sample_diverse_chunks(
    chunks: List[TextChunk],
    cluster_labels: np.ndarray,
    num_clusters: int,
    num_chunks: int = 2
) -> List[TextChunk]:
    """
    Sample 1-3 chunks from different clusters to enable cross-file/topic synthesis.
    """
    # Create cluster-to-chunks mapping
    cluster_to_chunks = defaultdict(list)
    for idx, (chunk, label) in enumerate(zip(chunks, cluster_labels)):
        cluster_to_chunks[label].append((idx, chunk))
    
    # Randomly select clusters and sample chunks from each
    num_sampled_clusters = min(np.random.randint(1, 4), num_clusters)
    selected_clusters = np.random.choice(
        num_clusters,
        size=num_sampled_clusters,
        replace=False
    )
    
    sampled_chunks = []
    for cluster_id in selected_clusters:
        if cluster_to_chunks[cluster_id]:
            # Sample one chunk from this cluster
            cluster_items = cluster_to_chunks[cluster_id]
            selected_item = np.random.choice(len(cluster_items))
            _, chunk = cluster_items[selected_item]
            sampled_chunks.append(chunk)
    
    return sampled_chunks


# =============================================================================
# LLM-Based Q&A Generation
# =============================================================================
def generate_qa_pairs(
    chunks: List[TextChunk],
    cluster_labels: np.ndarray,
    num_pairs: int = 40
) -> List[Dict[str, Any]]:
    """
    Generate Q&A pairs using OpenAI-compatible API with persona and type rotation.
    """
    logger.info(f"Initializing LLM client...")
    
    # Initialize OpenAI client - works with OpenAI API, TritonAI, and compatible services
    from openai import OpenAI
    
    client = OpenAI(
        api_key=CONFIG["openai_api_key"],
        base_url=CONFIG["openai_base_url"]
    )
    
    qa_pairs = []
    persona_idx = 0
    question_type_idx = 0
    
    num_clusters = len(np.unique(cluster_labels))
    
    for pair_idx in range(num_pairs):
        try:
            # Rotate persona and question type
            persona = PERSONAS[persona_idx % len(PERSONAS)]
            question_type = QUESTION_TYPES[question_type_idx % len(QUESTION_TYPES)]
            
            # Sample diverse chunks
            sampled_chunks = sample_diverse_chunks(
                chunks,
                cluster_labels,
                num_clusters,
                num_chunks=np.random.randint(1, 4)
            )
            
            if not sampled_chunks:
                logger.warning(f"Could not sample chunks for pair {pair_idx}")
                continue
            
            # Construct context from sampled chunks
            context = "\n\n---\n\n".join([
                f"[{chunk.filename}]\n{chunk.content}"
                for chunk in sampled_chunks
            ])
            
            # Construct prompt
            prompt = _build_generation_prompt(
                context=context,
                persona=persona,
                question_type=question_type,
                pair_idx=pair_idx
            )
            
            # Call LLM
            logger.info(f"Generating pair {pair_idx + 1}/{num_pairs} - "
                       f"Type: {question_type}, Persona: {persona}")
            
            response = client.chat.completions.create(
                model=CONFIG["model_id"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            qa_data = _parse_qa_response(response_text)
            
            # Add sources
            qa_data["sources"] = [
                {"file": chunk.filename, "lines": [chunk.start_line, chunk.end_line]}
                for chunk in sampled_chunks
            ]
            qa_data["question_type"] = question_type
            qa_data["persona"] = persona
            
            qa_pairs.append(qa_data)
            
            # Rate limiting
            time.sleep(CONFIG["delay_seconds"])
            
        except Exception as e:
            logger.error(f"Error generating pair {pair_idx}: {e}")
            continue
        
        # Rotate indices
        persona_idx += 1
        question_type_idx += 1
    
    logger.info(f"Generated {len(qa_pairs)} Q&A pairs")
    return qa_pairs


def _build_generation_prompt(
    context: str,
    persona: str,
    question_type: str,
    pair_idx: int
) -> str:
    """Build the prompt for Q&A generation."""
    
    type_descriptions = {
        "factual_lookup": "Ask a specific factual question about information in the documents.",
        "conceptual_explanation": "Ask for an explanation of a concept or idea discussed in the documents.",
        "procedural": "Ask how to perform a specific procedure or task described in the documents.",
        "comparative": "Ask a question comparing two or more approaches, features, or concepts.",
        "feature_enumeration": "Ask the user to list or enumerate key features, components, or elements.",
        "multi_file_synthesis": "Ask a question that requires synthesizing information from multiple source documents."
    }
    
    prompt = f"""You are a {persona} with expertise in machine learning and AI systems.
Your task is to generate a high-quality Q&A pair based on the provided documentation.

Question Type: {question_type}
Description: {type_descriptions[question_type]}

Documentation Context:
{context}

Requirements:
1. Generate a thoughtful question that a {persona} would ask
2. The question should be specific to the provided context
3. Provide a comprehensive, accurate answer based solely on the documentation
4. For "{question_type}" questions, {type_descriptions[question_type].lower()}
5. Ensure the answer is well-structured and clear

Format your response as JSON with exactly this structure:
{{
  "question": "your question here",
  "answer": "your comprehensive answer here"
}}

Important: Return ONLY the JSON object, no other text."""
    
    return prompt


def _parse_qa_response(response_text: str) -> Dict[str, str]:
    """Parse LLM response to extract Q&A."""
    try:
        # Try to extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            qa_data = json.loads(json_str)
            return {
                "question": qa_data.get("question", ""),
                "answer": qa_data.get("answer", "")
            }
    except Exception as e:
        logger.error(f"Error parsing response: {e}")
    
    # Fallback: split by question/answer markers
    return {
        "question": "Unable to parse question",
        "answer": "Unable to parse answer"
    }


# =============================================================================
# Output Generation
# =============================================================================
def generate_question_type_distribution(
    qa_pairs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Generate question type distribution statistics."""
    
    distribution = {}
    for question_type in QUESTION_TYPES:
        matching_pairs = [
            pair for pair in qa_pairs
            if pair.get("question_type") == question_type
        ]
        
        distribution[question_type] = {
            "question_ids": [i for i, pair in enumerate(qa_pairs) 
                           if pair.get("question_type") == question_type],
            "count": len(matching_pairs)
        }
    
    # Summary statistics
    summary = {
        "total_questions": len(qa_pairs),
        "unique_type_assignments": len([t for t in QUESTION_TYPES if distribution[t]["count"] > 0]),
        "questions_with_single_type": len([
            pair for pair in qa_pairs
            if isinstance(pair.get("question_type"), str)
        ]),
        "questions_with_multiple_types": 0  # Can be extended for multi-type support
    }
    
    return {
        "question_type_distribution": distribution,
        "summary": summary
    }


def save_outputs(qa_pairs: List[Dict[str, Any]]) -> None:
    """Save Q&A pairs and distribution statistics to JSON files."""
    
    # Prepare Q&A pairs output (remove internal fields)
    output_qa_pairs = []
    for pair in qa_pairs:
        output_qa_pairs.append({
            "question": pair["question"],
            "reference_answer": pair["answer"],
            "source_evidence": pair["sources"]
        })
    
    # Save Q&A pairs
    with open(CONFIG["output_qa_path"], 'w') as f:
        json.dump(output_qa_pairs, f, indent=2)
    logger.info(f"Saved Q&A pairs to {CONFIG['output_qa_path']}")
    
    # Generate and save distribution
    distribution = generate_question_type_distribution(qa_pairs)
    with open(CONFIG["output_distribution_path"], 'w') as f:
        json.dump(distribution, f, indent=2)
    logger.info(f"Saved distribution to {CONFIG['output_distribution_path']}")


# =============================================================================
# Main Pipeline
# =============================================================================
def run_golden_qa_pipeline():
    """Execute the complete Golden Q&A generation pipeline."""
    
    logger.info("="*70)
    logger.info("Starting Golden Q&A Generation Pipeline")
    logger.info("="*70)
    
    try:
        # Step 1: Load and chunk RST files
        logger.info("\n[Step 1] Loading and chunking RST files...")
        chunks = load_rst_files(CONFIG["sourcedocs_path"])
        
        if not chunks:
            logger.error("No chunks loaded. Exiting.")
            return
        
        # Step 2: Semantic clustering
        logger.info("\n[Step 2] Performing semantic clustering...")
        chunks, cluster_labels, kmeans = cluster_chunks(chunks)
        
        # Step 3: Generate Q&A pairs
        logger.info("\n[Step 3] Generating Q&A pairs...")
        qa_pairs = generate_qa_pairs(
            chunks,
            cluster_labels,
            num_pairs=CONFIG["min_qa_pairs"]
        )
        
        if len(qa_pairs) < CONFIG["min_qa_pairs"]:
            logger.warning(f"Generated {len(qa_pairs)} pairs, but {CONFIG['min_qa_pairs']} were requested")
        
        # Step 4: Save outputs
        logger.info("\n[Step 4] Saving outputs...")
        save_outputs(qa_pairs)
        
        logger.info("\n" + "="*70)
        logger.info("Golden Q&A Generation Pipeline Completed Successfully!")
        logger.info(f"Generated {len(qa_pairs)} Q&A pairs")
        logger.info(f"Outputs saved to:")
        logger.info(f"  - {CONFIG['output_qa_path']}")
        logger.info(f"  - {CONFIG['output_distribution_path']}")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    run_golden_qa_pipeline()
