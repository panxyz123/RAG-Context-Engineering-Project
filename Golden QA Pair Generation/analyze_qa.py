"""
Analysis and Reporting Tool for Golden Q&A Dataset
==================================================
Generate comprehensive reports and statistics from generated Q&A pairs.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter
import statistics


class GoldenQAAnalyzer:
    """Analyze generated Q&A pairs and generate reports."""
    
    def __init__(self, qa_file: str = "golden_qa_pairs.json", 
                 dist_file: str = "question-type-distribution.json"):
        self.qa_file = qa_file
        self.dist_file = dist_file
        self.qa_pairs = []
        self.distribution = {}
        self.load_data()
    
    def load_data(self):
        """Load Q&A pairs and distribution from JSON files."""
        try:
            with open(self.qa_file, 'r') as f:
                self.qa_pairs = json.load(f)
            print(f"✓ Loaded {len(self.qa_pairs)} Q&A pairs from {self.qa_file}")
        except FileNotFoundError:
            print(f"✗ {self.qa_file} not found")
            sys.exit(1)
        
        try:
            with open(self.dist_file, 'r') as f:
                self.distribution = json.load(f)
            print(f"✓ Loaded distribution from {self.dist_file}")
        except FileNotFoundError:
            print(f"⚠ {self.dist_file} not found (will generate)")
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate quality metrics for the dataset."""
        print("\n" + "="*70)
        print("QUALITY METRICS")
        print("="*70)
        
        report = {
            "total_pairs": len(self.qa_pairs),
            "pairs_with_sources": sum(1 for p in self.qa_pairs if p.get("source_evidence")),
            "pairs_with_questions": sum(1 for p in self.qa_pairs if p.get("question")),
            "pairs_with_answers": sum(1 for p in self.qa_pairs if p.get("reference_answer")),
        }
        
        # Question statistics
        questions = [p.get("question", "") for p in self.qa_pairs]
        answers = [p.get("reference_answer", "") for p in self.qa_pairs]

        # Ensure answers are strings (handle cases where they might be dicts)
        answers = [str(a) if not isinstance(a, str) else a for a in answers]
        
        question_lengths = [len(q.split()) for q in questions]
        answer_lengths = [len(a.split()) for a in answers]
        
        report["question_length_stats"] = {
            "avg_words": round(statistics.mean(question_lengths), 2),
            "min_words": min(question_lengths),
            "max_words": max(question_lengths),
        }
        
        report["answer_length_stats"] = {
            "avg_words": round(statistics.mean(answer_lengths), 2),
            "min_words": min(answer_lengths),
            "max_words": max(answer_lengths),
        }
        
        # Source statistics
        all_sources = []
        sources_per_pair = []
        for p in self.qa_pairs:
            sources = p.get("source_evidence", [])
            sources_per_pair.append(len(sources))
            all_sources.extend([s["file"] for s in sources])
        
        report["source_stats"] = {
            "avg_sources_per_pair": round(statistics.mean(sources_per_pair), 2),
            "max_sources_per_pair": max(sources_per_pair),
            "unique_source_files": len(set(all_sources)),
            "most_used_files": Counter(all_sources).most_common(3),
        }
        
        # Print report
        print(f"\nDataset Size:")
        print(f"  Total Q&A pairs: {report['total_pairs']}")
        print(f"  Pairs with sources: {report['pairs_with_sources']}")
        print(f"  Pairs with questions: {report['pairs_with_questions']}")
        print(f"  Pairs with answers: {report['pairs_with_answers']}")
        
        print(f"\nQuestion Length (words):")
        print(f"  Average: {report['question_length_stats']['avg_words']}")
        print(f"  Min: {report['question_length_stats']['min_words']}")
        print(f"  Max: {report['question_length_stats']['max_words']}")
        
        print(f"\nAnswer Length (words):")
        print(f"  Average: {report['answer_length_stats']['avg_words']}")
        print(f"  Min: {report['answer_length_stats']['min_words']}")
        print(f"  Max: {report['answer_length_stats']['max_words']}")
        
        print(f"\nSource Attribution:")
        print(f"  Average sources per pair: {report['source_stats']['avg_sources_per_pair']}")
        print(f"  Unique source files: {report['source_stats']['unique_source_files']}")
        print(f"  Most used files:")
        for filename, count in report['source_stats']['most_used_files']:
            print(f"    - {filename}: {count} times")
        
        return report
    
    def generate_type_distribution_report(self) -> Dict[str, Any]:
        """Generate question type distribution report."""
        print("\n" + "="*70)
        print("QUESTION TYPE DISTRIBUTION")
        print("="*70)
        
        if self.distribution:
            dist_data = self.distribution.get("question_type_distribution", {})
            summary = self.distribution.get("summary", {})
        else:
            # Generate from scratch if not available
            dist_data = {}
            for q_type in ["factual_lookup", "conceptual_explanation", "procedural", 
                          "comparative", "feature_enumeration", "multi_file_synthesis"]:
                dist_data[q_type] = {"question_ids": [], "count": 0}
            summary = {
                "total_questions": len(self.qa_pairs),
                "unique_type_assignments": 0,
                "questions_with_single_type": 0,
                "questions_with_multiple_types": 0
            }
        
        report = {
            "distribution": dist_data,
            "summary": summary
        }
        
        print(f"\nTotal Questions: {summary.get('total_questions', len(self.qa_pairs))}")
        print(f"Unique Type Assignments: {summary.get('unique_type_assignments', 0)}")
        print(f"\nType Breakdown:")
        
        total = sum(v.get("count", 0) for v in dist_data.values())
        for q_type, data in sorted(dist_data.items()):
            count = data.get("count", 0)
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {q_type:25} {count:3} pairs ({percentage:5.1f}%)")
        
        return report
    
    def generate_coverage_report(self) -> Dict[str, Any]:
        """Generate documentation coverage report."""
        print("\n" + "="*70)
        print("DOCUMENTATION COVERAGE")
        print("="*70)
        
        # Analyze source coverage
        all_sources = []
        file_usage_count = Counter()
        
        for p in self.qa_pairs:
            sources = p.get("source_evidence", [])
            for source in sources:
                filename = source["file"]
                all_sources.append(filename)
                file_usage_count[filename] += 1
        
        total_unique_files = len(file_usage_count)
        
        # Analyze source diversity
        pairs_with_single_file = sum(1 for p in self.qa_pairs 
                                     if len(p.get("source_evidence", [])) == 1)
        pairs_with_multiple_files = sum(1 for p in self.qa_pairs 
                                       if len(p.get("source_evidence", [])) > 1)
        
        report = {
            "total_unique_files_referenced": total_unique_files,
            "pairs_with_single_file_source": pairs_with_single_file,
            "pairs_with_multiple_file_sources": pairs_with_multiple_files,
            "file_coverage": dict(file_usage_count.most_common(20)),
        }
        
        print(f"\nUnique Source Files Referenced: {total_unique_files}")
        print(f"Pairs with single-file source: {pairs_with_single_file} ({pairs_with_single_file/len(self.qa_pairs)*100:.1f}%)")
        print(f"Pairs with multi-file synthesis: {pairs_with_multiple_files} ({pairs_with_multiple_files/len(self.qa_pairs)*100:.1f}%)")
        
        print(f"\nTop Referenced Files:")
        for filename, count in list(file_usage_count.most_common(10)):
            print(f"  - {filename}: {count} references")
        
        return report
    
    def print_sample_pairs(self, num_samples: int = 3):
        """Print sample Q&A pairs."""
        print("\n" + "="*70)
        print(f"SAMPLE Q&A PAIRS (first {num_samples})")
        print("="*70)
        
        for i, pair in enumerate(self.qa_pairs[:num_samples], 1):
            print(f"\n[Pair {i}]")
            print(f"Question: {pair.get('question', 'N/A')[:100]}...")
            print(f"Answer: {pair.get('reference_answer', 'N/A')[:100]}...")
            sources = pair.get("source_evidence", [])
            if sources:
                print(f"Sources: {', '.join([s['file'] for s in sources])}")
    
    def generate_comprehensive_report(self, output_file: str = "qa_analysis_report.json"):
        """Generate comprehensive report and save to file."""
        print("\n" + "="*70)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*70)
        
        report = {
            "metadata": {
                "qa_file": self.qa_file,
                "dist_file": self.dist_file,
            },
            "quality_metrics": self.generate_quality_report(),
            "type_distribution": self.generate_type_distribution_report(),
            "coverage": self.generate_coverage_report(),
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n✓ Comprehensive report saved to {output_file}")
        return report
    
    def validate_dataset(self) -> bool:
        """Validate dataset integrity and quality."""
        print("\n" + "="*70)
        print("DATASET VALIDATION")
        print("="*70)
        
        issues = []
        
        # Check minimum pairs
        if len(self.qa_pairs) < 40:
            issues.append(f"⚠ Only {len(self.qa_pairs)} pairs (minimum 40 recommended)")
        else:
            print(f"✓ Sufficient pairs: {len(self.qa_pairs)} >= 40")
        
        # Check completeness
        incomplete_pairs = [i for i, p in enumerate(self.qa_pairs) 
                           if not p.get("question") or not p.get("reference_answer")]
        if incomplete_pairs:
            issues.append(f"✗ {len(incomplete_pairs)} incomplete pairs (missing Q or A)")
        else:
            print(f"✓ All pairs have questions and answers")
        
        # Check source attribution
        pairs_without_sources = sum(1 for p in self.qa_pairs if not p.get("source_evidence"))
        if pairs_without_sources > 0:
            issues.append(f"⚠ {pairs_without_sources} pairs without source attribution")
        else:
            print(f"✓ All pairs have source attribution")
        
        # Check diversity
        unique_files = len(set(s["file"] for p in self.qa_pairs for s in p.get("source_evidence", [])))
        if unique_files < 3:
            issues.append(f"⚠ Low file diversity: only {unique_files} unique source files")
        else:
            print(f"✓ Good file diversity: {unique_files} unique source files")
        
        # Check question/answer quality (length)
        short_questions = sum(1 for p in self.qa_pairs if len(p.get("question", "").split()) < 5)
        short_answers = sum(1 for p in self.qa_pairs if len(str(p.get("reference_answer", "")).split()) < 10)
        if short_questions > len(self.qa_pairs) * 0.2:
            issues.append(f"⚠ {short_questions} questions are too short")
        if short_answers > len(self.qa_pairs) * 0.2:
            issues.append(f"⚠ {short_answers} answers are too short")
        
        if not short_questions and not short_answers:
            print(f"✓ Question and answer lengths are appropriate")
        
        # Summary
        print("\n" + "-"*70)
        if issues:
            print(f"Validation: ⚠ {len(issues)} issue(s) found:")
            for issue in issues:
                print(f"  {issue}")
        else:
            print("Validation: ✓ Dataset passed all checks!")
        
        return len(issues) == 0


def main():
    """Run analysis tool."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Golden Q&A Dataset")
    parser.add_argument("--qa-file", default="golden_qa_pairs.json", 
                       help="Path to Q&A pairs JSON file")
    parser.add_argument("--dist-file", default="question-type-distribution.json",
                       help="Path to distribution JSON file")
    parser.add_argument("--output", default="qa_analysis_report.json",
                       help="Output report file")
    parser.add_argument("--samples", type=int, default=3,
                       help="Number of sample pairs to display")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("GOLDEN Q&A DATASET ANALYZER")
    print("="*70)
    
    analyzer = GoldenQAAnalyzer(args.qa_file, args.dist_file)
    
    # Run all analyses
    analyzer.generate_quality_report()
    analyzer.generate_type_distribution_report()
    analyzer.generate_coverage_report()
    analyzer.print_sample_pairs(args.samples)
    analyzer.validate_dataset()
    analyzer.generate_comprehensive_report(args.output)


if __name__ == "__main__":
    main()
