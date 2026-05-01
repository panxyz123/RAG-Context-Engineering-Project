"""
Human Evaluation Script for Golden QA Pairs
Displays all QA pairs with their source evidence and original text excerpts
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class HumanEvaluation:
    def __init__(self, qa_file: str, source_dir: str):
        """
        Initialize the evaluation class.
        
        Args:
            qa_file: Path to golden_qa_pairs.json
            source_dir: Path to sourcedocs directory
        """
        self.qa_file = qa_file
        self.source_dir = source_dir
        self.qa_pairs = []
        self.load_qa_pairs()
    
    def load_qa_pairs(self) -> None:
        """Load QA pairs from JSON file."""
        try:
            with open(self.qa_file, 'r', encoding='utf-8') as f:
                self.qa_pairs = json.load(f)
            print(f"✓ Loaded {len(self.qa_pairs)} QA pairs from {self.qa_file}\n")
        except FileNotFoundError:
            print(f"✗ Error: File {self.qa_file} not found")
            exit(1)
        except json.JSONDecodeError:
            print(f"✗ Error: Invalid JSON format in {self.qa_file}")
            exit(1)
    
    def get_source_text(self, filename: str, start_line: int, end_line: int) -> Tuple[str, bool]:
        """
        Extract text from source file at specified line range.
        
        Args:
            filename: Name of the source file (e.g., 'ragspecs.rst')
            start_line: Starting line number (1-indexed)
            end_line: Ending line number (1-indexed)
            
        Returns:
            Tuple of (extracted_text, success_bool)
        """
        filepath = os.path.join(self.source_dir, filename)
        
        if not os.path.exists(filepath):
            return f"[FILE NOT FOUND: {filepath}]", False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Convert to 0-indexed
            start_idx = max(0, start_line - 1)
            end_idx = min(len(lines), end_line)
            
            if start_idx >= len(lines):
                return f"[START LINE {start_line} OUT OF RANGE - File has {len(lines)} lines]", False
            
            extracted = ''.join(lines[start_idx:end_idx])
            return extracted, True
        
        except Exception as e:
            return f"[ERROR READING FILE: {str(e)}]", False
    
    def format_source_evidence(self, evidence: Dict) -> str:
        """Format and retrieve source evidence text."""
        filename = evidence.get('file', 'unknown')
        lines = evidence.get('lines', [0, 0])
        start_line, end_line = lines[0], lines[1] if len(lines) > 1 else lines[0]
        
        text, success = self.get_source_text(filename, start_line, end_line)
        
        output = f"\n  📄 File: {filename}\n"
        output += f"     Lines: {start_line}-{end_line}\n"
        output += f"     {'─' * 70}\n"
        
        # Format the extracted text with line numbers
        text_lines = text.split('\n')
        for i, line in enumerate(text_lines, start=start_line):
            if line.strip():  # Only show non-empty lines
                output += f"     {i:4d} | {line}\n"
        
        output += f"     {'─' * 70}"
        
        return output
    
    def display_qa_pair(self, index: int, qa_pair: Dict, show_source: bool = True) -> str:
        """
        Format a single QA pair with all details.
        
        Args:
            index: Index of the QA pair (0-based)
            qa_pair: The QA pair dictionary
            show_source: Whether to include source evidence
            
        Returns:
            Formatted string representation
        """
        output = ""
        output += f"\n{'=' * 80}\n"
        output += f"📌 QA PAIR #{index + 1}\n"
        output += f"{'=' * 80}\n"
        
        # Question
        question = qa_pair.get('question', 'N/A')
        output += f"\n❓ QUESTION:\n"
        output += f"  {question}\n"
        
        # Reference Answer
        answer = qa_pair.get('reference_answer', 'N/A')
        output += f"\n✓ REFERENCE ANSWER:\n"
        
        # Handle both string and dict answers
        if isinstance(answer, dict):
            import json as json_module
            answer_str = json_module.dumps(answer, indent=2)
        else:
            answer_str = str(answer)
        
        answer_lines = answer_str.split('\n')
        for line in answer_lines:
            output += f"  {line}\n"
        
        # Source Evidence
        if show_source:
            evidence_list = qa_pair.get('source_evidence', [])
            output += f"\n📚 SOURCE EVIDENCE ({len(evidence_list)} source(s)):\n"
            
            for i, evidence in enumerate(evidence_list, 1):
                output += f"\n  Source #{i}:"
                output += self.format_source_evidence(evidence)
                output += "\n"
        
        return output
    
    def display_summary_table(self) -> None:
        """Display a summary table of all QA pairs."""
        print("\n" + "=" * 80)
        print("SUMMARY TABLE: GOLDEN QA PAIRS OVERVIEW")
        print("=" * 80)
        
        print(f"\n{'Index':<6} {'Question':<50} {'Source Files':<30}")
        print("-" * 86)
        
        for idx, qa_pair in enumerate(self.qa_pairs, 1):
            question = qa_pair.get('question', 'N/A')
            if len(question) > 47:
                question = question[:44] + "..."
            
            source_files = ", ".join([
                evidence.get('file', 'unknown')
                for evidence in qa_pair.get('source_evidence', [])
            ])
            if len(source_files) > 27:
                source_files = source_files[:24] + "..."
            
            print(f"{idx:<6} {question:<50} {source_files:<30}")
        
        print("-" * 86)
        print(f"Total: {len(self.qa_pairs)} QA pairs\n")
    
    def export_to_file(self, output_file: str, indices: Optional[List[int]] = None) -> None:
        """
        Export QA pairs to a file.
        
        Args:
            output_file: Path to output file
            indices: List of QA pair indices to export (1-based). If None, export all.
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("HUMAN EVALUATION: GOLDEN QA PAIRS WITH SOURCE TEXT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total QA Pairs: {len(self.qa_pairs)}\n\n")
            
            if indices is None:
                indices = list(range(1, len(self.qa_pairs) + 1))
            
            for idx in indices:
                if 1 <= idx <= len(self.qa_pairs):
                    qa_pair = self.qa_pairs[idx - 1]
                    f.write(self.display_qa_pair(idx - 1, qa_pair))
            
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"✓ EVALUATION COMPLETE: Exported {len(indices)} QA pairs\n")
            f.write("=" * 80 + "\n")
        
        print(f"✓ Exported {len(indices)} QA pairs to {output_file}")
    
    def display_pair_by_index(self, index: int, show_source: bool = True) -> None:
        """Display a specific QA pair by index."""
        if 1 <= index <= len(self.qa_pairs):
            qa_pair = self.qa_pairs[index - 1]
            print(self.display_qa_pair(index - 1, qa_pair, show_source=show_source))
        else:
            print(f"✗ Index {index} out of range. Valid range: 1-{len(self.qa_pairs)}")


def main():
    """Main execution function."""
    import sys
    
    # Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    qa_file = os.path.join(current_dir, 'validation-set-golden-qa-pairs.json')
    source_dir = os.path.join(os.path.dirname(current_dir), 'sourcedocs')
    
    print("Initializing Human Evaluation...")
    print(f"QA File: {qa_file}")
    print(f"Source Directory: {source_dir}\n")
    
    # Create evaluator
    evaluator = HumanEvaluation(qa_file, source_dir)
    
    # Display summary table
    evaluator.display_summary_table()
    
    # Check for command-line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'all':
            # Export all QA pairs to file
            output_file = sys.argv[2] if len(sys.argv) > 2 else 'human_eval_output.txt'
            evaluator.export_to_file(output_file)
            print(f"\nOpen {output_file} to view all QA pairs with source text.")
        
        elif command == 'export':
            # Export specific indices
            output_file = sys.argv[3] if len(sys.argv) > 3 else 'human_eval_output.txt'
            try:
                indices = [int(x) for x in sys.argv[2].split(',')]
                evaluator.export_to_file(output_file, indices)
            except (ValueError, IndexError):
                print("Usage: python human_eval.py export <indices> [output_file]")
                print("Example: python human_eval.py export 1,2,3,4,5")
        
        elif command.isdigit():
            # View specific pair
            index = int(command)
            evaluator.display_pair_by_index(index)
        
        else:
            print_help()
    else:
        print_help()


def print_help():
    """Print usage help."""
    print("\n" + "=" * 80)
    print("USAGE EXAMPLES:")
    print("=" * 80)
    print("\n1. View this summary:")
    print("   python human_eval.py")
    print("\n2. View specific QA pair (e.g., pair #5):")
    print("   python human_eval.py 5")
    print("\n3. Export all QA pairs to file:")
    print("   python human_eval.py all [output_file.txt]")
    print("   (default: human_eval_output.txt)")
    print("\n4. Export specific QA pairs (e.g., pairs 1,3,5):")
    print("   python human_eval.py export 1,3,5 [output_file.txt]")
    print("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    main()

