import json
import re
import openai
from pathlib import Path

# Load API key
with open('/home/yup031/Project1/api-key.txt', 'r') as f:
    api_key = f.read().strip()
openai.api_key = api_key

# Directory for source docs
SOURCE_DIR = Path('/home/yup031/Project1/sourcedocs')

def read_source_content(file, lines):
    """Read content from source file for given lines."""
    file_path = SOURCE_DIR / file
    if not file_path.exists():
        return f"File {file} not found."
    with open(file_path, 'r') as f:
        content_lines = f.readlines()
    start, end = lines[0] - 1, lines[1]  # 0-based
    return ''.join(content_lines[start:end])

def parse_qa_pairs(file_path):
    """Parse QA pairs from human_eval_complete.txt"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split by QA pair sections
    pairs = re.split(r'================================================================================\n📌 QA PAIR #\d+\n================================================================================', content)[1:]  # Skip header
    
    qa_pairs = []
    for pair in pairs:
        # Extract question
        q_match = re.search(r'❓ QUESTION:\s*(.*?)\n\n✓ REFERENCE ANSWER:', pair, re.DOTALL)
        question = q_match.group(1).strip() if q_match else ""
        
        # Extract answer
        a_match = re.search(r'✓ REFERENCE ANSWER:\s*(.*?)\n\n📚 SOURCE EVIDENCE', pair, re.DOTALL)
        answer = a_match.group(1).strip() if a_match else ""
        
        # Extract sources
        sources = []
        source_matches = re.findall(r'Source #\d+:\s*📄 File: (\S+)\s*Lines: (\d+)-(\d+)', pair)
        for match in source_matches:
            file, start, end = match
            sources.append({
                'file': file,
                'lines': [int(start), int(end)]
            })
        
        qa_pairs.append({
            'question': question,
            'answer': answer,
            'sources': sources
        })
    
    return qa_pairs

def get_context(sources):
    """Get concatenated context from sources."""
    context_parts = []
    for source in sources:
        content = read_source_content(source['file'], source['lines'])
        context_parts.append(f"From {source['file']} lines {source['lines'][0]}-{source['lines'][1]}:\n{content}")
    return '\n\n'.join(context_parts)

def judge_answer(question, answer, context):
    """Use LLM to judge the answer on three metrics."""
    prompt = f"""
You are an evaluator. Judge the following answer based on the question and provided context.

Question: {question}

Answer: {answer}

Context: {context}

For each metric, respond with PASS or FAIL, and a brief reason.

Metrics:
- Correctness: Does the answer accurately address the question based on the retrieved context?
- Completeness: Does the answer cover all relevant aspects of the question, or does it miss important information?
- Faithfulness: Is the answer grounded in the retrieved context, or does it introduce information not supported by the chunks?

Format your response as:
Correctness: PASS/FAIL - reason
Completeness: PASS/FAIL - reason
Faithfulness: PASS/FAIL - reason
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    judgment = response.choices[0].message.content.strip()
    
    # Parse the judgment
    results = {}
    for line in judgment.split('\n'):
        if ': ' in line:
            metric, rest = line.split(': ', 1)
            status, reason = rest.split(' - ', 1)
            results[metric.lower()] = {'status': status.upper(), 'reason': reason}
    
    return results

def regenerate_answer(question, context, failed_metrics):
    """Regenerate answer using LLM."""
    failed_list = ', '.join(failed_metrics)
    prompt = f"""
You failed to meet the following metrics: {failed_list}.

Regenerate the answer based on the following content and question.

Question: {question}

Content: {context}

Provide a new answer that addresses the failures.
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )
    new_answer = response.choices[0].message.content.strip()
    return new_answer

def main():
    input_file = '/home/yup031/Project1/Golden QA Pair Generation/human_eval_complete.txt'
    qa_pairs = parse_qa_pairs(input_file)
    
    final_pairs = []
    for i, pair in enumerate(qa_pairs):
        print(f"Evaluating QA Pair #{i+1}")
        question = pair['question']
        answer = pair['answer']
        sources = pair['sources']
        context = get_context(sources)
        
        max_iterations = 5  # Prevent infinite loop
        iteration = 0
        while iteration < max_iterations:
            judgment = judge_answer(question, answer, context)
            failed = [m for m, r in judgment.items() if r['status'] == 'FAIL']
            if not failed:
                print(f"  Passed all metrics after {iteration+1} iterations")
                break
            else:
                print(f"  Failed: {failed}. Regenerating...")
                answer = regenerate_answer(question, context, failed)
                iteration += 1
        else:
            print(f"  Max iterations reached, using last answer")
        
        final_pairs.append({
            'question': question,
            'reference_answer': answer,
            'source_evidence': sources
        })
    
    # Save to JSON
    with open('/home/yup031/Project1/Golden QA Pair Generation/llm_judged_qa_pairs.json', 'w') as f:
        json.dump(final_pairs, f, indent=2)
    
    # Save to TXT
    with open('/home/yup031/Project1/Golden QA Pair Generation/llm_judged_qa_pairs.txt', 'w') as f:
        f.write("================================================================================\n")
        f.write("LLM JUDGED QA PAIRS\n")
        f.write("================================================================================\n")
        f.write(f"Total QA Pairs: {len(final_pairs)}\n\n")
        for i, pair in enumerate(final_pairs):
            f.write(f"================================================================================\n")
            f.write(f"📌 QA PAIR #{i+1}\n")
            f.write("================================================================================\n\n")
            f.write("❓ QUESTION:\n")
            f.write(f"  {pair['question']}\n\n")
            f.write("✓ JUDGED ANSWER:\n")
            f.write(f"  {pair['reference_answer']}\n\n")
            f.write("📚 SOURCE EVIDENCE:\n")
            for j, source in enumerate(pair['source_evidence']):
                f.write(f"  Source #{j+1}:\n")
                f.write(f"  📄 File: {source['file']}\n")
                f.write(f"     Lines: {source['lines'][0]}-{source['lines'][1]}\n")
                content = read_source_content(source['file'], source['lines'])
                f.write("     ──────────────────────────────────────────────────────────────────────\n")
                f.write(f"     {content.replace(chr(10), chr(10) + '     ')}\n")
                f.write("     ──────────────────────────────────────────────────────────────────────\n\n")

if __name__ == "__main__":
    main()</content>
<parameter name="filePath">/home/yup031/Project1/Golden QA Pair Generation/llm-as-a-judge-eval.py