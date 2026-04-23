"""
Validation and Testing Script for Golden Q&A Generator
======================================================
Run this script to verify all components are working correctly before running the full pipeline.
"""

import os
import sys
import json
from pathlib import Path

def check_dependencies():
    """Verify all required packages are installed."""
    print("\n" + "="*70)
    print("1. Checking Dependencies")
    print("="*70)
    
    required_packages = [
        ("langchain", "LangChain"),
        ("langchain_text_splitters", "LangChain Text Splitters"),
        ("langchain_huggingface", "LangChain HuggingFace"),
        ("langchain_openai", "LangChain OpenAI"),
        ("sklearn", "scikit-learn"),
        ("numpy", "NumPy"),
        ("dotenv", "python-dotenv"),
        ("openai", "OpenAI"),
        ("sentence_transformers", "Sentence Transformers"),
        ("torch", "PyTorch"),
    ]
    
    all_ok = True
    for module, name in required_packages:
        try:
            __import__(module)
            print(f"✓ {name:30} installed")
        except ImportError:
            print(f"✗ {name:30} NOT installed")
            all_ok = False
    
    return all_ok


def check_sourcedocs():
    """Verify sourcedocs directory and RST files."""
    print("\n" + "="*70)
    print("3. Checking Sourcedocs Directory")
    print("="*70)
    
    sourcedocs_path = Path("./sourcedocs")
    
    if not sourcedocs_path.exists():
        print(f"✗ {sourcedocs_path} directory not found")
        return False
    
    print(f"✓ {sourcedocs_path} directory found")
    
    rst_files = list(sourcedocs_path.glob("*.rst"))
    if rst_files:
        print(f"✓ Found {len(rst_files)} RST files:")
        for f in sorted(rst_files)[:5]:  # Show first 5
            size_kb = f.stat().st_size / 1024
            print(f"  - {f.name} ({size_kb:.1f} KB)")
        if len(rst_files) > 5:
            print(f"  ... and {len(rst_files) - 5} more files")
        return True
    else:
        print("✗ No RST files found in sourcedocs/")
        return False


def test_embeddings():
    """Test HuggingFace embeddings."""
    print("\n" + "="*70)
    print("4. Testing HuggingFace Embeddings")
    print("="*70)
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        
        print("Initializing embeddings model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        
        # Test embedding
        test_text = "This is a test document about machine learning."
        embedding = embeddings.embed_documents([test_text])[0]
        
        print(f"✓ Successfully created embedding")
        print(f"  Embedding dimension: {len(embedding)}")
        return True
        
    except Exception as e:
        print(f"✗ Embeddings test failed: {e}")
        return False


def test_clustering():
    """Test K-Means clustering."""
    print("\n" + "="*70)
    print("5. Testing K-Means Clustering")
    print("="*70)
    
    try:
        import numpy as np
        from sklearn.cluster import KMeans
        
        # Create dummy data
        X = np.random.rand(20, 384)  # 20 samples, 384 dimensions (embedding size)
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        print(f"✓ Successfully clustered {len(X)} samples into 3 clusters")
        unique, counts = np.unique(labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            print(f"  - Cluster {cluster_id}: {count} samples")
        return True
        
    except Exception as e:
        print(f"✗ Clustering test failed: {e}")
        return False


def test_openai_connection():
    """Test OpenAI API connection."""
    print("\n" + "="*70)
    print("6. Testing OpenAI API Connection")
    print("="*70)
    
    try:
        from openai import OpenAI
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = open("/home/yup031/Project1/api-key.txt").read().strip()
        base_url = "https://tritonai-api.ucsd.edu"
        
        # Simple initialization - works with OpenAI, TritonAI, and other compatible APIs
        client = OpenAI(api_key=api_key, base_url=base_url)

        
        # Test with a simple completion
        print("Sending test request to API...")
        response = client.chat.completions.create(
            model="mistral.mistral-large-3-675b-instruct",
            messages=[{"role": "user", "content": "Say 'test successful'"}],
            max_tokens=10
        )
        
        print("✓ Successfully connected to OpenAI API")
        print(f"  Response: {response.choices[0].message.content[:50]}...")
        return True
        
    except Exception as e:
        print(f"✗ OpenAI connection test failed: {e}")
        print("  Note: Verify API key and base URL in .env")
        return False


def test_text_splitting():
    """Test LangChain text splitting."""
    print("\n" + "="*70)
    print("7. Testing Text Splitting")
    print("="*70)
    
    try:
        import numpy as np
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            separators=["\n\n", "\n", " ", ""]
        )
        
        sample_text = "This is a sample document. " * 20  # Create longer text
        chunks = splitter.split_text(sample_text)
        
        print(f"✓ Successfully split text into {len(chunks)} chunks")
        print(f"  Original length: {len(sample_text)} characters")
        print(f"  Average chunk size: {np.mean([len(c) for c in chunks]):.0f} characters")
        return True
        
    except Exception as e:
        print(f"✗ Text splitting test failed: {e}")
        return False


def check_configuration_file():
    """Check if main script has proper configuration."""
    print("\n" + "="*70)
    print("8. Checking Golden QA Script Configuration")
    print("="*70)
    
    script_path = Path("golden_qa_generate.py")
    
    if not script_path.exists():
        print("✗ golden_qa_generate.py not found")
        return False
    
    print("✓ golden_qa_generate.py found")
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check for key components
    checks = [
        ("CONFIG dictionary", "CONFIG = {"),
        ("load_rst_files function", "def load_rst_files"),
        ("cluster_chunks function", "def cluster_chunks"),
        ("generate_qa_pairs function", "def generate_qa_pairs"),
        ("save_outputs function", "def save_outputs"),
        ("Pipeline main", "run_golden_qa_pipeline"),
    ]
    
    all_ok = True
    for name, search_str in checks:
        if search_str in content:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} missing")
            all_ok = False
    
    return all_ok


def run_full_test():
    """Run all validation tests."""
    print("\n" + "="*70)
    print("GOLDEN Q&A GENERATOR - VALIDATION TEST SUITE")
    print("="*70)
    
    results = []
    
    # Run all tests
    results.append(("Dependencies", check_dependencies()))
    results.append(("Sourcedocs", check_sourcedocs()))
    results.append(("Embeddings", test_embeddings()))
    results.append(("Clustering", test_clustering()))
    results.append(("API Connection", test_openai_connection()))
    results.append(("Text Splitting", test_text_splitting()))
    results.append(("Script Config", check_configuration_file()))
    
    # Note: Skipping OpenAI test as it requires valid API key and costs money
    print("\n" + "="*70)
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} - {name}")
    
    total_passed = sum(1 for _, p in results if p)
    total_tests = len(results)
    
    print("\n" + "-"*70)
    print(f"Result: {total_passed}/{total_tests} tests passed")
    print("-"*70)
    
    if total_passed == total_tests:
        print("\n🎉 All validations passed! Ready to run golden_qa_generate.py")
        return True
    else:
        print("\n⚠️  Some validations failed. Please fix issues above before running main script.")
        return False


if __name__ == "__main__":
    success = run_full_test()
    sys.exit(0 if success else 1)
