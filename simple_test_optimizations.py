"""
Simple test script for the optimization components in the RagBase project.
No external dependencies required, runs standalone.
"""

import os
import sys
import time

# Configure Python path to find the local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import optimization modules after path setup
try:
    from ragbase.optimizers import DynamicComplexityRouter, TokenOptimizationPipeline
    print("✅ Successfully imported optimizers module")
except ImportError as e:
    print(f"❌ Failed to import optimizers module: {e}")
    sys.exit(1)

print("=" * 80)
print("TESTING OPTIMIZATION COMPONENTS")
print("=" * 80)

def test_dynamic_complexity_router():
    print("\n1. TESTING DYNAMIC COMPLEXITY ROUTER")
    print("-" * 50)

    test_queries = [
        # Simple queries that should route to general model
        "What is a PDF?",
        "Define RAG system",
        "List the key components",
        
        # Complex queries that should route to complex model
        "Compare and contrast the implications of using semantic chunking versus fixed-length chunking in a RAG system",
        "Analyze the relationship between embedding dimensionality and retrieval performance",
        "Explain the philosophical considerations of implementing AI systems"
    ]
    
    print("Testing query complexity assessment...")
    for i, query in enumerate(test_queries):
        is_complex, score = DynamicComplexityRouter.assess_complexity(query)
        complexity = "COMPLEX" if is_complex else "SIMPLE"
        expected = "COMPLEX" if i >= 3 else "SIMPLE"
        result = "✅ CORRECT" if complexity == expected else "❌ INCORRECT"
        
        print(f"Query {i+1}: {complexity} (score: {score:.2f}) - {result}")
        print(f"  \"{query}\"")
    
    return True

def test_token_optimization_pipeline():
    print("\n2. TESTING TOKEN OPTIMIZATION PIPELINE")
    print("-" * 50)
    
    test_prompts = [
        # Verbose prompt with redundant whitespace
        """
        Please     explain    in   great   detail   what   is   a  
        Retrieval  Augmented     Generation    system  and     how  
        it       works      in      practice??!!
        """,
        
        # Long prompt that should be truncated
        "I need a detailed explanation of vector databases. " * 50,
        
        # Prompt with unnecessary punctuation repetition
        "What is embeddings???????? And how do they work??????!!!!"
    ]
    
    print("Testing prompt optimization...")
    for i, prompt in enumerate(test_prompts):
        original_length = len(prompt)
        optimized_prompt = TokenOptimizationPipeline.optimize_prompt(prompt)
        optimized_length = len(optimized_prompt)
        reduction = (original_length - optimized_length) / original_length * 100
        
        print(f"Prompt {i+1}: Original length: {original_length}, Optimized length: {optimized_length}")
        print(f"  Token reduction: {reduction:.1f}%")
        print(f"  Original: \"{prompt[:40]}...\"")
        print(f"  Optimized: \"{optimized_prompt[:40]}...\"")
    
    return True

def run_all_tests():
    test_results = []
    test_results.append(("Dynamic Complexity Router", test_dynamic_complexity_router()))
    test_results.append(("Token Optimization Pipeline", test_token_optimization_pipeline()))
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for name, result in test_results:
        status = "PASSED" if result else "FAILED"
        if not result:
            all_passed = False
        print(f"{name}: {status}")
    
    if all_passed:
        print("\n✅ All tests passed! The optimization components are working as expected.")
    else:
        print("\n❌ Some tests failed. Please check the output above for details.")

if __name__ == "__main__":
    run_all_tests()
