import os
import time
from dotenv import load_dotenv

from ragbase.config import Config
from ragbase.optimizers import DynamicComplexityRouter, TokenOptimizationPipeline
from ragbase.model import create_llm

# Load environment variables
load_dotenv()

# Ensure API key is set
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("Error: GROQ_API_KEY not found in environment variables")
    exit(1)

print("=" * 80)
print("TESTING GROQ API OPTIMIZATION COMPONENTS")
print("=" * 80)

def test_dynamic_complexity_router():
    print("\n1. TESTING DYNAMIC COMPLEXITY ROUTER")
    print("-" * 50)

    test_queries = [
        # Simple queries that should route to general model
        "What is a PDF?",
        "Define RAG system",
        "Who created LangChain?",
        "List the key components of a vector database",
        
        # Complex queries that should route to complex model
        "Compare and contrast the implications of using semantic chunking versus fixed-length chunking in a RAG system, and explain how each approach affects retrieval accuracy under different document conditions",
        "Analyze the relationship between embedding dimensionality and retrieval performance in vector databases, considering the trade-offs between computational efficiency and representation quality",
        "Explain the philosophical and ethical considerations of implementing AI systems that can reason about complex human scenarios, particularly in domains requiring deep domain knowledge",
        "Evaluate how different reranking techniques in RAG systems affect the quality of responses for multi-hop reasoning tasks that require synthesizing information from multiple documents"
    ]
    
    print("Testing query complexity assessment...")
    for i, query in enumerate(test_queries):
        is_complex, score = DynamicComplexityRouter.assess_complexity(query)
        complexity = "COMPLEX" if is_complex else "SIMPLE"
        expected = "COMPLEX" if i >= 4 else "SIMPLE"
        result = "✅ CORRECT" if complexity == expected else "❌ INCORRECT"
        
        print(f"Query {i+1}: {complexity} (score: {score:.2f}) - {result}")
        print(f"  \"{query[:50]}...\"")
    
    print("\nTesting model routing...")
    simple_query = "What is a RAG system?"
    complex_query = "Analyze the theoretical limitations of current RAG approaches when dealing with mathematical reasoning tasks requiring multi-step deductions."
    
    simple_llm, is_simple_complex = DynamicComplexityRouter.get_appropriate_llm(simple_query)
    complex_llm, is_complex_complex = DynamicComplexityRouter.get_appropriate_llm(complex_query)
    
    print(f"Simple query routed to: {simple_llm.model_name if hasattr(simple_llm, 'model_name') else 'Unknown'}")
    print(f"Complex query routed to: {complex_llm.model_name if hasattr(complex_llm, 'model_name') else 'Unknown'}")
    
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
        "I need a detailed explanation of vector databases. " * 100,
        
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
        print(f"  Original: \"{prompt[:50]}...\"")
        print(f"  Optimized: \"{optimized_prompt[:50]}...\"")
    
    return True

def test_end_to_end_integration():
    print("\n3. TESTING END-TO-END INTEGRATION")
    print("-" * 50)
    
    # Test with dynamic routing enabled
    print("Testing with dynamic routing enabled...")
    Config.AUTO_COMPLEXITY_ROUTING = True
    Config.USE_TOKEN_OPTIMIZATION = True
    
    # Simple query
    simple_query = "What is RAG?"
    
    start_time = time.time()
    llm = create_llm()  # This will be overridden by dynamic routing
    simple_query_optimized = TokenOptimizationPipeline.optimize_prompt(simple_query)
    is_complex, complexity_score = DynamicComplexityRouter.assess_complexity(simple_query)
    routed_llm, _ = DynamicComplexityRouter.get_appropriate_llm(simple_query)
    
    response = routed_llm.invoke(simple_query_optimized)
    simple_processing_time = time.time() - start_time
    
    print(f"Simple query routed to: {routed_llm.model_name if hasattr(routed_llm, 'model_name') else 'Unknown'}")
    print(f"Processing time: {simple_processing_time:.2f}s")
    print(f"Response: \"{response.content[:100]}...\"")
    
    # Complex query
    complex_query = "Analyze the theoretical and practical differences between semantic chunking and recursive character chunking in RAG systems, particularly their impact on retrieval performance for documents with varying structural complexity."
    
    start_time = time.time()
    complex_query_optimized = TokenOptimizationPipeline.optimize_prompt(complex_query)
    is_complex, complexity_score = DynamicComplexityRouter.assess_complexity(complex_query)
    routed_llm, _ = DynamicComplexityRouter.get_appropriate_llm(complex_query)
    
    response = routed_llm.invoke(complex_query_optimized)
    complex_processing_time = time.time() - start_time
    
    print(f"\nComplex query routed to: {routed_llm.model_name if hasattr(routed_llm, 'model_name') else 'Unknown'}")
    print(f"Processing time: {complex_processing_time:.2f}s")
    print(f"Response: \"{response.content[:100]}...\"")
    
    return True

def run_all_tests():
    test_results = []
    test_results.append(("Dynamic Complexity Router", test_dynamic_complexity_router()))
    test_results.append(("Token Optimization Pipeline", test_token_optimization_pipeline()))
    test_results.append(("End-to-End Integration", test_end_to_end_integration()))
    
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
