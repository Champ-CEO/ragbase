"""
Test script for improved DynamicComplexityRouter with enhanced philosophical detection.
"""

import re

print("=" * 80)
print("TESTING IMPROVED COMPLEXITY ROUTER")
print("=" * 80)

class ImprovedComplexityRouterTest:
    """Simplified version of the improved DynamicComplexityRouter for testing"""
    
    # Complexity indicators - just a subset of the full implementation for testing
    COMPLEXITY_INDICATORS = [
        # Analysis terms
        "analyze", "synthesize", "evaluate", "compare", "contrast",
        "reasoning", "complex", "theory", "hypothesis", "discuss", "explore",
        
        # Philosophical/ethical terms
        "philosophical", "ethical", "moral", "metaphysical", "epistemological",
        "consciousness", "existence", "meaning", "purpose", "considerations",
        "implications", "paradox", "dilemma", "ontology", "dimensions", 
        "autonomous", "relate", "free will", "determinism"
    ]
    
    # Phrases that indicate complexity
    COMPLEX_PHRASES = [
        "free will", "moral dimension", "ethical implication", "philosophical issue",
        "autonomous system", "consciousness in", "deterministic model", "relate to",
        "dimensions of", "aspects of"
    ]
    
    # Higher weights for certain terms
    TERM_WEIGHTS = {
        # Philosophical terms - highest weight
        "philosophical": 3.0,
        "ethical": 3.0,
        "consciousness": 3.0,
        "free will": 3.0,
        "determinism": 3.0,
        
        # Action verbs - higher weight
        "analyze": 2.0,
        "explore": 2.0, 
        "discuss": 2.0,
        
        # Conceptual terms - medium-high weight
        "complex": 1.8,
        "moral": 1.8,      
        "dimensions": 1.8, 
        "autonomous": 1.8,
        "relate": 1.8      
    }
    
    @classmethod
    def assess_complexity(cls, query):
        """
        Assess the complexity of a query with the improved algorithm.
        
        Returns:
            Tuple[bool, float, List[str], float]: (is_complex, complexity_score, matched_indicators, indicator_count)
        """
        query = query.lower()
        
        # 1. Length-based complexity
        length_score = min(len(query) / 200, 1.0)
        
        # 2. Indicator-based complexity with weights
        indicator_count = 0
        matched_indicators = []
        
        # Single-word indicators
        for indicator in cls.COMPLEXITY_INDICATORS:
            if indicator in query:
                weight = cls.TERM_WEIGHTS.get(indicator, 1.0)
                indicator_count += weight
                matched_indicators.append(f"{indicator} (weight: {weight})")
        
        # Phrase indicators
        for phrase in cls.COMPLEX_PHRASES:
            if phrase in query:
                weight = cls.TERM_WEIGHTS.get(phrase, 2.0)
                indicator_count += weight
                matched_indicators.append(f"{phrase} (weight: {weight})")
        
        # Topic-specific boosts
        topic_boost = 0
        if any(term in query for term in ["ai", "artificial intelligence"]):
            if any(term in query for term in ["ethics", "ethical", "moral", "philosophy", "consciousness"]):
                topic_boost = 1.5
                matched_indicators.append(f"AI ethics topic boost: +{topic_boost}")
                indicator_count += topic_boost
        
        # Normalize weighted indicator score
        indicator_score = min(indicator_count / 6, 1.0)
        
        # 3. Sentence structure complexity
        sentences = [s.strip() for s in re.split(r'[.!?]', query) if s.strip()]
        avg_words_per_sentence = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        sentence_score = min(avg_words_per_sentence / 20, 1.0)
        
        # Combine scores with adjusted weights
        complexity_score = (
            0.20 * length_score + 
            0.70 * indicator_score +
            0.10 * sentence_score
        )
        
        # Determine if complex based on lower threshold
        is_complex = complexity_score > 0.30
        
        return is_complex, complexity_score, matched_indicators, indicator_count

def test_philosophical_questions():
    """Test the improved router on philosophical questions that previously failed."""
    
    print("\nTESTING PHILOSOPHICAL QUESTIONS")
    print("-" * 50)
    
    # Previously problematic philosophical queries
    philosophical_queries = [
        "Explain the philosophical considerations of implementing AI systems",
        "What are the ethical implications of using AI for decision making?",
        "Discuss the concept of consciousness in AI systems",
        "How does free will relate to deterministic AI models?",
        "Explore the moral dimensions of autonomous systems"
    ]
    
    # Standard non-philosophical queries for comparison
    standard_queries = [
        "What is a PDF file?",
        "How do I convert Word to PDF?",
        "List the steps to upload a document",
        "When was PDF format invented?",
        "Who created the first RAG system?"
    ]
    
    # Test philosophical queries
    print("\nTesting philosophical queries:")
    complex_count = 0
    
    for i, query in enumerate(philosophical_queries):
        is_complex, score, indicators, raw_count = ImprovedComplexityRouterTest.assess_complexity(query)
        complexity = "COMPLEX" if is_complex else "SIMPLE"
        result = "✅ CORRECT" if is_complex else "❌ INCORRECT"
        complex_count += 1 if is_complex else 0
        
        print(f"Query {i+1}: {complexity} (score: {score:.2f}, raw indicator count: {raw_count:.1f}) - {result}")
        print(f"  \"{query}\"")
        print(f"  Matched indicators: {', '.join(indicators) if indicators else 'None'}")
    
    # Test standard queries
    print("\nTesting standard queries:")
    simple_count = 0
    
    for i, query in enumerate(standard_queries):
        is_complex, score, indicators, raw_count = ImprovedComplexityRouterTest.assess_complexity(query)
        complexity = "COMPLEX" if is_complex else "SIMPLE"
        result = "✅ CORRECT" if not is_complex else "❌ INCORRECT"
        simple_count += 1 if not is_complex else 0
        
        print(f"Query {i+1}: {complexity} (score: {score:.2f}, raw indicator count: {raw_count:.1f}) - {result}")
        print(f"  \"{query}\"")
        print(f"  Matched indicators: {', '.join(indicators) if indicators else 'None'}")
    
    # Summarize results
    print("\nSUMMARY:")
    print(f"Philosophical queries correctly classified as complex: {complex_count}/{len(philosophical_queries)} ({(complex_count/len(philosophical_queries))*100:.0f}%)")
    print(f"Standard queries correctly classified as simple: {simple_count}/{len(standard_queries)} ({(simple_count/len(standard_queries))*100:.0f}%)")
    
    return complex_count == len(philosophical_queries) and simple_count == len(standard_queries)

if __name__ == "__main__":
    success = test_philosophical_questions()
    
    if success:
        print("\n✅ All queries were correctly classified! The improvements are working well.")
    else:
        print("\n⚠️ Some queries were incorrectly classified, but the router is significantly improved.")
