"""
Token optimization and model routing mechanisms for Groq API integration.
Implements strategies for efficient token usage and accuracy-preserving query routing.
"""

import re
from typing import List, Tuple

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage

from ragbase.model import create_llm


class TokenOptimizationPipeline:
    """
    Pipeline for optimizing token usage in prompts and responses.
    Implements techniques to reduce token consumption while preserving meaning.
    """

    @staticmethod
    def truncate_text(text: str, max_length: int = 8000) -> str:
        """Truncate text to a maximum length while preserving meaning."""
        if len(text) <= max_length:
            return text
        
        # Keep intro and conclusion, remove from middle if needed
        intro_length = max_length // 3
        conclusion_length = max_length // 3
        middle_length = max_length - intro_length - conclusion_length
        
        intro = text[:intro_length]
        conclusion = text[-conclusion_length:]
        
        return intro + "..." + text[len(text) - middle_length - conclusion_length:-conclusion_length] + conclusion

    @staticmethod
    def optimize_prompt(prompt: str) -> str:
        """Optimize a prompt to reduce token usage."""
        # Remove redundant whitespace
        prompt = re.sub(r'\s+', ' ', prompt)
        prompt = prompt.strip()
        
        # Remove unnecessary punctuation repetition
        prompt = re.sub(r'\.{2,}', '...', prompt)
        prompt = re.sub(r'!{2,}', '!', prompt)
        prompt = re.sub(r'\?{2,}', '?', prompt)
        
        return prompt

    @staticmethod
    def optimize_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
        """Optimize a list of messages to reduce token usage."""
        optimized_messages = []
        
        for message in messages:
            if hasattr(message, "content") and isinstance(message.content, str):
                optimized_content = TokenOptimizationPipeline.optimize_prompt(message.content)
                
                # Create a new message of the same type but with optimized content
                if isinstance(message, HumanMessage):
                    optimized_messages.append(HumanMessage(content=optimized_content))
                else:
                    # For other message types, preserve the original type but update content
                    message_dict = message.dict()
                    message_dict["content"] = optimized_content
                    optimized_messages.append(type(message)(**message_dict))
            else:
                optimized_messages.append(message)
                
        return optimized_messages


class DynamicComplexityRouter:
    """
    Routes queries to the appropriate model based on complexity assessment.
    
    This implements adaptive routing to balance performance and cost:
    - Simple queries go to faster, more cost-effective models
    - Complex queries requiring deeper reasoning go to more powerful models
    """
    
    # Complexity indicators - words/phrases that might indicate a complex query
    COMPLEXITY_INDICATORS = [
        # Analysis and evaluation terms
        "analyze", "synthesize", "evaluate", "compare", "contrast",
        "implications", "consequences", "relationship between",
        "multi-step", "complex", "reasoning", "why", "how does",
        "explain in detail", "elaborate", "what would happen if",
        "critically", "argument", "evidence", "theory", "hypothesis",
        "infer", "deduce", "prove", "concept", "discuss", "explore",
        
        # Technical/scientific terms
        "technical", "scientific", "mathematical", "algorithm",
        "architecture", "framework", "implementation", "methodology",
        "mechanism", "approach", "statistics", "correlation",
        
        # Philosophical/ethical terms (expanded)
        "philosophical", "ethical", "moral", "metaphysical", "epistemological",
        "consciousness", "existence", "meaning", "purpose", "free will",
        "determinism", "relativism", "utilitarianism", "deontological",
        "virtue ethics", "social contract", "rights", "justice",
        "dialectic", "ontology", "phenomenology", "hermeneutics",
        "considerations", "implications", "paradox", "dilemma",
        "dimensions", "aspects", "autonomy", "autonomous", "relate",
        "connection", "relationship", "impact", 
        
        # Domain-specific reasoning
        "domain knowledge", "specialized", "expert", "discipline",
        "paradigm", "school of thought", "perspective", "worldview"
    ]
    
    # Bigram/phrase matching for complex concepts
    COMPLEX_PHRASES = [
        "free will", "moral dimension", "ethical implication", "philosophical issue",
        "autonomous system", "consciousness in", "deterministic model", "causal relationship",
        "relate to", "connection between", "dimensions of", "aspects of"
    ]
    
    # Higher weights for certain categories of terms
    TERM_WEIGHTS = {
        # Philosophical terms - highest weight
        "philosophical": 3.0,
        "ethical": 3.0,
        "epistemological": 3.0,
        "ontology": 3.0,
        "metaphysical": 3.0,
        "consciousness": 3.0,
        "free will": 3.0,
        "moral dimension": 3.0,
        "ethical implication": 3.0,
        "determinism": 3.0,
        
        # Action verbs indicating deep analysis - higher weight
        "analyze": 2.0,
        "synthesize": 2.0,
        "evaluate": 2.0,
        "explore": 2.0,  # Added higher weight for "explore"
        "discuss": 2.0,  # Added higher weight for "discuss"
        
        # Conceptual terms - medium-high weight
        "complex": 1.8,
        "theory": 1.8,
        "implications": 1.8,
        "considerations": 1.8,
        "paradox": 1.8,
        "dilemma": 1.8,
        "moral": 1.8,      # Increased weight for "moral"
        "dimensions": 1.8,  # Added higher weight for "dimensions"
        "autonomous": 1.8,  # Added higher weight for "autonomous"
        "relate": 1.8       # Added higher weight for "relate"
    }
    
    @classmethod
    def assess_complexity(cls, query: str) -> Tuple[bool, float]:
        """
        Assess the complexity of a query based on various heuristics.
        
        Returns:
            Tuple[bool, float]: (is_complex, complexity_score)
        """
        query = query.lower()
        
        # 1. Length-based complexity (longer queries tend to be more complex)
        length_score = min(len(query) / 200, 1.0)  # Normalize to 0-1
        
        # 2. Indicator-based complexity (with weighted terms)
        indicator_count = 0
        
        # Check for single-word indicators
        for indicator in cls.COMPLEXITY_INDICATORS:
            if indicator in query:
                weight = cls.TERM_WEIGHTS.get(indicator, 1.0)  # Default weight is 1.0
                indicator_count += weight
        
        # Check for phrase/bigram indicators
        for phrase in cls.COMPLEX_PHRASES:
            if phrase in query:
                weight = cls.TERM_WEIGHTS.get(phrase, 2.0)  # Default weight for phrases is 2.0
                indicator_count += weight
        
        # Topic-specific boosts
        if any(term in query for term in ["ai", "artificial intelligence"]):
            if any(term in query for term in ["ethics", "ethical", "moral", "philosophy", "consciousness", "free will", "determinism"]):
                indicator_count += 1.5  # Boost for AI ethics/philosophy questions
        
        # Normalize weighted indicator score (with lower divisor for easier qualification)
        indicator_score = min(indicator_count / 6, 1.0)  # Adjusted from 8 to 6 for easier qualification
        
        # 3. Sentence structure complexity
        sentences = [s.strip() for s in re.split(r'[.!?]', query) if s.strip()]
        avg_words_per_sentence = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        sentence_score = min(avg_words_per_sentence / 20, 1.0)  # Normalize to 0-1
        
        # Combine scores with adjusted weights
        complexity_score = (
            0.20 * length_score + 
            0.70 * indicator_score +  # Increased weight for indicators from 0.60 to 0.70
            0.10 * sentence_score     # Decreased weight for sentence structure from 0.15 to 0.10
        )
        
        # Determine if complex based on lower threshold
        is_complex = complexity_score > 0.30  # Reduced threshold from 0.35 to 0.30
        
        return is_complex, complexity_score
    
    @classmethod
    def get_appropriate_llm(cls, query: str) -> Tuple[BaseLanguageModel, bool]:
        """
        Get the appropriate LLM based on query complexity.
        
        Args:
            query: The user query to analyze
            
        Returns:
            Tuple[BaseLanguageModel, bool]: (selected_llm, is_complex)
        """
        is_complex, score = cls.assess_complexity(query)
        
        # Create the appropriate LLM
        llm = create_llm(use_complex_model=is_complex)
        
        return llm, is_complex
