import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import sys

# Load environment variables
load_dotenv()

# Check if API key is set
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("Error: GROQ_API_KEY not found in environment variables")
    sys.exit(1)

print(f"API key found: {api_key[:5]}...{api_key[-5:]}")

# Test general model
try:
    print("\nTesting general model (llama-3.3-70b-versatile)...")
    general_llm = ChatGroq(
        temperature=0.0,
        model_name="llama-3.3-70b-versatile",
        max_tokens=100
    )
    
    response = general_llm.invoke("What is retrieval-augmented generation?")
    print("Response from general model:", response.content[:200] + "...")
    print("✅ General model test successful")
except Exception as e:
    print(f"❌ General model test failed: {str(e)}")

# Test complex model
try:
    print("\nTesting complex model (deepseek-r1-distill-llama-70b)...")
    complex_llm = ChatGroq(
        temperature=0.0,
        model_name="deepseek-r1-distill-llama-70b",
        max_tokens=100
    )
    
    response = complex_llm.invoke("Explain the concept of vector embeddings in RAG systems")
    print("Response from complex model:", response.content[:200] + "...")
    print("✅ Complex model test successful")
except Exception as e:
    print(f"❌ Complex model test failed: {str(e)}")
