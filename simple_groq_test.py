import os

# Check if API key is set
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("Error: GROQ_API_KEY not found in environment variables")
    print("Make sure to run this script in an environment where .env is loaded")
    exit(1)

print(f"API key found: {api_key[:5]}...{api_key[-5:]}")

try:
    # Import required packages
    from groq import Groq
    
    # Initialize Groq client
    client = Groq(api_key=api_key)
    
    # Test general model
    print("\nTesting general model (llama-3.3-70b-versatile)...")
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": "What is retrieval-augmented generation?"}
        ],
        temperature=0.0,
        max_tokens=100
    )
    print("Response from general model:", completion.choices[0].message.content[:200] + "...")
    print("✅ General model test successful")
    
    # Test complex model
    print("\nTesting complex model (deepseek-r1-distill-llama-70b)...")
    completion = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[
            {"role": "user", "content": "Explain the concept of vector embeddings in RAG systems"}
        ],
        temperature=0.0,
        max_tokens=100
    )
    print("Response from complex model:", completion.choices[0].message.content[:200] + "...")
    print("✅ Complex model test successful")
    
except ImportError:
    print("❌ Error: Required package 'groq' is not installed.")
    print("Please install it using: pip install groq")
except Exception as e:
    print(f"❌ Test failed: {str(e)}")
