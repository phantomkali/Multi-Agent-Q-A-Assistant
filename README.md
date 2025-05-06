# Multi-Agent-Q-A-Assistant
Knowledge Assistant
Overview
The Knowledge Assistant is a Retrieval-Augmented Generation (RAG)-powered multi-agent Q&A system that answers user queries by leveraging a document collection, mathematical calculations, or word definitions. It uses a modular architecture with a retrieval pipeline, an LLM for answer generation, and an agentic workflow to route queries to appropriate tools, all accessible via a Streamlit web UI.
Architecture

Data Ingestion: Reads and chunks Q&A pairs from text files in the docs directory, splitting by double newlines.
Vector Store & Retrieval: Uses sentence-transformers (all-MiniLM-L6-v2) to embed document chunks and FAISS for efficient vector indexing. Retrieves top 3 relevant chunks for each query.
LLM Integration: Employs OpenAI’s gpt-3.5-turbo-instruct for generating natural-language answers based on retrieved context.
Agentic Workflow: A LangChain agent (create_openai_functions_agent) routes queries to one of three tools based on intent:
Calculator: Evaluates mathematical expressions (e.g., "2 + 2").
Dictionary: Fetches word definitions from the free dictionaryapi.dev API.
RAG: Combines retrieval and LLM generation for general queries.


Interface: Streamlit web UI displays the answer, decision log, and retrieved context (for RAG queries).

Key Design Choices

FAISS for Retrieval: Lightweight, fast, and open-source, suitable for small document collections.
Sentence Transformers: Compact and efficient for embedding text, balancing performance and resource use.
LangChain Agent: Simplifies tool routing with a clear, extensible framework, using OpenAI functions for precise tool selection.
Streamlit UI: Quick to set up, user-friendly, and effective for demo purposes, showing both results and agent decisions.
OpenAI LLM: Chosen for robust natural-language generation, though it requires an API key (alternatives like Hugging Face models could be used for free).
Free Dictionary API: No authentication needed, reliable for word definitions, and aligns with cost-free requirements for some components.

Prerequisites

Install Dependencies:pip install sentence-transformers faiss-cpu openai requests streamlit langchain-core langchain-openai


Set Up OpenAI API Key:
Replace openai.api_key = "" in the code with your OpenAI API key.


Prepare Documents:
Create a docs directory in the same directory as the script.
Add 3–5 text files (e.g., FAQ1.txt) with Q&A pairs, separated by double newlines. Example:Q: What products does TechCorp offer?
A: TechCorp offers smart home devices, wearable technology, and enterprise software solutions.

Q: Where can I buy TechCorp products?
A: TechCorp products are available on our official website and authorized retailers.





How to Run

Save the code as knowledge_assistant.py.
Ensure the docs directory is set up with text files.
Install dependencies (see above).
Set your OpenAI API key in the script.
Run the Streamlit app:streamlit run knowledge_assistant.py


Open the provided URL (e.g., http://localhost:8501) in a browser.
Enter queries like:
"What products does TechCorp offer?" (RAG)
"Calculate 2 + 2" (Calculator)
"Define apple" (Dictionary)



Notes

The assistant logs each decision step (tool used, input, and observation) for transparency.
The OpenAI API requires a paid key; for a fully free alternative, consider replacing the LLM with a Hugging Face model (e.g., distilgpt2).
The dictionary API is free and rate-limited; handle errors gracefully for production use.
Ensure the docs directory exists, or the script will fail to load documents.

