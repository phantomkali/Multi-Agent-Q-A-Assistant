 
TechCorp Knowledge Assistant
Overview
The TechCorp Knowledge Assistant is a Retrieval-Augmented Generation (RAG)-powered Q&A system that answers queries using a document collection, mathematical calculations, or word definitions. It features a modular design with vector-based retrieval, a free Hugging Face model for answer generation (with a rule-based fallback), and a custom agent to route queries to appropriate tools, all accessible via a Streamlit web UI.
Architecture
•	Data Ingestion: Loads Q&A pairs from text files in the docs directory, chunking by double newlines.
•	Vector Store & Retrieval: Uses sentence-transformers (all-MiniLM-L6-v2) to embed chunks and FAISS for fast vector search, retrieving the top 3 relevant chunks per query.
•	Answer Generation: Leverages Hugging Face models (e.g., distilgpt2, gpt2, facebook/opt-125m, google/flan-t5-small) for free text generation. Includes a rule-based fallback for keyword matching if the transformers library or model fails.
•	Agentic Workflow: A custom SimpleAgentExecutor routes queries to one of three tools based on keywords:
o	Calculator: Evaluates math expressions (e.g., "Calculate 2 + 2").
o	Dictionary: Fetches definitions from the free dictionaryapi.dev API (e.g., "Define apple").
o	RAG: Combines retrieval and generation for general queries (e.g., "What products does TechCorp offer?").
•	Interface: Streamlit UI allows model selection, query input, and displays answers, decision logs, and retrieved context (for RAG queries).
Key Design Choices
•	FAISS: Open-source, lightweight, and efficient for vector indexing on small datasets.
•	Sentence Transformers: Compact model for embeddings, balancing speed and accuracy.
•	Hugging Face Models: Free, local LLMs eliminate API costs; multiple options (distilgpt2, etc.) provide flexibility.
•	Rule-Based Fallback: Ensures functionality without transformers, using keyword matching for common queries.
•	Custom Agent: Simplistic, keyword-based routing avoids heavy dependencies like LangChain’s full agent framework, improving portability.
•	Streamlit UI: User-friendly, with model selection and clear display of answers and logs.
•	Free Dictionary API: No authentication, reliable for definitions, and cost-free.
Prerequisites
1.	Install Dependencies:
2.	pip install sentence-transformers faiss-cpu requests streamlit torch langchain-core
Optionally, for Hugging Face models:
pip install transformers
3.	Prepare Documents:
o	Create a docs directory in the same directory as the script.
o	Add 3–5 text files (e.g., FAQ1.txt) with Q&A pairs, separated by double newlines. Example:
o	Q: What products does TechCorp offer?
o	A: TechCorp offers AI Assistant, Cloud Storage Pro, and Security Shield.
o	
o	Q: When was TechCorp founded?
o	A: TechCorp was founded in 2010 by Jane Smith.
4.	Hardware:
o	CPU is sufficient for distilgpt2 or rule-based mode.
o	GPU (optional) accelerates larger models like gpt2 if using transformers.
How to Run
1.	Save the code as knowledge_assistant.py.
2.	Set up the docs directory with text files.
3.	Install dependencies (see above).
4.	Run the Streamlit app:
5.	streamlit run knowledge_assistant.py
6.	Open the provided URL (e.g., http://localhost:8501) in a browser.
7.	In the UI:
o	Select a Hugging Face model from the sidebar (or use rule-based mode if transformers is not installed).
o	Enter queries like:
	"What products does TechCorp offer?" (RAG)
	"Calculate 2 + 2" (Calculator)
	"Define apple" (Dictionary)
o	View the answer and decision log (tool used, input, observation).
How It Works
•	Query Processing: The custom agent inspects the query for keywords (e.g., "calculate", "define") to select a tool.
•	Calculator: Uses Python’s eval for math expressions.
•	Dictionary: Queries dictionaryapi.dev for word definitions.
•	RAG: Retrieves relevant document chunks via FAISS, then generates an answer using the selected Hugging Face model or rule-based logic.
•	Fallback: If transformers is unavailable or the model fails, the system uses predefined answers for common queries (e.g., "products", "founded") or extracts sentences from context.
•	UI: Streamlit shows the answer, logs each step, and, for RAG, the retrieved context.
Notes
•	Model Selection: distilgpt2 is lightweight (~500MB download); larger models like gpt2 need more memory. flan-t5-small is tuned for Q&A but slower.
•	Rule-Based Mode: Active without transformers, using keyword matching for predefined queries or context extraction.
•	Dictionary API: Free but rate-limited; handle errors for production use.
•	Performance: Hugging Face models are less accurate than paid LLMs (e.g., OpenAI). For better results, consider fine-tuning or larger models if resources allow.
•	Docs Directory: Must exist with valid text files, or the script will fail.

