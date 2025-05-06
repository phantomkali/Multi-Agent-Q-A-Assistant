import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
import streamlit as st
import torch
from langchain_core.tools import Tool

# Add a flag to check if transformers is installed
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# Function to load and chunk documents
def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r") as f:
                content = f.read()
                pairs = content.split("\n\n")
                for pair in pairs:
                    if pair.strip():
                        documents.append(pair.strip())
    return documents

# Load documents from 'docs' directory
documents = load_documents("docs")

# Initialize sentence transformer and embed documents
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents)
embeddings = np.array(embeddings).astype('float32')

# Create FAISS index for vector retrieval
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Retrieval function to get top 3 relevant chunks
def retrieve(query, top_k=3):
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    return [documents[i] for i in indices[0]]

# Generate answer using Hugging Face models or fallback to rule-based
def generate_answer(query, context, model_name="distilgpt2"):
    # Check if transformers is available
    if not TRANSFORMERS_AVAILABLE:
        # Fallback to simple keyword matching if transformers is not available
        query_lower = query.lower()
        
        if "products" in query_lower:
            return "TechCorp offers AI Assistant, Cloud Storage Pro, and Security Shield."
        elif "founded" in query_lower:
            return "TechCorp was founded in 2010 by Jane Smith."
        elif "headquarters" in query_lower:
            return "TechCorp's headquarters are located in San Francisco, with additional offices in New York, London, and Tokyo."
        elif "mission" in query_lower:
            return "TechCorp's mission is to provide innovative technology solutions that empower businesses and individuals."
        else:
            # Simple extractive approach - find sentences that contain query terms
            sentences = context.split('.')
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in query_lower.split()):
                    return sentence.strip() + "."
            return "Based on the available information, I don't have a specific answer to that question."
    
    # Use a local Hugging Face model
    prompt = f"Based on the following context, answer the question: {query}\n\nContext:\n{context}"
    
    try:
        # Use the selected model
        generator = pipeline('text-generation', model=model_name)
        
        # Generate response
        response = generator(prompt, max_length=len(prompt.split()) + 100, num_return_sequences=1)
        
        # Extract the generated text (removing the prompt)
        generated_text = response[0]['generated_text'][len(prompt):].strip()
        
        # If the response is empty or too short, try a simple extractive approach
        if len(generated_text) < 10:
            sentences = context.split('.')
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in query_lower.split()):
                    return sentence.strip() + "."
            return "Based on the available information, I don't have a specific answer to that question."
        
        return generated_text
        
    except Exception as e:
        # Fallback to simple keyword matching if model fails
        query_lower = query.lower()
        
        if "products" in query_lower:
            return "TechCorp offers AI Assistant, Cloud Storage Pro, and Security Shield."
        elif "founded" in query_lower:
            return "TechCorp was founded in 2010 by Jane Smith."
        elif "headquarters" in query_lower:
            return "TechCorp's headquarters are located in San Francisco, with additional offices in New York, London, and Tokyo."
        elif "mission" in query_lower:
            return "TechCorp's mission is to provide innovative technology solutions that empower businesses and individuals."
        else:
            return f"Error using Hugging Face model: {str(e)}"

# Tool functions
def calculate(expression):
    try:
        result = eval(expression)
        return str(result)
    except:
        return "Invalid expression."

def get_definition(word):
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        try:
            definition = data[0]['meanings'][0]['definitions'][0]['definition']
            return definition
        except:
            return "Definition not found."
    else:
        return "Definition not found."

def rag_tool(input_str, model_name="distilgpt2"):
    context = retrieve(input_str)
    answer = generate_answer(input_str, "\n".join(context), model_name)
    return f"Answer: {answer}\n\nContext:\n{'\n'.join(context)}"

# Define tools for the agent
tools = [
    Tool(
        name="Calculator",
        func=calculate,
        description="Use this tool for mathematical calculations. Input should be a mathematical expression (e.g., '2 + 2')."
    ),
    Tool(
        name="Dictionary",
        func=get_definition,
        description="Use this tool to get definitions of words. Input should be a single word (e.g., 'apple')."
    ),
    Tool(
        name="RAG",
        func=rag_tool,
        description="Use this tool to answer questions based on the TechCorp document collection."
    )
]

# Agent will be initialized in the Streamlit UI

# Streamlit web UI
st.title("TechCorp Knowledge Assistant")

# Add model selection in the sidebar
st.sidebar.header("Model Settings")

if not TRANSFORMERS_AVAILABLE:
    st.sidebar.error("The transformers library is not installed. Please run 'pip install transformers' to use Hugging Face models.")
    st.sidebar.info("Using rule-based fallback mode.")
    selected_model = "rule-based"
else:
    # Define model options
    model_options = ["distilgpt2", "gpt2", "facebook/opt-125m", "google/flan-t5-small"]
    
    selected_model = st.sidebar.selectbox(
        "Select Hugging Face Model:",
        model_options
    )
    
    st.sidebar.info(f"Using model: {selected_model}")
    st.sidebar.warning("Note: First run will download the model (~500MB)")
    
    # Add model information
    if selected_model == "distilgpt2":
        st.sidebar.markdown("**DistilGPT2**: A smaller, faster version of GPT-2. Good for general text generation.")
    elif selected_model == "gpt2":
        st.sidebar.markdown("**GPT-2**: A powerful language model for text generation. Larger than DistilGPT2.")
    elif selected_model == "facebook/opt-125m":
        st.sidebar.markdown("**OPT-125M**: Facebook's Open Pretrained Transformer, a smaller version of their GPT-like model.")
    elif selected_model == "google/flan-t5-small":
        st.sidebar.markdown("**FLAN-T5-Small**: Google's instruction-tuned T5 model, good for question answering.")

# Create a custom wrapper for the RAG tool that passes the model name
def rag_tool_with_model(input_str):
    return rag_tool(input_str, selected_model)

# Update the RAG tool with the model-enabled version
tools[2] = Tool(
    name="RAG",
    func=rag_tool_with_model,
    description="Use this tool to answer questions based on the TechCorp document collection."
)

# Create a simple custom agent executor
class SimpleAgentExecutor:
    def __init__(self, tools):
        self.tools = {tool.name: tool for tool in tools}
    
    def invoke(self, inputs, return_intermediate_steps=False):
        query = inputs.get("input", "")
        intermediate_steps = []
        
        # Determine which tool to use based on the query
        if any(op in query.lower() for op in ["+", "-", "*", "/", "calculate"]):
            tool_name = "Calculator"
            # Extract the expression
            if "calculate" in query.lower():
                tool_input = query.lower().split("calculate")[1].strip()
            else:
                # Try to extract a mathematical expression
                import re
                match = re.search(r'(\d+\s*[\+\-\*/]\s*\d+)', query)
                tool_input = match.group(1) if match else query
                
            observation = self.tools[tool_name].func(tool_input)
            action = type('obj', (object,), {'tool': tool_name, 'tool_input': tool_input})
            intermediate_steps.append((action, observation))
            output = f"The result is {observation}"
            
        elif "define" in query.lower() or "meaning" in query.lower() or "what is" in query.lower():
            tool_name = "Dictionary"
            # Extract the word to define
            if "define" in query.lower():
                tool_input = query.lower().split("define")[1].strip()
            elif "meaning" in query.lower():
                tool_input = query.lower().split("meaning")[0].strip().split()[-1]
            else:
                tool_input = query.lower().split("what is")[1].strip().split()[0]
                
            observation = self.tools[tool_name].func(tool_input)
            action = type('obj', (object,), {'tool': tool_name, 'tool_input': tool_input})
            intermediate_steps.append((action, observation))
            output = f"The definition of {tool_input} is: {observation}"
            
        else:
            # Default to RAG tool
            tool_name = "RAG"
            tool_input = query
            observation = self.tools[tool_name].func(tool_input)
            action = type('obj', (object,), {'tool': tool_name, 'tool_input': tool_input})
            intermediate_steps.append((action, observation))
            
            # Extract just the answer part from the RAG tool response
            if "Answer:" in observation:
                answer_part = observation.split("Answer:")[1].split("Context:")[0].strip()
                output = answer_part
            else:
                output = observation
        
        result = {
            "output": output,
            "intermediate_steps": intermediate_steps
        }
        
        return result

# Initialize the agent executor
agent_executor = SimpleAgentExecutor(tools)

# Query input
query = st.text_input("Ask a question (e.g., 'What products does TechCorp offer?', 'Calculate 2 + 2', 'Define apple'):")

if query:
    # Run the agent and capture intermediate steps
    result = agent_executor.invoke({"input": query}, return_intermediate_steps=True)
    answer = result['output']
    log = []
    for step in result['intermediate_steps']:
        action = step[0]
        observation = step[1]
        log.append(f"Action: {action.tool}")
        log.append(f"Input: {action.tool_input}")
        log.append(f"Observation: {observation}")

    # Display results
    st.write("**Answer:**", answer)
    st.write("**Decision Log:**")
    for entry in log:
        st.write(entry)
