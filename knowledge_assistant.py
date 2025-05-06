import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import openai
import requests
import streamlit as st
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentType
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI

# Set up OpenAI API key 
openai.api_key = ""

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

# Generate answer using OpenAI's LLM
def generate_answer(query, context):
    prompt = f"Based on the following context, answer the question: {query}\n\nContext:\n{context}"
    client = openai.OpenAI(api_key=openai.api_key)
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",  
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

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

def rag_tool(input_str):
    context = retrieve(input_str)
    answer = generate_answer(input_str, "\n".join(context))
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

# Initialize LLM and agent
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=openai.api_key)

# Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can use tools to answer questions."),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Streamlit web UI
st.title("TechCorp Knowledge Assistant")

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