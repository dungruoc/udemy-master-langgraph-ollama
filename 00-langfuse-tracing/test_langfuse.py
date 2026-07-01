from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langfuse.langchain import CallbackHandler
from langfuse import Langfuse
from langchain_ollama import ChatOllama

from dotenv import load_dotenv

# Load the .env file
load_dotenv()

external_request_id = "req_12345"
predefined_trace_id = Langfuse.create_trace_id(seed=external_request_id)

# 1. Initialize Langfuse Callback Handler
langfuse_handler = CallbackHandler(trace_context={"trace_id": predefined_trace_id})

# 2. Configure ChatOllama (Ensure your local Ollama instance is running)

ollama_base_url = 'http://localhost:11434'
model = 'gemma4:31b'

llm = ChatOllama(
    base_url=ollama_base_url,
    model=model,
    validate_model_on_init=True,
    temperature=0.8
)

# 3. Create a simple chain
prompt = ChatPromptTemplate.from_template("Explain the concept of {topic} in one short sentence.")
chain = prompt | llm | StrOutputParser()

# 4. Invoke the chain with the tracking callback
response = chain.invoke(
    {"topic": "Quantum Computing"},
    config={"callbacks": [langfuse_handler]}
)

print("LLM Response:", response)
