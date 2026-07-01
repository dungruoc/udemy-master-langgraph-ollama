from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv

# Load the .env file
load_dotenv()


ollama_base_url = 'http://localhost:11434'
model = 'gemma4:31b'

llm = ChatOllama(
    base_url=ollama_base_url,
    model=model,
    validate_model_on_init=True,
    temperature=0.8
)

langfuse_callback_hdl = CallbackHandler()

