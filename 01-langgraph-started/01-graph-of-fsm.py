from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph

from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv

# Load the .env file
load_dotenv()


class SimpleState(TypedDict):
    input_text: str
    output_text: str


def process_input(in_state: SimpleState) -> SimpleState:
    return {'output_text': in_state['input_text'].upper()}


def add_prefix(in_state: SimpleState) -> SimpleState:
    return {'output_text': "Hello, added this prefix - " + in_state['output_text']}

def add_suffix(in_state: SimpleState) -> SimpleState:
    return {'output_text': in_state['output_text'] + " - and suffix added here"}

def create_simple_graph():
    builder = StateGraph(SimpleState)

    builder.add_node("process_input", process_input)
    builder.add_node("add_prefix", add_prefix)
    builder.add_node("add_suffix", add_suffix)

    builder.add_edge(START, "process_input")
    builder.add_edge("process_input", "add_prefix")
    builder.add_edge("add_prefix", "add_suffix")
    builder.add_edge("add_suffix", END)

    return builder.compile()



# 1. Initialize Langfuse Callback Handler
langfuse_handler = CallbackHandler()


print(create_simple_graph().invoke(
    {'input_text': 'my first graph'},
    config={"callbacks": [langfuse_handler]}
))

