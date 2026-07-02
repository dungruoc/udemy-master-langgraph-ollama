from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import (
    StrOutputParser
)

from typing_extensions import TypedDict, Annotated
from pydantic import BaseModel, Field
import operator

from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv

import agent_tools

# Load the .env file
load_dotenv()


ollama_base_url = 'http://localhost:11434'
# model = 'gemma4:31b'
model = 'ornith:35b'
# model = 'qwen3.5:35b'
max_react_loop = 15

all_tools = [agent_tools.get_weather, agent_tools.calculate]


llm = ChatOllama(
    base_url=ollama_base_url,
    model=model,
    validate_model_on_init=True,
    temperature=0.8
)

langfuse_callback_hdl = CallbackHandler()

config = {"callbacks": [langfuse_callback_hdl], "configurable": {"thread_id": "1"}}


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


def agent_node(in_state: AgentState):
    llm_with_tools = llm.bind_tools(all_tools)
    system_prompt = """You are a helpful fact-based assistant.
    Use the available tools when needed to help the users!
    If the information exists in the conversation history, better to use it!
    Use ONLY the data provided from tools or by user!
    Do not explain where you got the data or why you did not get data!
    Do not invent any fact from youself, just say sorry that you do not know if you have data!
    Make the answer direct and consise."""
    messages = in_state["messages"] + [SystemMessage(system_prompt)]
    response = llm_with_tools.invoke(messages)
    return {'messages': [response]}
 
def end_loop(in_state: AgentState):
    in_state['messages'].append(HumanMessage("You tried too much to solve, but we have no more time. Say sorry by answering with information you got so far."))
    response = llm.invoke(in_state['messages'])
    return {'messages': [response]}
 
def tool_call_route(in_sate: AgentState):
    last_msg = in_sate['messages'][-1]

    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
        if len(in_sate["messages"]) > max_react_loop:
            return "end_loop"
        return "tool_calls"
    
    return END


def create_react_agent():
    builder = StateGraph(AgentState)

    builder.add_node("agent_node", agent_node)
    builder.add_node("end_loop", end_loop)
    builder.add_node("tool_calls", ToolNode(all_tools))

    builder.add_edge(START, "agent_node")
    builder.add_conditional_edges("agent_node", tool_call_route, [END, "end_loop", "tool_calls"])
    builder.add_edge("tool_calls", "agent_node")
    builder.add_edge("end_loop", END)

    return builder.compile(checkpointer=MemorySaver())

agent = create_react_agent()
res = agent.invoke({'messages': [HumanMessage("Hello! I am Toto, living in San Francisco. Is it a nice day to go out?")]}, config=config)
print(res)
res['messages'][-1].pretty_print()

res = agent.invoke({'messages': [HumanMessage("Hi dude! What is the weather like today in here?")]}, config=config)
print(res)
res['messages'][-1].pretty_print()

res = agent.invoke({'messages': [HumanMessage("Hi again, do you recognize me? What is the weather like in Hanoi?")]}, config=config)
print(res)
res['messages'][-1].pretty_print()

res = agent.invoke({'messages': [HumanMessage("Hi, me again? Say my name and tell me about Messi?")]}, config=config)
print(res)
res['messages'][-1].pretty_print()
