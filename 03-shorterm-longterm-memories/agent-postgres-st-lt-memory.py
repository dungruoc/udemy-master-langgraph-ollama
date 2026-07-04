from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver

# long term store
from langgraph.store.postgres import PostgresStore
from langchain_ollama import OllamaEmbeddings

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
import psycopg
import os

from agent_tools import get_weather, calculate
from lt_mem_tools import save_user_preferences, search_user_preferences, search_user_prerefences_from_text

# Load the .env file
load_dotenv()


ollama_base_url = 'http://localhost:11434'
# model = 'gemma4:31b'
model = 'ornith:35b'
# model = 'qwen3.5:35b'
max_react_loop = 15
emb_model = 'nomic-embed-text:v1.5'


all_tools = [get_weather, calculate, search_user_preferences, save_user_preferences]


llm = ChatOllama(
    base_url=ollama_base_url,
    model=model,
    validate_model_on_init=True,
    temperature=0.8
)

langfuse_callback_hdl = CallbackHandler()


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    user_id: str


def agent_node(in_state: AgentState):
    llm_with_tools = llm.bind_tools(all_tools)
    user_id = in_state["user_id"]
    last_message = in_state['messages'][-1].content
    user_preferences = search_user_prerefences_from_text(user_id=user_id, query=last_message)
    system_prompt = f"""You are a helpful fact-based assistant, with long-term memory and utilities tools

    User ID: {user_id}
    Current User Preferences:
    {user_preferences}

    MEMORY TOOLS USAGE:

    1. save_user_preferences: Use when user shares NEW prererences information
    - Always pass user_id: "{user_id}"
    - Food preferences (diet, likes, dislikes, allergies)
    - Work information (role, company, interests)
    - Hobbies and activities
    - Schedule and availability
    - Location and timezone

    2. search_user_preferences: Use when you need to recall specific category about user's preferences
    - Always pass user_id: "{user_id}"
    - When answering questions about past preferences
    - When user asks "what do you know about me?"
    - When making recommendations based on preferences

    UTILITY TOOLS USAGE:

    3. get_weather: Use to retrieve current weather information
    - Pass location as parameter (city name, zip code, or coordinates)
    - Use when user asks about weather conditions
    - Use when planning activities that depend on weather
    - Examples: "What's the weather in London?", "Will it rain today?"

    4. calculate: Use to perform mathematical calculations
    - Pass mathematical expression as string parameter
    - Supports basic arithmetic (+, -, *, /)
    - Supports advanced operations (powers, roots, trigonometry)
    - Use when user needs numerical computations
    - Examples: "What's 15% of 250?", "Calculate the area of a circle with radius 5"

    GUIDELINES:
    - Always save when user shares personal information
    - Retrieve specific categories when needed for context
    - Use semantic search results shown above for general context
    - Use get_weather when location-based weather info is needed
    - Use calculate for any mathematical operations or conversions
    - Be conversational and natural when using all tools
    - Combine tools when appropriate (e.g., weather + saved location preference)
    - If the information exists in the conversation history, rather use it than call tools again!
    - Use ONLY the data provided from tools or by user!
    - Do not explain where you got the data or why you did not get data!
    - Do not invent any fact from youself, just say sorry that you do not know if you have data!
    - Make the answer direct and consise."""
    messages = [SystemMessage(system_prompt)] + in_state["messages"]
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


def create_react_agent(checkpointer=None):
    builder = StateGraph(AgentState)

    builder.add_node("agent_node", agent_node)
    builder.add_node("end_loop", end_loop)
    builder.add_node("tool_calls", ToolNode(all_tools))

    builder.add_edge(START, "agent_node")
    builder.add_conditional_edges("agent_node", tool_call_route, [END, "end_loop", "tool_calls"])
    builder.add_edge("tool_calls", "agent_node")
    builder.add_edge("end_loop", END)

    return builder.compile(checkpointer=checkpointer)

user_id = 'toto-learing-ai'
config = {"callbacks": [langfuse_callback_hdl], "configurable": {"thread_id": f"{user_id}_longterm"}}


# with PostgresSaver.from_conn_string(os.getenv('POSTGRES_URL')) as checkpointer:
#     checkpointer.setup()
#     agent = create_react_agent(checkpointer)

#     prompt = "Hi dude! I am Toto, I like learning AI and programming. My favorite programming langages ar Python, C++, rust"
#     res = agent.invoke({'messages': [HumanMessage(prompt)], 'user_id': user_id}, config=config)
#     res['messages'][-1].pretty_print()


agent = create_react_agent(checkpointer=None)

prompt = "Hi dude! what could you recommend me?"
res = agent.invoke({'messages': [HumanMessage(prompt)], 'user_id': user_id}, config=config)
res['messages'][-1].pretty_print()
