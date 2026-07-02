from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
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
# model = 'gemma4:31b' # 'ornith:35b'
model = 'ornith:35b'
max_react_loop = 5

all_tools = [agent_tools.get_weather, agent_tools.calculate]


llm = ChatOllama(
    base_url=ollama_base_url,
    model=model,
    validate_model_on_init=True,
    temperature=0.8
)

langfuse_callback_hdl = CallbackHandler()

config = {"callbacks": [langfuse_callback_hdl]}


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


def agent_node(in_state: AgentState):
    llm_with_tools = llm.bind_tools(all_tools)
    system_prompt = "You are a helpful assistant. Using the conversation history and the available tools when needed to help the users."
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

    return builder.compile()


print(agent_node({'messages': [HumanMessage("Hello, what is your name?")]}))
print(agent_node({'messages': [HumanMessage("Hello! Is it nice today in Hanoi")]}))

agent = create_react_agent()
res = agent.invoke({'messages': [HumanMessage("Hello! Is it nice today in Hanoi")]}, config=config)
print(res)
res['messages'][-1].pretty_print()

res = agent.invoke({'messages': [
    HumanMessage("Hello! Is it nice today in Hanoi"),
    ToolMessage('[{"FeelsLikeC": "30", "FeelsLikeF": "86", "cloudcover": "50", "humidity": "94", "observation_time": "03:03 PM", "precipInches": "0.0", "precipMM": "0.0", "pressure": "1008", "pressureInches": "30", "temp_C": "27", "temp_F": "81", "uvIndex": "0", "visibility": "8", "visibilityMiles": "4", "weatherCode": "116", "weatherDesc": [{"value": "Partly cloudy"}], "weatherIconUrl": [{"value": "https://cdn.worldweatheronline.com/images/wsymbols01_png_64/wsymbol_0004_black_low_cloud.png"}], "winddir16Point": "S", "winddirDegree": "184", "windspeedKmph": "9", "windspeedMiles": "5"}]',
                name='get_weather', tool_call_id='de3b82ff-9261-482d-8f87-737a958a3ef8')
]}, config=config)
print(res)
res['messages'][-1].pretty_print()