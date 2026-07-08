from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage

from typing_extensions import TypedDict, Annotated

from langchain_ollama import ChatOllama

from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio

from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv
import operator
import sys


load_dotenv()

ollama_base_url = 'http://localhost:11434'
chat_model = 'qwen3.5:9b-mlx'

llm = ChatOllama(
    base_url=ollama_base_url,
    model=chat_model,
    validate_model_on_init=True,
    temperature=0.8
)


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


async def get_tools():

    client = MultiServerMCPClient({
        "openbnb-airbnb": {
            "command": "docker",
            "args": [
                "run",
                "-i",
                "--rm",
                "mcp/openbnb-airbnb"
            ],
            "transport": "stdio"
        }
    })


    tools = await client.get_tools()

    print(f"Loaded {len(tools)} Tools")
    print(f"Tools Available: {tools}")

    return tools


async def agent_node(agent_state: AgentState):
    tools = await get_tools()
    llm_with_tools = llm.bind_tools(tools)

    response = llm_with_tools.invoke(agent_state["messages"])
    return {'messages': [response]}


async def create_agent(checkpointer=None):
    builder = StateGraph(AgentState)

    builder.add_node("agent", agent_node)
    tools = await get_tools()
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "agent")
    builder.add_edge("tools", "agent")
    builder.add_conditional_edges("agent", tools_condition)

    return builder.compile(checkpointer=checkpointer)



async def search(query: str):
    langfuse_callback_hdl = CallbackHandler()
    config = {"callbacks": [langfuse_callback_hdl]}
    agent = await create_agent()
    response = await agent.ainvoke({"messages": [HumanMessage(query)]}, config=config)
    res = response['messages'][-1].content
    print(res)
    return res

if __name__ == "__main__":
    query = " ".join(sys.argv[1:])
    print("[AGENT] query: ", query)
    asyncio.run(search(query))