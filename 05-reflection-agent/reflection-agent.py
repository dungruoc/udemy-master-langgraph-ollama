from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from langchain_core.messages import SystemMessage, HumanMessage
from typing_extensions import TypedDict, Annotated

from langchain_ollama import ChatOllama

from langgraph.checkpoint.postgres import PostgresSaver
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv
import operator

from search_tools import web_search

load_dotenv()

ollama_base_url = 'http://localhost:11434'
# chat_model = 'ornith:35b' # 'gemma4:31b', 'qwen3.5:35b'
# chat_model = 'qwen3.5:35b-mlx'
# chat_model = 'ornith:9b'
chat_model = 'qwen3.5:9b-mlx'
max_agent_loop = 15

langfuse_callback_hdl = CallbackHandler()
config = {"callbacks": [langfuse_callback_hdl]}

llm = ChatOllama(
    base_url=ollama_base_url,
    model=chat_model,
    validate_model_on_init=True,
    temperature=0.8
)


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    research: str
    critique: str
    iteration: int


def researcher_node(agent_state: AgentState):
    llm_with_tools = llm.bind_tools([web_search])
    critique = agent_state.get('critique', None)

    feedback_prompt = ''
    if critique:
        feedback_prompt = f"""
        Provided critique: {critique}
        Try new research queries to address the missing points in the critique!
        """
    
    system_prompt = SystemMessage(f"""
    Your are a research agent with web search capabilities.
    {feedback_prompt}
    INSTRUCTIONS:
    - Use your search tools first to gather information!
    - Provide comprehensive research based on search results!
    - Always provide research output from search results only.
    """)

    messages = [system_prompt] + agent_state['messages']
    return {'messages': [llm_with_tools.invoke(messages, config=config)]}


def critique_node(agent_state: AgentState):
    user_question = agent_state["messages"][0].content
    research_content = agent_state["messages"][-1].content
    systemp_prompt  = SystemMessage(f"""
        You are a critique agent. Evaluate if the research is good enough.
        
        Check:
        - If it answers the main question?
        - Is there reasonable details?
                                    
        Response Format:
        DECISION: APPROVE or REVISE
        
        Be lenient! APPROVE if the research is decent enough
        Only REVISE if critical information is completely missing!
    """)

    critique_prompt = HumanMessage(f"""
        Main question:
        {user_question}

        Evaluate this research:
        {research_content}
        """)
    messages = [systemp_prompt, critique_prompt]
    response = llm.invoke(messages)
    iteration = agent_state.get("iteration", 0) + 1

    return {
        'research': research_content,
        'critique': response.content,
        'iteration': iteration
    }

def tool_call_route(agent_state: AgentState):
    last_msg = agent_state['messages'][-1]

    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
        return "tool_calls"
    
    return "critique"

def critique_route(agent_state: AgentState):
    critique = agent_state.get("critique", '')
    iterations = agent_state.get("iteration", 0)

    if iterations >= max_agent_loop:
        print("Max iterations reached")
        return END
    
    if 'approve' in critique.lower():
        print(f"Critique approved after {iterations}")
        return END
    
    print(f"[SYSTEM] Continue revision, iteration {iterations}")
    return "researcher"



def create_reflection_agent(checkpointer=None):
    builder = StateGraph(AgentState)

    builder.add_node("researcher", researcher_node)
    builder.add_node("critique", critique_node)
    builder.add_node("tool_calls", ToolNode([web_search]))

    builder.add_edge(START, "researcher")
    builder.add_conditional_edges("researcher", tool_call_route, ["critique", "tool_calls"])
    builder.add_edge("tool_calls", "researcher")
    builder.add_conditional_edges("critique", critique_route, ["researcher", END])

    return builder.compile(checkpointer=checkpointer)


agent = create_reflection_agent()

# question = "Hi! which team is the most likely to win the Fifa worldcup 2026?"
question = "What are the latest developments in LangGraph for building AI agents?"

res = agent.invoke({'messages': [HumanMessage(question)]}, config=config)

print(res)
print(res['research'])
print(res['critique'])