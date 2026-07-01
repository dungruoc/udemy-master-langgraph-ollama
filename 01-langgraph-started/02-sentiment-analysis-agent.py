from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import (
    StrOutputParser
)

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

config = {"callbacks": [langfuse_callback_hdl]}

class SentimentAnalysis(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="The classified sentiment, either positive or negative")
    confidence: float = Field(ge=0, le=1.0, description="Confidence score of the classification, from 0.0 to 1.0")
    reason: str = Field(description="Brief explanation")

class SentimentState(TypedDict):
    original_tweet: str
    sentiment: str
    confidence: float

    response_tweet: str

def analyze_sentiment(in_state: SentimentState) -> SentimentState:
    structured_llm = llm.with_structured_output(SentimentAnalysis)

    messages = [
        SystemMessage("Analyze the sentiment and provide the structured output. Use 0.0 to 1.0 scale for confidence. Higher score is for highly confident about the sentiment classification (highly positive or highly negative). Score zero should be for neutral."),
        HumanMessage(in_state['original_tweet'])
    ]

    analysis = structured_llm.invoke(messages)

    return {
        'sentiment': analysis.sentiment,
        'confidence': analysis.confidence
    }

def generate_positive_response(in_state: SentimentState) -> SentimentState:
    messages = [
        SystemMessage(f"""Generate a warm response to this positive tweet under 250 chars.
        Confidence: {in_state['confidence']}.
        Confidence is scored from 0 to 1.0. High confidence means be enthusiastic, otherwise be fair enough.
        """),
        HumanMessage(in_state['original_tweet'])
    ]

    chain = llm | StrOutputParser()

    return {'response_tweet': chain.invoke(messages)}

def generate_negative_response(in_state: SentimentState) -> SentimentState:
    messages = [
        SystemMessage(f"""Generate an empathetic response to this positive tweet under 250 chars.
        Confidence: {in_state['confidence']}.
        Confidence is scored from 0 to 1.0. High confidence means it was very bad and would need support, otherwise be understanding.
        """),
        HumanMessage(in_state['original_tweet'])
    ]

    chain = llm | StrOutputParser()

    return {'response_tweet': chain.invoke(messages)}

def route_by_sentiment(in_state: SentimentState):
    if in_state['sentiment'] == 'positive':
        return 'generate_positive_response'

    return 'generate_negative_response'

def create_agent_graph():
    builder = StateGraph(SentimentState)

    builder.add_node("analyze_sentiment", analyze_sentiment)
    builder.add_node("generate_positive_response", generate_positive_response)
    builder.add_node("generate_negative_response", generate_negative_response)

    builder.add_edge(START, "analyze_sentiment")

    builder.add_conditional_edges(
        "analyze_sentiment",
        route_by_sentiment,
        ["generate_positive_response", "generate_negative_response"]
    )
    
    builder.add_edge("generate_positive_response", END)
    builder.add_edge("generate_negative_response", END)

    return builder.compile()

graph = create_agent_graph()
# print(graph.invoke(input={'original_tweet': "I have just had dinner."}, config=config))
print(graph.invoke(input={'original_tweet': "The weather is nice today. I do not know what to do."}, config=config))
print(graph.invoke(input={'original_tweet': "The world is ridiculous. I am fed up."}, config=config))
