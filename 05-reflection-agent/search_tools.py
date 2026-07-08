from langchain_core.tools import tool
from ddgs import DDGS

from dotenv import load_dotenv

load_dotenv()

@tool
def web_search(query: str, max_results:int=5) -> str:
    """Search the latest inform mation from Internet using DuckDuckGoSearch
    Use this tool whenever you need to access the latest information from Internet.

    Args:
        query: Search query string
        max_results: maximum number of items to return from search results (default: 5)

    Returns:
        Formatted search results with titles, descriptions, URLs

    """
    founds = DDGS().text(query=query, max_results=max_results)
    def format_item(it):
        return(f"- title: {it['title']}\n  description: {it['body']}\n  url: {it['href']}")
    if founds:
        return "\n\n".join(format_item(it) for it in founds)
    
    return "Nothing found"


if __name__ == "__main__":
    print(web_search.invoke({"query": "What are the soccer world-cup matchs today?"}))