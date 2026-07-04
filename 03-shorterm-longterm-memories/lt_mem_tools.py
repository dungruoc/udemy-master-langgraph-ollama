# long term store
from langgraph.store.postgres import PostgresStore
from langchain_ollama import OllamaEmbeddings

from langchain_core.tools import tool

from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

ollama_base_url = 'http://localhost:11434'
emb_model = 'nomic-embed-text:v1.5'

embeddings = OllamaEmbeddings(model=emb_model, base_url=ollama_base_url)

def embed_texts(texts: list[str]) -> list[list[float]]:
    return embeddings.embed_documents(texts=texts)

emb_index = {"embed": embed_texts, "dims": 768}

@tool
def search_user_preferences(user_id: str, category: str) -> str:
    """Retrieve user preferences for personalization.

    Args:
        user_id: User identifier
        category: Category of information to retrieve (e.g., 'food', 'work', 'hobbies')
    """
    with PostgresStore.from_conn_string(os.getenv("POSTGRES_URL")) as store:    
        namespace = (user_id, "preferences")

        item = store.get(namespace, category)

        if item:
            return f"{category}: {item.value}"

    return f"{category}: No information found!"

@tool
def save_user_preferences(user_id: str, category: str, information: dict) -> str:
    """Save user preferences to long-term memory.

    Args:
        user_id: User identifier
        category: Category of information (e.g., 'food', 'work', 'hobbies', 'schedule', 'location')
        information: Dictionary containing the information to save
    """
    with PostgresStore.from_conn_string(os.getenv("POSTGRES_URL")) as store:    
        namespace = (user_id, "preferences")

        store.put(namespace, category, information)
        return "Information Saved"

    return "Nothing Saved"

def setup_store():
    with PostgresStore.from_conn_string(os.getenv("POSTGRES_URL")) as store:
        store.setup()

def search_user_prerefences_from_text(user_id: str, query: str, limit=3):
    with PostgresStore.from_conn_string(os.getenv("POSTGRES_URL"), index=emb_index) as store:
        namespace = (user_id, "preferences")
        items = store.search(namespace, query=query, limit=limit)
        if items:
            return '\n\n'.join([f" - {it.key}: {it.value}" for it in items])
    return "None"


if __name__ == '__main__':
    def test_store():
        with PostgresStore.from_conn_string(os.getenv("POSTGRES_URL"), index=emb_index) as store:
            store.setup()
            user_id = "test_user_id"
            namespace = (user_id, "preferences")
            food_ref = {'diet': 'veg', 'likes': ['pasta', 'pizza', 'veggies']}
            color_ref = {'favorites': ['blue', 'green'], 'dislikes': ['black', 'orange', 'brown']}
            store.put(namespace, 'food', food_ref)
            store.put(namespace, 'color', color_ref)


    def test_read():
        with PostgresStore.from_conn_string(os.getenv("POSTGRES_URL"), index=emb_index) as store:
            store.setup()
            user_id = "test_user_id"
            namespace = (user_id, "preferences")
            print(store.get(namespace, 'food'))
            print(store.get(namespace, 'color'))

    def test_search():
        user_id = "test_user_id"
        query = "what does the user like to eat?"
        print(query, " -> ", search_user_prerefences_from_text(user_id, query=query, limit=1))
        query = "what does the user dislike?"
        print(query, " -> ", search_user_prerefences_from_text(user_id, query=query, limit=1))

    def test_delete():
        with PostgresStore.from_conn_string(os.getenv("POSTGRES_URL"), index=emb_index) as store:
            store.setup()
            user_id = "test_user_id"
            namespace = (user_id, "preferences")
            store.delete(namespace, 'food')
            store.delete(namespace, 'color')

    def test_tools():
        print(save_user_preferences.invoke({'user_id': 'test_user_id', 'category': 'work', 'information': {'techs': ['ai', 'ITxOT'], 'domain': 'consulting'}}))
        print(search_user_preferences.invoke({'user_id': 'test_user_id', 'category': 'work'}))

    test_store()
    test_read()
    test_search()
    test_delete()
    test_read()
    test_tools()
