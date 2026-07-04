from langgraph.checkpoint.postgres import PostgresSaver
import os
import psycopg
from dotenv import load_dotenv
# Load the .env file
load_dotenv()

# postgres_conn = psycopg.connect(os.getenv('POSTGRES_URL'), autocommit=True, prepare_threshold=0)
# checkpointer = PostgresSaver(postgres_conn)
# checkpointer.setup()

with PostgresSaver.from_conn_string(os.getenv('POSTGRES_URL')) as checkpointer:
    config = {"configurable": {"thread_id": "1"}}
    checkpoints = list(checkpointer.list(config=config, limit=10))
    print(checkpoints)