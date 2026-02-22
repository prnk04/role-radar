from pathlib import Path
import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DB_PATH = BASE_DIR / "data" / "chroma_v1"
CHROMA_DB_JOBS = os.getenv('CHROMA_DB_JOBS', 'jobs')
CHROMA_DB_USERS = os.getenv('CHROMA_DB_USERS', 'users')

print("chroma db: ", CHROMA_DB_PATH)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)


def get_vector_store_jobs():
    return Chroma(CHROMA_DB_JOBS, persist_directory=str(CHROMA_DB_PATH),
                  embedding_function=embeddings)


def get_vector_store_users():
    return Chroma(CHROMA_DB_USERS, persist_directory=str(CHROMA_DB_PATH),
                  embedding_function=embeddings)
