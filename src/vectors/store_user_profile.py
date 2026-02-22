import datetime

from langchain_core.documents import Document

import hashlib

import os
from dotenv import load_dotenv

import logging

from chroma_store import get_vector_store_users
from src.utils.commons import get_hashed

load_dotenv()
# model = SentenceTransformer(model_name_or_path="all-MiniLM-L6-v2")

CHROMA_DB_PATH = os.getenv('CHROMA_DB_PATH')
CHROMA_DB_USERS = os.getenv('CHROMA_DB_USERS', 'users')


def store_user_profile_vector(user_profile: str, prev_roles: list[str], skills: list[str], email_id: str, db_id):
    try:

        # vector_store = Chroma(CHROMA_DB_USERS, persist_directory=CHROMA_DB_PATH,
        #                       embedding_function=embeddings)

        vector_store = get_vector_store_users()

        user_id = get_hashed(email_id)

        # first, check if the data exists: if yes: delete it, and then proceed
        existing_data = vector_store.get(where={"user_id": user_id})
        if len(existing_data.get("ids", [])) > 0:
            vector_store.delete(ids=existing_data.get("ids"))

        prev_roles_embedded = [Document(page_content=x, metadata={
            'user_id': user_id,
            'doc_id': str(db_id),
            'type': 'prev_roles',
            'created_at': datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S %z")
        })

            for x in prev_roles]

        user_profile_doc = [Document(page_content=user_profile, metadata={
            'user_id': user_id,
            'doc_id': str(db_id),
            'type': 'user_profile',
            'created_at': datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S %z")
        }),
            Document(page_content='|'.join(skills), metadata={
                'user_id': user_id,
                'doc_id': str(db_id),
                'type': 'skills',
                'created_at': datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S %z")
            }),

        ]
        user_profile_doc.extend(prev_roles_embedded)

        vector_store.add_documents(user_profile_doc)

        logging.info(f"Stored user profile in chroma db")
        return 1

    except Exception as e:
        logging.error(
            f"Error in creating embeddings of user profile and storing them in db: {e}")
        return 0
