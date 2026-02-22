import datetime

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

import hashlib

import os
from dotenv import load_dotenv

from chroma_store import get_vector_store_jobs

import logging

load_dotenv()
# model = SentenceTransformer(model_name_or_path="all-MiniLM-L6-v2")

CHROMA_DB_PATH = os.getenv('CHROMA_DB_PATH')
CHROMA_DB_JOBS = os.getenv('CHROMA_DB_JOBS', 'jobs')
print("CWD:", os.getcwd())
print("Persist dir:", CHROMA_DB_PATH)


def store_job_profile_vector(job_profile: str, skills_reqd: list[str], skills_optional: list[str], job_role: str, job_id: str, what: str, where: str, db_id: str):
    try:
        vector_store = get_vector_store_jobs()
        all_skills = list()
        all_skills.extend(skills_reqd)
        all_skills.extend(skills_optional)

        # first, check if the data exists: if yes: delete it, and then proceed
        existing_data = vector_store.get(where={"job_id": job_id})
        if len(existing_data.get("ids", [])) > 0:
            vector_store.delete(ids=existing_data.get("ids"))

        job_profile_doc = [Document(page_content=job_profile, metadata={
            'job_id': job_id,
            'doc_id': str(db_id),
            'what': what,
            'where': where,
            'type': 'job_profile',
            'created_at': datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S %z")
        }),
            Document(page_content='|'.join(all_skills), metadata={
                'job_id': job_id,
                'doc_id': str(db_id),
                'what': what,
                'where': where,
                'type': 'job_skills',
                'created_at': datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S %z")
            }),
            Document(page_content=job_role, metadata={
                'job_id': job_id,
                'doc_id': str(db_id),
                'what': what,
                'where': where,
                'type': 'job_role',
                'created_at': datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S %z")
            }),

        ]

        vector_store.add_documents(job_profile_doc)

        logging.info(f"Stored job profile in chroma db")
        return 1

    except Exception as e:
        logging.error(
            f"Error in creating embeddings of job profile and storing them in db: {e}")
        return 0
