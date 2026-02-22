import os
from fastapi import FastAPI
from pydantic_settings import BaseSettings, SettingsConfigDict

DOTENV = os.path.join(os.path.dirname(__file__), ".env")


class Settings(BaseSettings):

    model_config = SettingsConfigDict(env_file=DOTENV)

    ADZUNA_API_KEY: str
    ADZUNA_APP_ID: str
    ADZUNA_BASE_URL: str
    MODEL_NAME: str
    MODEL_NAME_SMALL: str
    MODEL_NAME_QWEN_SMALL: str
    MODEL_RESUME_PARSING: str
    MODEL_ROLE_ANALYSIS: str
    MODEL_JOB_FORMAT: str

    MONGO_URI: str
    MONGO_DB_NAME: str
    CHROMA_DB_PATH: str
    CHROMA_DB_JOBS: str
    CHROMA_DB_USERS: str


settings = Settings()
