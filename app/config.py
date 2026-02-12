"""
Configuration – loads settings from .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class Settings:
    # GCP
    GCP_PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "general-ak")
    GCP_LOCATION: str = os.getenv("GCP_LOCATION", "us-central1")

    # RAG
    RAG_CORPUS_RESOURCE_NAME: str = os.getenv(
        "RAG_CORPUS_RESOURCE_NAME",
        "projects/general-ak/locations/us-central1/ragCorpora/2305843009213693952",
    )

    # Speech-to-Text (Chirp 3 is in the "us" multi-region)
    STT_MODEL: str = os.getenv("STT_MODEL", "chirp_3")
    STT_LOCATION: str = os.getenv("STT_LOCATION", "us")
    STT_LANGUAGE: str = os.getenv("STT_LANGUAGE", "auto")

    # LLM (gemini-2.5-flash works in us-central1 alongside RAG)
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.5-flash")

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))


settings = Settings()
