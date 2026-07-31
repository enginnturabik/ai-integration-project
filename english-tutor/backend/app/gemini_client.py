"""Thin wrapper around the Gemini API: text generation and embeddings.

Reuses the same load_dotenv() + genai.Client(api_key=...) pattern as
src/02_conversation.py at the repo root - python-dotenv's load_dotenv()
walks up parent directories to find the root .env, so no separate key
needs to be configured for the backend.
"""

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

GENERATION_MODEL = "gemini-flash-latest"
EMBEDDING_MODEL = "gemini-embedding-001"


def embed(texts: list[str], task_type: str) -> list[list[float]]:
    """Embed a batch of texts. task_type is one of Gemini's embedding task
    types, e.g. "RETRIEVAL_DOCUMENT" for KB/mistake entries being stored,
    or "RETRIEVAL_QUERY" for the text being searched with.
    """
    response = _client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return [e.values for e in response.embeddings]


def generate_structured(prompt: str, response_schema: type) -> object:
    """Call Gemini asking for JSON output matching response_schema (a Pydantic
    model class), and return an instance of that model."""
    response = _client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
        ),
    )
    return response.parsed
