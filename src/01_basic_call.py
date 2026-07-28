"""
Step 1: The simplest possible AI integration.

This script sends one prompt to Gemini and prints the reply.
Every AI integration, no matter how complex, boils down to this same
pattern: build a request, send it to the API, read the response.
"""

import os

from dotenv import load_dotenv
from google import genai

# Reads the GEMINI_API_KEY value out of the .env file into the
# environment, so we never hardcode secrets in source code.
load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

response = client.models.generate_content(
    model="gemini-flash-latest",
    contents="In one sentence, what is an API?",
)

print(response.text)
