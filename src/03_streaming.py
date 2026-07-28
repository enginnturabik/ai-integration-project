"""
Step 3: Streaming responses.

01_basic_call.py waits for Gemini to finish the entire reply before
printing anything. For longer responses that feels slow. Streaming
lets you print each chunk of text as it arrives, the way ChatGPT or
Gemini's own chat UI displays text - which is what most real
AI-integrated apps use for a responsive feel.
"""

import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

stream = client.models.generate_content_stream(
    model="gemini-flash-latest",
    contents="Write a short poem about learning to code.",
)

for chunk in stream:
    if chunk.text:
        print(chunk.text, end="", flush=True)

print()
