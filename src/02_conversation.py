"""
Step 2: Multi-turn conversation.

The API itself has no memory. Every request is stateless - to have a
"conversation," your program must resend the entire message history
each time, with roles alternating between "user" and "model" (Gemini's
name for what Anthropic/OpenAI call "assistant").

This script is a simple command-line chat loop that builds up that
history turn by turn.

(Gemini's SDK also has a client.chats.create() helper that manages this
history for you automatically - once this pattern clicks, try swapping
to that and see how much code it removes.)
"""

import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# The full conversation so far, sent in full on every request.
contents = []

print("Chat with Gemini. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")
    if user_input.strip().lower() == "quit":
        break

    contents.append({"role": "user", "parts": [{"text": user_input}]})

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=contents,
    )

    reply_text = response.text
    print(f"Gemini: {reply_text}\n")

    # The model's reply must also go into history, or the next
    # request won't know what was already said.
    contents.append({"role": "model", "parts": [{"text": reply_text}]})
