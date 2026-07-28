# AI Integration Project

Learning project for integrating an LLM (Google Gemini) into code via API calls, using Gemini's free tier - no credit card required.

## 1. Get an API key

1. Go to https://aistudio.google.com/apikey and sign in with a Google account.
2. Click **Create API key**. No billing/card setup is required for the free tier.
3. Copy the key.

The free tier has rate limits (requests per minute/day) that vary by model, but they're generous enough for learning and experimentation. See https://ai.google.dev/gemini-api/docs/rate-limits for current numbers.

## 2. Configure the project

1. Copy `.env.example` to a new file named `.env`.
2. Paste your key in: `GEMINI_API_KEY=...`
3. `.env` is already listed in `.gitignore` so it won't get committed if you use git.

## 3. Set up the environment

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

(The venv and dependencies are already installed if you're continuing this session - just activate it.)

## 4. Run the scripts, in order

| Script | Concept |
|---|---|
| `src/01_basic_call.py` | One request, one response - the core pattern. |
| `src/02_conversation.py` | Multi-turn chat - why you resend full history each turn. |
| `src/03_streaming.py` | Streaming tokens as they generate, instead of waiting. |

```bash
python src/01_basic_call.py
python src/02_conversation.py
python src/03_streaming.py
```

## Cost note

The free tier doesn't bill you - it just rate-limits you (fewer requests per minute/day than the paid tier). If you hit a rate-limit error, wait a bit and retry.

## Where to go next

Once these three click, natural next steps: system instructions (steering tone/behavior), function calling / tool use (letting the model call functions in your code), and structured output (getting JSON back instead of prose). The `google-genai` SDK supports all of these the same way the Anthropic/OpenAI SDKs do, just with slightly different parameter names.
