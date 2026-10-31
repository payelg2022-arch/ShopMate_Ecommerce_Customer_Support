
# ShopMate — E-Commerce Customer Support Assistant

An evaluation-ready MVP demonstrating a customer support assistant with RAG, mock commerce tools, optional LLM generation, a LangGraph workflow (when installed), and a responsive browser UI.

## Features
- Order-status lookup using synthetic order records.
- Return-request creation using a mock API function (writes only to local JSON).
- Refund-policy explanation and safe handling of disputed/unknown payment states.
- RAG over FAQ and policy documents: SentenceTransformer embeddings (`all-MiniLM-L6-v2`) with cosine similarity; falls back to TF-IDF retrieval if the embedding model is unavailable.
- Optional Groq LLM via `langchain-groq`. Without a key, a grounded extractive fallback keeps the demo usable.
- Intent routing, missing-information clarification, source labels, and simulated human escalation.
- FastAPI backend and a custom HTML/CSS/JavaScript chat interface.
- Evaluation test cases, sample logs, technical report, and presentation are included alongside the project ZIP.

## Quick start (Python 3.11 recommended)
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env  # Windows; on macOS/Linux use: cp .env.example .env
# Optional: add GROQ_API_KEY to .env (GROQ_MODEL defaults to llama-3.3-70b-versatile)
uvicorn app.main:app --reload
```
Open `http://127.0.0.1:8000`.

The first use of SentenceTransformer may download a model. If that is unavailable, TF-IDF fallback is used. The optional Groq LLM is enabled only if `GROQ_API_KEY` is configured and the package can initialize.

## Demo prompts
- `Where is my order ORD-1001?`
- `I want to return ORD-1002 because the shoes are too small`
- `How long does a refund take?`
- `I see a duplicate charge on my card`
- `How can I speak to a human agent?`
- `My order is missing and I don't have an order ID`

## Architecture
Browser UI → FastAPI `/api/chat` → intent routing → LangGraph workflow if available → mock tools for transactional intents or local RAG for policy/FAQ intents → optional LLM grounded in retrieved passages → response + source labels.

## Data and safety
All order data is synthetic. Return requests mutate only local `data/orders.json` and `data/returns.json`. Escalation is simulated and does not contact a real human. Never enter real payment credentials. The assistant does not cancel orders, process payments, or confirm live payment-provider events.

## API
- `GET /api/health`: runtime capabilities and retrieval mode.
- `POST /api/chat`: `{"message":"Where is my order ORD-1001?"}`.
- `GET /api/demo-orders`: sample order IDs.

## Evaluation
See `evaluation/test_cases.csv` and `evaluation/sample_conversation.json`. Record whether intent, tool, groundedness, clarification, and escalation behavior match expectations. The results in the report are a manual smoke-test template, not a claim of statistically validated performance.

## Repository structure
```text
app/
  agent.py       intent routing, LangGraph flow, LLM prompt and fallback
  config.py      environment configuration
  knowledge.py   vector retrieval and lexical fallback
  tools.py       synthetic order/return/refund/escalation tools
  main.py        FastAPI endpoints
  static/        browser UI
data/            FAQ/policies and mock order/return records
evaluation/      labeled cases and sample conversation log
```

## Known limitations / future work
- Synthetic order records only; no real commerce or payment provider integration.
- Simple intent routing; production use requires stronger intent classification and monitoring.
- Embedding retrieval may require an initial model download.
- No authentication, per-customer authorization, durable database, or rate limiting; do not deploy with real customer data.
- Add human ticketing integration, multilingual evaluation, adversarial tests, PII redaction, and a real database before production use.
