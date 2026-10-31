
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from app.agent import SupportAgent
from app.config import DATA_DIR

app = FastAPI(title="ShopMate E-Commerce Customer Support Assistant", version="1.0.0")
agent = SupportAgent(DATA_DIR)
STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[str] = Field(default_factory=list, max_length=20)
class ChatResponse(BaseModel):
    answer: str
    intent: str
    sources: list = []
    escalation: bool = False

@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")
@app.get("/api/health")
def health():
    return {"status":"ok", "llm_enabled": agent.llm is not None,
            "retrieval_mode":agent.kb.mode, "langgraph_enabled":agent.graph is not None}
@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = agent.answer(req.message, req.history)
    return ChatResponse(**result)
@app.get("/api/demo-orders")
def demo_orders():
    return {"examples":["ORD-1001","ORD-1002","ORD-1003","ORD-1004"]}
