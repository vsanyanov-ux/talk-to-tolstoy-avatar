from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from yandex_gpt import YandexGPTHelper
from graph import app_graph
import os
import logging
import sys
import io

# Force UTF-8 encoding for Windows to prevent [Errno 22]
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Tolstoy Chat API")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from typing import List, Optional

class Message(BaseModel):
    role: str
    text: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Message]] = None

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Extract history for context
        history_data = [m.dict() for m in request.history] if request.history else []
        
        # 1. Initialize Graph State
        initial_state = {
            "query": request.message,
            "original_query": request.message,
            "history": history_data,
            "documents": [],
            "intent": "GENERAL",
            "response": "",
            "retry_count": 0
        }
        
        # 2. Run LangGraph Workflow
        final_state = app_graph.invoke(initial_state)
        response_text = final_state["response"]
        
        # 3. Save Trace for Evaluation
        try:
            log_dir = "backend/logs"
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "eval_traces.jsonl")
            import json, time
            trace = {
                "timestamp": time.time(),
                "original_query": request.message,
                "retrieved_results": [{"source": r['source'], "text": r['text']} for r in final_state["documents"]],
                "final_response": response_text,
                "workflow_meta": {
                    "retry_count": final_state["retry_count"]
                }
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace, ensure_ascii=False) + "\n")
        except Exception as log_err:
            logger.error(f"Failed to save traces: {log_err}")

        return {"response": response_text}
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
