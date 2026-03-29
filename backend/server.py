from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from yandex_gpt import YandexGPTHelper
from rag import SimpleRAG
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

# Initialize RAG and Helper
helpers = YandexGPTHelper()
rag = SimpleRAG()

# Pre-build index if not exists
try:
    rag.build_index()
except Exception as e:
    logger.error(f"Failed to build index: {e}")

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
        
        # 1. Smart Query Condensation
        # If there's history, we rewrite the user query to be standalone for RAG search
        search_query = request.message
        if history_data:
            search_query = helpers.condense_query(request.message, history_data)
        
        # 2. Search for context (using the condensed search query)
        results = rag.search(search_query, k=4)
        
        context_parts = []
        for r in results:
            context_parts.append(f"[{r['source']}, стр. {r.get('page', '?')}]:\n{r['text']}")
        
        context = "\n---\n".join(context_parts)

        # 3. Simple System Prompt
        system_prompt = (
            "Ты — граф Лев Николаевич Толстой. Твоя речь полна достоинства и мудрости. Говори как человек XIX века. "
            "Твоя память ограничена предоставленным контекстом. Если в нем нет ответа, признай это мудро. "
            "Цитируй источники в конце ответа (напр. 'Дневники, с. 12')."
        )
        
        # 4. Call GPT with History
        augmented_user_message = f"Контекст из моих трудов:\n{context}\n\nСобеседник задал вопрос: {request.message}"
        response_text = helpers.chat_completion(system_prompt, augmented_user_message, history=history_data)
        
        # 5. Save Trace for Evaluation (Observability)
        try:
            log_dir = "backend/logs"
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "eval_traces.jsonl")
            import json
            trace = {
                "timestamp": str(sys.modules['time'].time()) if 'time' in sys.modules else "0",
                "original_query": request.message,
                "search_query": search_query,
                "retrieved_results": [{"source": r['source'], "page": r.get('page','?'), "text": r['text']} for r in results],
                "final_response": response_text
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
