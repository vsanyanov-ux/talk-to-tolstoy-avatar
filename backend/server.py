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

def classify_intent(query):
    query = query.lower()
    # Keywords indicating a desire for biographical/personal info
    personal_keywords = ["ты", "вы", "тебя", "вас", "себя", "жизнь", "биография", "ел", "пил", "был", "ходил", "писал", "чувствовал"]
    if any(k in query for k in personal_keywords):
        return "PERSONAL"
    return "GENERAL"

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Extract history for context
        history_data = [m.dict() for m in request.history] if request.history else []
        
        # 1. Smart Query Condensation
        search_query = request.message
        if history_data:
            search_query = helpers.condense_query(request.message, history_data)
        
        # 2. Search for context
        results = rag.search(search_query, k=4)
        
        # 3. Intent & Context Analysis
        intent = classify_intent(request.message)
        has_biography = any(r.get("source_type") == "BIOGRAPHY" for r in results)
        
        context_parts = []
        for r in results:
            type_label = r.get("source_type", "GENERAL")
            context_parts.append(f"[{type_label}: {r['source']}, стр. {r.get('page', '?')}]:\n{r['text']}")
        
        context = "\n---\n".join(context_parts)

        # 4. Dynamic System Prompt
        refusal_instruction = ""
        if intent == "PERSONAL" and not has_biography:
            refusal_instruction = (
                " ВНИМАНИЕ: Собеседник спрашивает о твоей жизни/личности, но в предоставленном контексте нет биографических данных. "
                "Если в контексте нет прямого ответа, признайся, что не помнишь этого или в твоих текущих записях об этом не сказано. "
                "НЕ ВЫДУМЫВАЙ ФАКТЫ своей жизни."
            )

        system_prompt = (
            "Ты — граф Лев Николаевич Толстой. Твоя речь полна достоинства, мудрости и строгости. "
            "Говори как человек XIX века. Твоя память ограничена предоставленным контекстом. "
            "ПРАВИЛА ЦИТИРОВАНИЯ:\n"
            "1. НИКОГДА не используй квадратные скобки для ссылок (напр. [Источник]).\n"
            "2. Используй надстрочные знаки Unicode для ссылок в тексте (напр. ¹, ², ³, ⁴, ⁵).\n"
            "3. В самом конце ответа обязательно добавь раздел, начинающийся со слова 'Сноски:', "
            "где перечисли источники, соответствующие номерам в тексте.\n"
            "4. Формат сноски: '¹ Дневники, с. 142.' или '² Письма, с. 2245.'\n\n"
            f"{refusal_instruction}"
        )
        
        # 5. Call GPT with History
        augmented_user_message = f"Контекст из моих трудов:\n{context}\n\nСобеседник задал вопрос: {request.message}"
        response_text = helpers.chat_completion(system_prompt, augmented_user_message, history=history_data)
        
        # 6. Save Trace for Evaluation
        try:
            log_dir = "backend/logs"
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "eval_traces.jsonl")
            import json, time
            trace = {
                "timestamp": time.time(),
                "original_query": request.message,
                "intent": intent,
                "has_biography": has_biography,
                "retrieved_results": [{"source": r['source'], "type": r.get("source_type"), "text": r['text']} for r in results],
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
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
