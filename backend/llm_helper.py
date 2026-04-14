import os
import requests
import json
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMHelper:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL")
        self.model = os.getenv("OPENAI_MODEL", "mistral-large")
        
        if not self.api_key or not self.base_url:
            logger.warning("OPENAI_API_KEY or OPENAI_BASE_URL not found in .env. Falling back to defaults.")

    def chat_completion(self, system_prompt: str, user_message: str, history=None):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add history
        if history:
            for m in history[-10:]:
                # Normalize roles
                role = "assistant" if m["role"] == "tolstoy" else m["role"]
                messages.append({"role": role, "content": m["text"]})
        
        # Current message
        messages.append({"role": "user", "content": user_message})
        
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.5, # Slightly higher for more "live" dialogue
            "max_tokens": 2000
        }
        
        try:
            url = f"{self.base_url.rstrip('/')}/chat/completions"
            response = requests.post(url, headers=headers, json=body, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                logger.error(f"Mistral Proxy Error ({response.status_code}): {response.text}")
                return f"Извините, мои мысли затуманились... (Ошибка {response.status_code})"
        except Exception as e:
            logger.error(f"Mistral Request Exception: {e}")
            return f"Произошла ошибка в моих размышлениях: {str(e)}"

    def condense_query(self, user_text, history):
        """Rewrites contextual query into standalone search query."""
        if not history or len(history) == 0:
            return user_text
            
        history_str = "\n".join([f"{m['role']}: {m['text']}" for m in history[-5:]])
        system_prompt = (
            "Ты — интеллектуальный помощник. Твоя задача — переписать последний вопрос пользователя в самостоятельный поисковый запрос, "
            "используя контекст предыдущей беседы. Выдай ТОЛЬКО текст перефразированного вопроса без лишних слов."
        )
        user_message = f"История диалога:\n{history_str}\n\nПоследний вопрос пользователя: {user_text}\nПерепиши вопрос для поиска в базе знаний:"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.1,
            "max_tokens": 500
        }
        
        try:
            url = f"{self.base_url.rstrip('/')}/chat/completions"
            response = requests.post(url, headers=headers, json=body)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
        except:
            pass
        return user_text

    def evaluate_response(self, query: str, context: str, answer: str):
        """LLM-as-a-Judge using Mistral Large."""
        system_prompt = "Ты — независимый эксперт по оценке качества RAG-систем."
        user_message = (
            "Проанализируй случай:\n"
            f"ВОПРОС: {query}\n"
            f"КОНТЕКСТ: {context}\n"
            f"ОТВЕТ: {answer}\n\n"
            "Выдай JSON: relevance(1-5), faithfulness(1-5), root_cause, reason."
        )
        # Simplified for brevity in this helper
        return self.chat_completion(system_prompt, user_message)

    def grade_documents(self, query: str, documents: list):
        """Grades documents for relevance."""
        if not documents: return []
        
        results = []
        for doc in documents:
            prompt = f"ВОПРОС: {query}\nТЕКСТ: {doc}\nПолезен ли текст? Ответь 'YES' или 'NO'."
            res = self.chat_completion("Ты эксперт по анализу текстов. Отвечай строго YES или NO.", prompt)
            results.append(True if "YES" in res.upper() else False)
        return results
