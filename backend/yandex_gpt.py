import os
import requests
import numpy as np
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YandexGPTHelper:
    def __init__(self):
        self.api_key = os.getenv("YANDEX_API_KEY")
        self.folder_id = os.getenv("YANDEX_FOLDER_ID")
        self.gpt_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        self.embed_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"

    def get_embedding(self, text: str):
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "x-folder-id": self.folder_id
        }
        body = {
            "modelUri": f"emb://{self.folder_id}/text-search-doc/latest",
            "text": text
        }
        response = requests.post(self.embed_url, headers=headers, json=body)
        if response.status_code == 200:
            return response.json()["embedding"]
        else:
            logger.error(f"Embeddings Error: {response.text}")
            return None

    def get_query_embedding(self, text: str):
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "x-folder-id": self.folder_id
        }
        body = {
            "modelUri": f"emb://{self.folder_id}/text-search-query/latest",
            "text": text
        }
        response = requests.post(self.embed_url, headers=headers, json=body)
        if response.status_code == 200:
            return response.json()["embedding"]
        else:
            logger.error(f"Query Embeddings Error: {response.text}")
            return None

    def chat_completion(self, system_prompt: str, user_message: str, history=None):
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "x-folder-id": self.folder_id
        }
        
        messages = [{"role": "system", "text": system_prompt}]
        
        # Add history if available
        if history:
            # Only keep last 10 messages for context window stability
            for m in history[-10:]:
                # Map 'tolstoy' role to 'assistant' for Yandex GPT compatibility
                role = "assistant" if m["role"] == "tolstoy" else m["role"]
                messages.append({"role": role, "text": m["text"]})
        
        # Current message
        messages.append({"role": "user", "text": user_message})
        
        body = {
            "modelUri": f"gpt://{self.folder_id}/yandexgpt/latest",
            "completionOptions": {
                "stream": False,
                "temperature": 0.3,
                "maxTokens": 2000
            },
            "messages": messages
        }
        response = requests.post(self.gpt_url, headers=headers, json=body)
        if response.status_code == 200:
            result = response.json()
            return result["result"]["alternatives"][0]["message"]["text"]
        else:
            logger.error(f"GPT Error: {response.text}")
            return f"Извините, произошла ошибка в моих размышлениях. ({response.status_code})"

    def condense_query(self, user_text, history):
        """Rewrites a contextual user message into a standalone search query."""
        if not history or len(history) == 0:
            return user_text
            
        history_str = "\n".join([f"{m['role']}: {m['text']}" for m in history[-5:]])
        system_prompt = (
            "Ты — интеллектуальный помощник. Твоя задача — переписать последний вопрос пользователя в самостоятельный поисковый запрос, "
            "используя контекст предыдущей беседы. Выдай ТОЛЬКО текст перефразированного вопроса без лишних слов."
        )
        user_message = f"История диалога:\n{history_str}\n\nПоследний вопрос пользователя: {user_text}\nПерепиши вопрос для поиска в базе знаний:"
        
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "x-folder-id": self.folder_id
        }
        body = {
            "modelUri": f"gpt://{self.folder_id}/yandexgpt/latest",
            "completionOptions": {"stream": False, "temperature": 0.1, "maxTokens": 500},
            "messages": [
                {"role": "system", "text": system_prompt},
                {"role": "user", "text": user_message}
            ]
        }
        response = requests.post(self.gpt_url, headers=headers, json=body)
        if response.status_code == 200:
            condensed = response.json()["result"]["alternatives"][0]["message"]["text"].strip()
            logger.info(f"Condensed query: '{user_text}' -> '{condensed}'")
            return condensed
        return user_text

    def grade_documents(self, query: str, documents: list):
        """Grades a list of documents for relevance to the user query."""
        if not documents:
            return []
            
        system_prompt = (
            "Ты — эксперт по анализу текстов. Твоя задача — оценить, содержит ли данный фрагмент текста "
            "полезную информацию для ответа на вопрос пользователя. "
            "Ответь строго 'YES', если фрагмент полезен, и 'NO', если нет. Не давай никаких пояснений."
        )
        
        results = []
        for doc in documents:
            user_message = f"ВОПРОС: {query}\n\nФРАГМЕНТ ТЕКСТА: {doc}\n\nПолезен ли этот текст?"
            
            headers = {
                "Authorization": f"Api-Key {self.api_key}",
                "x-folder-id": self.folder_id
            }
            body = {
                "modelUri": f"gpt://{self.folder_id}/yandexgpt/latest",
                "completionOptions": {"stream": False, "temperature": 0.1, "maxTokens": 10},
                "messages": [
                    {"role": "system", "text": system_prompt},
                    {"role": "user", "text": user_message}
                ]
            }
            try:
                response = requests.post(self.gpt_url, headers=headers, json=body)
                if response.status_code == 200:
                    text = response.json()["result"]["alternatives"][0]["message"]["text"].strip().upper()
                    results.append(True if "YES" in text else False)
                else:
                    results.append(True) # Fallback to keeping it
            except:
                results.append(True)
        return results

    def evaluate_response(self, query: str, context: str, answer: str):
        """Uses LLM-as-a-Judge to evaluate the quality of a RAG interaction."""
        system_prompt = (
            "Ты — независимый эксперт по оценке качества RAG-систем. Твоя задача — объективно оценить "
            "взаимодействие между пользователем и ИИ-аватаром. Используй шкалу 1-5."
        )
        
        user_message = (
            "Проанализируй следующий случай:\n\n"
            f"ВОПРОС ПОЛЬЗОВАТЕЛЯ: {query}\n"
            f"НАЙДЕННЫЙ КОНТЕКСТ: {context}\n"
            f"ОТВЕТ АВАТАРА: {answer}\n\n"
            "Выдай результат в формате JSON с полями:\n"
            "- relevance (насколько ответ соответствует сути вопроса)\n"
            "- faithfulness (опирается ли ответ ТОЛЬКО на контекст или галлюцинирует)\n"
            "- root_cause (в чем причина неудачи: 'missing_info' (в базе нет данных), 'poor_retrieval' (поиск нашел не то), 'hallucination' (ИИ проигнорировал контекст))\n"
            "- reason (краткое пояснение на русском языке)"
        )
        
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "x-folder-id": self.folder_id
        }
        body = {
            "modelUri": f"gpt://{self.folder_id}/yandexgpt/latest",
            "completionOptions": {"stream": False, "temperature": 0.1, "maxTokens": 1000},
            "messages": [
                {"role": "system", "text": system_prompt},
                {"role": "user", "text": user_message}
            ]
        }
        try:
            response = requests.post(self.gpt_url, headers=headers, json=body)
            if response.status_code == 200:
                return response.json()["result"]["alternatives"][0]["message"]["text"].strip()
            return f"{{\"error\": \"Eval API error: {response.text}\"}}"
        except Exception as e:
            return f"{{\"error\": \"Eval Exception: {e}\"}}"
