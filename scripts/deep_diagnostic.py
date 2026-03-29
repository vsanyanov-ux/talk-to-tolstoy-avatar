import sys
import os
import json

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from rag import SimpleRAG
from yandex_gpt import YandexGPTHelper

def test():
    print("Initializing RAG and Helper (NO RERANKER)...")
    rag = SimpleRAG(data_dir="Data/processed", index_file="backend/index.json", use_reranker=False)
    helpers = YandexGPTHelper()
    
    query = "А за границей, где ты бывал?"
    print(f"Testing Query: {query}")
    
    try:
        # Step 1: Search
        results = rag.search(query)
        print(f"Results Found: {len(results)}")
        for r in results:
            print(f" - [{r.get('type')}] {r['source']}")
            
        # Step 2: System Prompt Construction
        system_prompt = "Ты - Лев Толстой..."
        context = "\n---\n".join([f"[{r.get('type')}] {r['source']}:\n{r['text']}" for r in results])
        user_message = f"Контекст:\n{context}\n\nВопрос: {query}"
        
        # Step 3: Call GPT (Dry Run if needed, but let's go full)
        print("Calling Yandex GPT...")
        # (Skip actual API call if you want to be safe, but we need to see if it fails)
        # However, Errno 22 usually happens at file open.
        
        # LOGGING CHECK (Often the culprit)
        log_dir = "backend/logs"
        log_path = os.path.join(log_dir, "eval_traces_test.jsonl")
        print(f"Checking log path: {log_path}")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("test log\n")
        print("Logging works.")
        
    except Exception as e:
        print(f"DETAILED ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
